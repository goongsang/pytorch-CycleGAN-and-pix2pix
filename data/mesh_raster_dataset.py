import os.path
from data.base_dataset import BaseDataset
import numpy as np
import torch
import torch.nn.functional as F
from libObj import libObj
import nvdiffrast.torch as dr
import imageio
import json
import shutil

class ObjToRaster:
    def __init__(self, objPath, res, device=None):
        self.res = res
        self.device = "cuda:0" if device is None else device
        self.glctx = dr.RasterizeGLContext()
        self.load(objPath)
        
    def load(self, objPath):
        self.obj = libObj(objPath)
        self.pnts = torch.tensor(self.obj.pnts, dtype=torch.float32).to(self.device)
        tmp = np.zeros((len(self.obj.uvs), 2))
        tmp[:,1] = 1.0
        uvs3d = np.hstack((self.obj.uvs*2.0-1.0, tmp))
        self.uvs3d = torch.tensor(uvs3d, dtype=torch.float32).unsqueeze(0).to(self.device)
        # self.uvs = torch.tensor(self.obj.uvs, dtype=torch.float32).to(self.device)
        self.uvTris = torch.tensor(np.array(self.obj.uvFaces, dtype=np.int32), dtype=torch.int32).to(self.device)
        self.tris = torch.tensor(np.array(self.obj.faces, dtype=np.int32), dtype=torch.int32).to(self.device)
        # FIXME: somehow first dr.rasterize call does not work and need to call twice
        self.rast, _ = dr.rasterize(self.glctx, self.uvs3d, self.uvTris, resolution=[self.res, self.res])
        self.rast, _ = dr.rasterize(self.glctx, self.uvs3d, self.uvTris, resolution=[self.res, self.res])

    def rasterize(self, pnts=None):
        if pnts is not None:
            if pnts.shape == self.pnts.shape:
                p = torch.tensor(pnts, dtype=torch.float32).to(self.device)
        else:
            p = self.pnts
        return dr.interpolate(p, self.rast, self.tris)[0]

    def dumpRast(self, rasts, filepath):
        # scale and flip vertically
        imgs = ((rasts + 1.0) * 128).cpu().numpy()[:, ::-1, :, :]
        imgs = np.clip(np.rint(imgs * 255), 0, 255).astype(np.uint8)
        for i, img in enumerate(imgs):
            imageio.imsave(filepath+"_%d.png"%(i), img)

# ### create training data from inputFrames/outputFrames from previous deepfacs data
# # it's not necessary to create ObjToRaster here, but just to check if nvdiffrast works..
# raster = ObjToRaster("F:/nvidia/stylegan3d/data/mark_deepfacs/neutral_tri.obj", 512)
# inFrames = np.load("F:/nvidia/deepfacs/cache/rig_mark_hi_uvm_ict_geom_cache_mark_hi/inputFrames.npy")
# outFrames = np.load("F:/nvidia/deepfacs/cache/rig_mark_hi_uvm_ict_geom_cache_mark_hi/outputFrames.npy")
# nFrames, nPnts = inFrames.shape
# nPnts //= 3
# inFrames = inFrames.reshape((nFrames, nPnts, 3))
# outFrames = outFrames.reshape((nFrames, nPnts, 3))
# inFrames -= raster.obj.pnts
# outFrames -= raster.obj.pnts
# dmax = max(abs(inFrames).max(), abs(outFrames).max())
# deltaScale = 0.9 / dmax
# inFrames *= deltaScale
# outFrames *= deltaScale
# # shuffle frames separate sets. don't create validation set. 
# shuffleFrames = np.arange(nFrames)
# np.random.shuffle(shuffleFrames)
# nValidFrames = 0
# nTestFrames = nFrames//10
# nTrainFrames = nFrames - (nValidFrames + nTestFrames)
# trainFrames = shuffleFrames[:nTrainFrames]
# validFrames = shuffleFrames[nTrainFrames:nTrainFrames+nValidFrames]
# testFrames = shuffleFrames[nTrainFrames+nValidFrames:]
# # save
# outdir = "F:/nvidia/stylegan3d/data/mark_deepfacs/ict_rig/"
# metadata = {'deltaScale':deltaScale, 'trainFrames':trainFrames, 'testFrames':testFrames, 'validFrames':validFrames}
# np.save(outdir+'metadata.npy', metadata)
# np.save(outdir+"train/input.npy", inFrames[trainFrames])
# np.save(outdir+"train/output.npy", outFrames[trainFrames])
# np.save(outdir+"test/input.npy", inFrames[testFrames])
# np.save(outdir+"test/output.npy", outFrames[testFrames])
# if nValidFrames > 0:
#     np.save(outdir+"valid/input.npy", inFrames[validFrames])
#     np.save(outdir+"valid/output.npy", outFrames[validFrames])
# shutil.copyfile("F:/nvidia/stylegan3d/data/mark_deepfacs/neutral_tri.obj", outdir+"neutral_tri.obj")

class MeshRasterDataset(BaseDataset):
    """A dataset class for paired image dataset. Images are rasterized on-the-fly from mesh data. 

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        res = int(opt.netG.split("_")[1])
        print("netG res", res)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the data directory
        self.rasterizer = ObjToRaster(os.path.join(opt.dataroot, "neutral_tri.obj"), res)
        print("Loading pntsA")
        self.pntsA = np.load(os.path.join(self.dir_AB, "input.npy"))
        print("Loading pntsB")
        self.pntsB = np.load(os.path.join(self.dir_AB, "output.npy"))
        if self.pntsA.shape[0] != self.pntsB.shape[0]:
            raise Exception("number of frames does not match between input and output")
        self.nFrames, self.nPnts, _ = self.pntsA.shape

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        A = self.rasterizer.rasterize(self.pntsA[index])[0].permute(2,0,1).detach()
        B = self.rasterizer.rasterize(self.pntsB[index])[0].permute(2,0,1).detach()
        AB_path = 'raster_idx_%06d' % index

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.nFrames
