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
from ObjToRaster import ObjToRaster

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
