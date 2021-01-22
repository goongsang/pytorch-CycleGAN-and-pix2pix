import os.path
import numpy as np
import torch
import torch.nn.functional as F
from libObj import libObj
import nvdiffrast.torch as dr
import imageio

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
                if type(pnts) == torch.Tensor:
                    p = pnts.clone().detach().to(self.device)
                else:
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
