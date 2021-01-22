import numpy as np
from ObjToRaster import ObjToRaster
import torch
from models import networks
import cv2

class BlendShapesDeltaToRaster:
    def __init__(self, objFilePath, shapesFilePath, res, areShapesDelta=False, device=None):
        self.device = "cuda:0" if device is None else device
        self.raster = ObjToRaster(objFilePath, res, device)
        self.shapes = np.load(shapesFilePath)
        if not areShapesDelta: # assume idx 0 is neutral
            self.shapes -= self.shapes[0]
            self.shapes = self.shapes[1:]
        # normalize to max = 0.9
        maxd = abs(self.shapes).max()
        self.deltaScale = 0.9 / maxd
        self.shapes *= self.deltaScale
        self.deltaScale = 1.0 / self.deltaScale
        s = self.shapes.shape
        if len(s) == 3:
            self.shapes = self.shapes.reshape(s[0], s[1]*3)
        self.nShapes = s[0]
        self.nPnts = s[1]//3
        self.shapes = torch.tensor(self.shapes, dtype=torch.float32).to(self.device)

    def rasterize(self, weights, scaleDelta=False):
        if type(weights) != torch.Tensor:
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.raster.rasterize(torch.matmul(weights, self.shapes).reshape((self.nPnts, 3))).permute(0,3,1,2)
            if scaleDelta:
                out *= self.deltaScale
            return out

class DeepFACS:
    def __init__(self, objFilePath, shapesFilePath, modelPath, res, areShapesDelta=False, device=None):
        self.device = "cuda:0" if device is None else device
        self.raster = BlendShapesDeltaToRaster(objFilePath, shapesFilePath, res, areShapesDelta, device)

        state_dict = torch.load(modelPath, map_location=device)
        self.generator = networks.define_G(3, 3, 64, 'unet_512', 'batch', False, 'normal', 0.02, [0]).module
        self.generator.load_state_dict(state_dict)

    def run(self, x):
        with torch.no_grad():
            y = self.raster.rasterize(x, False)
            z = self.generator(y)
            # cv2.imshow("window0", (y[0].permute(1,2,0)*0.5+0.5).detach().cpu().numpy())
            # cv2.imshow("window1", (z[0].permute(1,2,0)*0.5+0.5).detach().cpu().numpy())
            
            ## this is supposed to be multiplication, NOT division. a BUG in the training code??
            return z / self.raster.deltaScale 

# objPath = "F:\\nvidia\\stylegan3d\\data\\mark_deepfacs\\neutral_tri.obj"
# shapesPath = "F:\\nvidia\\deepfacs\\data\\rig_mark_hi_uvm_ict\\exp.npy"
# modelPath = "c:/users/jseo/nvidia/yay/pix2pix/checkpoints/deepfacs2/latest_net_G.pth"
# # meta = np.load("F:/nvidia/stylegan3d/data/mark_deepfacs/ict_rig/metadata.npy", allow_pickle=True).item()
# df = DeepFACS(objPath, shapesPath, modelPath, 512)
# w = np.zeros(53)
# w[0] = 1.0
# wmap = df.run(w)

# # bs = BlendShapesDeltaToRaster(objPath, shapesPath, 512)
# # a = bs.rasterize(np.ones(53))
