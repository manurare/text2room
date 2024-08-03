import argparse
import json
import os
from PIL import Image
import math
import numpy as np
import torch
from generate_scene import read_dpt
from model.trajectories.trajectory_util import _lemniscate
from model.mesh_fusion.util import car2sph, sph2erp 
from model.utils.utils import pil_to_torch
from model.utils.loss_utils import psnr, lpips_alex, ssim
from scipy.ndimage import map_coordinates
from tqdm import tqdm

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def sample_im(im, camera, persp_size, order=1):
    if im.ndim == 2:
        im = im[..., None]

    def fov2focal(fov, pixels):
        return pixels / (2 * math.tan(fov / 2))

    H,W,C = im.shape

    focal_x, focal_y = fov2focal(camera.FoVx, persp_size), fov2focal(camera.FoVy, persp_size)

    xx, yy = np.meshgrid(np.arange(persp_size), np.arange(persp_size), indexing="xy")
    points_cam = np.stack(((xx.flatten() - persp_size // 2) / focal_x, (yy.flatten() - persp_size // 2) / focal_y,
                            np.ones_like(xx.flatten())))
    
    c2w = camera.world_view_transform.transpose(1,0).inverse().cpu().numpy()
    points_world = c2w[:3, :3] @ points_cam + c2w[:3, [3]]

    points_world_sph = car2sph(points_world.T)
    erp_xx, erp_yy = sph2erp(points_world_sph[:, 0], points_world_sph[:, 1], H)
    erp_xx = erp_xx.reshape(persp_size, persp_size)
    erp_yy = erp_yy.reshape(persp_size, persp_size)

    persp_im = np.zeros((persp_size, persp_size, C))
    for c in range(C):
        persp_im[..., c] = map_coordinates(im[..., c], [erp_yy, erp_xx], order=order, mode='grid-wrap')

    return persp_im.squeeze(-1) if C==1 else persp_im

class Camera(torch.nn.Module):
    def __init__(self, R, T, FoVx, FoVy,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", **kwargs
                 ):
        super(Camera, self).__init__()

        self._R = R
        self._T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True, type=str)
args = parser.parse_args()

settings_file = os.path.join(args.experiment, "settings.json")
settings = json.load(open(settings_file, "r"))

input_rgb_fn = settings["input_image_path"]
input_depth_fn = settings["input_image_path"].replace("rgb_pano.jpg", "depth_pano.dpt")
rgb_path = settings["rgb_path"]
fov = np.radians(settings["fov"])

gts = []
preds = []
with torch.no_grad():
    for i in tqdm(range(1, 256), desc="reading data"):
        rgb = np.asarray(Image.open(input_rgb_fn.replace("0000", f"{i:04}"))) / 255.
        depth = read_dpt(input_depth_fn.replace("0000", f"{i:04}"))
        RT = _lemniscate(i-1).cpu().numpy()
        camera = Camera(RT[:3,:3].T, RT[:3, -1], FoVx=fov, FoVy=fov)
        im = sample_im(rgb, camera, 512)
        gts.append(torch.from_numpy(im).permute(2,0,1).type(torch.float32).clone().contiguous())

        pred = Image.open(os.path.join(rgb_path, f"rgb_{i:04}.png"))
        pred = pil_to_torch(pred, device="cpu").clone().contiguous()
        preds.append(pred)

    metrics = {"psnr": [], "ssim": [], "lpips": []}
    lpips_alex = lpips_alex.to("cuda")
    for gt, pred in tqdm(zip(gts, preds), desc="compute metrics"):
        gt = gt[None].to("cuda")
        pred = pred[None].to("cuda")
        metrics["psnr"].append(psnr(gt, pred).item())
        metrics["ssim"].append(ssim(gt, pred).item())
        metrics["lpips"].append(lpips_alex(gt, pred).item())

print({k:np.mean(v) for k,v in metrics.items()})



