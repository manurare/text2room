import argparse
import json
import os
from PIL import Image
import glob
import numpy as np
import torch
from generate_scene import read_dpt
from model.trajectories.trajectory_util import _lemniscate
from model.mesh_fusion.util import car2sph, sph2erp 
from model.utils.utils import pil_to_torch
from model.utils.loss_utils import psnr, lpips_alex, ssim, delta_inlier_ratio, abs_rel_error, lin_rms_sq_error
from scipy.ndimage import map_coordinates
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--exp_renders_dir", required=True, default=None)
parser.add_argument("--gt_renders_dir", required=True, default=None)
args = parser.parse_args()

settings_file = os.path.join(args.experiment, "settings.json")
settings = json.load(open(settings_file, "r"))

files = glob.glob(os.path.join(args.exp_renders_dir, "depth", "depth_*.npy"))
numfiles = len(files)

pred_rgb_files_format = os.path.join(args.exp_renders_dir, "rgb", "rgb_{:04}.png")
pred_depth_files_format = os.path.join(args.exp_renders_dir, "depth", "depth_{:04}.npy")
gt_rgb_files_format = os.path.join(args.gt_renders_dir, "{:05}_rgb.png")
gt_depth_files_format = os.path.join(args.gt_renders_dir, "{:05}_depth.dpt")

metrics = {"psnr": [], "ssim": [], "lpips": [], "absrel": [], "rms": [], "delta1": [], "delta2": [], "delta3": []}

with torch.no_grad():
    for i in tqdm.tqdm(range(numfiles)):
        pred_rgb = np.array(Image.open(pred_rgb_files_format.format(i))) / 255.
        gt_rgb = np.array(Image.open(gt_rgb_files_format.format(i))) / 255.
        pred_rgb = torch.from_numpy(pred_rgb).permute(2, 0, 1).type(torch.float32).contiguous().to("cuda")[None]
        gt_rgb = torch.from_numpy(gt_rgb).permute(2, 0, 1).type(torch.float32).contiguous().to("cuda")[None]

        psnr_ = psnr(pred_rgb, gt_rgb).cpu().item()
        ssim_ = ssim(pred_rgb, gt_rgb).cpu().item()
        lpips_ = lpips_alex(pred_rgb, gt_rgb).cpu().item()

        # Depth
        pred_depth = np.load(pred_depth_files_format.format(i))
        gt_depth = read_dpt(gt_depth_files_format.format(i))

        mask = gt_depth > 0

        abs_rel = abs_rel_error(pred_depth, gt_depth, mask)
        rms = lin_rms_sq_error(pred_depth, gt_depth, mask)
        delta1 = delta_inlier_ratio(pred_depth, gt_depth, mask, 1)
        delta2 = delta_inlier_ratio(pred_depth, gt_depth, mask, 2)
        delta3 = delta_inlier_ratio(pred_depth, gt_depth, mask, 3)

        metrics["psnr"].append(psnr_)
        metrics["ssim"].append(ssim_)
        metrics["lpips"].append(lpips_)
        metrics["absrel"].append(abs_rel)
        metrics["rms"].append(rms)
        metrics["delta1"].append(delta1)
        metrics["delta2"].append(delta2)
        metrics["delta3"].append(delta3)

means = {k:np.mean(v) for k,v in metrics.items()}
print(",".join(["exp_name"] + [*means.keys()]))
print(",".join(map(str, [*means.values()])))



