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

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True, type=str)
args = parser.parse_args()

settings_file = os.path.join(args.experiment, "settings.json")
settings = json.load(open(settings_file, "r"))

input_rgb_files = os.path.join(os.path.dirname(settings["input_image_path"]), "{:05}_rgb.png")
input_depth_files = os.path.join(os.path.dirname(settings["input_image_path"]), "{:05}_depth.dpt")
rgb_path = settings["rgb_path"]

gts = []
preds = []
with torch.no_grad():
    for i in tqdm(range(0, 255), desc="reading data"):
        rgb = np.asarray(Image.open(input_rgb_files.format(i))) / 255.
        depth = read_dpt(input_depth_files.format(i))
        Image.fromarray((rgb*255).astype(np.uint8)).save(os.path.join(rgb_path, f"rgb_{i:04}_gt.png"))
        gts.append(torch.from_numpy(rgb).permute(2,0,1).type(torch.float32).clone().contiguous())

        pred = Image.open(os.path.join(rgb_path, f"rgb_{i+1:04}.png"))
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

means = {k:np.mean(v) for k,v in metrics.items()}
print(",".join(["exp_name"] + [*means.keys()]))
print(",".join(map(str, [*means.values()])))



