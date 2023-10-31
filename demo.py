""" STEP-1 BELOW """
import torch
import config
from models import hmr

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='Path to network checkpoint')
args = parser.parse_args()

model = hmr(config.SMPL_MEAN_PARAMS)
checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()
""" STEP-1 ABOVE """

""" STEP-2 BELOW """
img_dir = "examples"
from datasets.demo_dataset import DemoDataset
demo_dataset = DemoDataset(img_dir)

from torch.utils.data import DataLoader
batch_size = 1
demo_loader = DataLoader(demo_dataset, batch_size=batch_size)

for batch in demo_loader:
    images = batch['img']
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(images)
""" STEP-2 ABOVE """

