# 1. Load Model
``` 
model = hmr(config.SMPL_MEAN_PARAMS)
checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()
```

## 1B. Resolve conflicts
Add the below code blocks **before** the above added code block
```
import torch
import config
from models import hmr
``` 
AND 
```
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
```
(Optional) Replace 
```
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
``` 
with 
```
parser.add_argument('--checkpoint', required=True, help='Path to network checkpoint')
``` 
(i.e. add `required=True` line).

# 2. Call Model
```
with torch.no_grad():
    pred_rotmat, pred_betas, pred_camera = model(images)
```
## Resolve Conflicts
Add:
```
images = batch['img']
```
In order to "resolve" `batch`, we'll need to define the dataloader (see step-3 for more details on this). But till then, let's build the barebones.

(A)Add below lines
```
for batch in demo_loader:
    images = batch['img']
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(images)
```
(B) Add below lines
```
from torch.utils.data import DataLoader
batch_size = 1
demo_loader = DataLoader(demo_dataset, batch_size=batch_size)
```

(C) Add below lines on top of the lines above
```
img_dir = "examples"
from datasets.demo_dataset import DemoDataset
demo_dataset = DemoDataset(img_dir)
```

# 3. Define DataLoader
We can either edit the existing dataloader or write one from scratch. If writing one from scratch, be careful about the input format expected by the model. See [this](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) for details.

## A. Outline-1
```
from torch.utils.data import Dataset

class DemoDataset(Dataset):
    def __init__(self):
        pass
    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass
```

## B. Outline-2
```
from torch.utils.data import Dataset
from glob import glob

class DemoDataset(Dataset):
    def __init__(self, img_dir):
        img_exts_allowed = ['jpg', 'png'] # What all img exts are considered valid?
        self.img_fps = []
        for img_ext in img_exts_allowed:
            self.img_fps+= glob(f"{img_dir}/*{img_ext}")
    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass
```

## C. Test till outline-2
Add below code lines
```
if __name__ == "__main__":
    img_dir = "../examples"
    demo_dataset = DemoDataset(img_dir)
```
and run `python demo_dataset.py` (after `cd datasets`)

## D. Outline-3 & It's Test
```
from torch.utils.data import Dataset
from glob import glob
import cv2

class DemoDataset(Dataset):
    def __init__(self, img_dir):
        img_exts_allowed = ['jpg', 'png'] # What all img exts are considered valid?
        self.img_fps = []
        for img_ext in img_exts_allowed:
            self.img_fps+= glob(f"{img_dir}/*{img_ext}")
    def __len__(self):
        return len(self.img_fps)
    def __getitem__(self, idx):
        img = cv2.imread(self.img_fps[idx])
        batch = {'img': img}
        return batch

if __name__ == "__main__":
    img_dir = "../examples"

    # Below lines are copied as-is from demo.py 
    demo_dataset = DemoDataset(img_dir)
    from torch.utils.data import DataLoader
    batch_size = 1
    demo_loader = DataLoader(demo_dataset, batch_size=batch_size)
    for batch in demo_loader:
        images = batch['img']
```