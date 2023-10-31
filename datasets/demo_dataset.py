from torch.utils.data import Dataset
from glob import glob
import cv2
import numpy as np
from torchvision.transforms import Normalize
import torch

class DemoDataset(Dataset):
    def __init__(self, img_dir):
        img_exts_allowed = ['jpg', 'png'] # What all img exts are considered valid?
        self.img_fps = []
        for img_ext in img_exts_allowed:
            self.img_fps+= glob(f"{img_dir}/*{img_ext}")

        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

    def __len__(self):
        return len(self.img_fps)
    def __getitem__(self, idx):
        item = {}
        imgname = self.fps[idx]
        img = cv2.imread(imgname)[:, :, ::-1].copy().astype(np.float32) # Copied from base_dataset_eval.py's __get_item__()
        orig_shape = np.array(img.shape)[:2]
        img = self.rgb_processing(img, center, sc * scale, rot, flip, pn)
        img = torch.from_numpy(img).float()
        # Store image before normalization to use it in visualization
        item['img'] = self.normalize_img(img)

        return item

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale,
                       [constants.IMG_RES, constants.IMG_RES], rot=rot)
        # flip the image
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
        rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
        rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
        return rgb_img



if __name__ == "__main__":
    img_dir = "../examples"

    # Below lines are copied as-is from demo.py 
    demo_dataset = DemoDataset(img_dir)
    from torch.utils.data import DataLoader
    batch_size = 1
    demo_loader = DataLoader(demo_dataset, batch_size=batch_size)
    for batch in demo_loader:
        images = batch['img']
    