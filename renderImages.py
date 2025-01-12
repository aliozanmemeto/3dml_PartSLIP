import os
import torch
import json
from glob import glob
from PIL import Image
from pytorch3d.io import IO
import numpy as np
from src.utils import normalize_pc
from src.render_pc import render_pc
from src.glip_inference import glip_inference_noLoad, load_model
from src.gen_superpoint import gen_superpoint
from src.bbox2seg import bbox2seg
from tqdm import tqdm


def load_imgs(file_name):
        pil_images = []

        for i in range(6):
            pil_images.append(np.array(Image.open(f"{file_name}/images/{i}.png").convert("RGB"))[:, :, [2, 1, 0]])
        return pil_images

class ShapeNetParts(torch.utils.data.Dataset):
    metaData = json.load(open("./PartNetE_meta.json",'r'))
    def __init__(self, split):
        assert split in ['train', 'val', 'test']
        self.items = glob(f"./data/*/*")
    def __getitem__(self, index):
        path = self.items[index]

        #pointcloudImages = load_imgs(path)
        segmentation_labels = np.load(f"{path}/label.npy",allow_pickle=True).item()
        return {
            'class': path.split("/")[-2],
            'path': path,
            'segmentation_labels': segmentation_labels["semantic_seg"]
        }
    def __len__(self):
        return len(self.items)
    
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
dataset_test = ShapeNetParts("test")
dataloader_test = torch.utils.data.dataloader.DataLoader(dataset_test,batch_size=1,shuffle=False)
io = IO()

for batch in tqdm(dataloader_test):
    directoryPath = batch["path"][0]
    xyz, rgb = normalize_pc(pc_file=f"{directoryPath}/pc.ply", save_dir="",io=io, device=device)
    img_list, pc_idx, screen_coords = render_pc(xyz, rgb,directoryPath, device)
    np.save(f"{directoryPath}/pc_idx.npy",pc_idx)
    np.save(f"{directoryPath}/screen_coords.npy",screen_coords)
    
 