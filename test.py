import os
import torch
import json
from glob import glob
from PIL import Image
from pytorch3d.io import IO
import numpy as np
from src.utils import normalize_pc
from src.render_pc import render_pc_no_vis
from src.glip_inference import glip_inference_noLoad, load_model
from src.gen_superpoint import gen_superpoint
from src.bbox2seg import bbox2seg
from tqdm import tqdm

class ShapeNetParts(torch.utils.data.Dataset):
    metaData = json.load(open("../PartNetEnsemble/PartNetE_meta.json",'r'))
    def __init__(self, split):
        assert split in ['train', 'val', 'test']
        self.items = glob(f"../PartNetEnsemble/{split}/*/*")
    def __getitem__(self, index):
        path = self.items[index]

        #pointcloudImages = load_imgs(path)
        segmentation_labels = np.load(f"{path}/label.npy",allow_pickle=True).item()
        return {
            'class': path.split("/")[-2],
            'path': f"{path}/pc.ply",
            'segmentation_labels': segmentation_labels["semantic_seg"]
        }
    def __len__(self):
        return len(self.items)


dataset_test = ShapeNetParts("test")
print(dataset_test.metaData)
dataloader_test = torch.utils.data.dataloader.DataLoader(dataset_test,batch_size=1,shuffle=False)
io = IO()

    
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)


print("Device: ",device)

config ="GLIP/configs/glip_Swin_L.yaml"
weight_path = "models/glip_large_model.pth"
glip_demo = load_model(config, weight_path)


correct_p = 0
total_p = 0
for batch in tqdm(dataloader_test):
    xyz, rgb = normalize_pc(pc_file=batch["path"][0], save_dir="",io=io, device=device)
    img_list, pc_idx, screen_coords = render_pc_no_vis(xyz, rgb, device)
    preds = glip_inference_noLoad(glip_demo,img_list ,ShapeNetParts.metaData[batch["class"][0]])
    superpoint = gen_superpoint(xyz, rgb, visualize=False, save_dir="")
    sem_seg, ins_seg = bbox2seg(xyz, superpoint, preds, screen_coords, pc_idx, ShapeNetParts.metaData[batch["class"][0]], "test",solve_instance_seg=False,visualize=False)
    correct_p += (sem_seg == batch["segmentation_labels"][0].numpy()).sum()
    total_p += sem_seg.shape[0]

print("Correctly Labeled #Points: ",correct_p)
print("Total #Points: ",total_p)
print("Accuracy: %",100*correct_p/total_p)