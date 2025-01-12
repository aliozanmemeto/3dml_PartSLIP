import sys
sys.path.insert(0,'../GroundingDINO')
sys.path.insert(0,'../SAM')
import os
import torch
import json
from pytorch3d.io import IO
import numpy as np
from src.utils import normalize_pc,save_colored_pc
from src.render_pc import render_pc
from src.gen_superpoint import gen_superpoint
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from torchvision.ops import nms
import json
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import distinctipy

def yolobbox2bbox(yolobox):
    x = yolobox[:,0]
    y = yolobox[:,1]
    w = yolobox[:,2]
    h = yolobox[:,3]
    xyxy = np.zeros_like(yolobox)
    xyxy[:,0] = x-w/2
    xyxy[:,1] = y-h/2
    xyxy[:,2] = x+w/2
    xyxy[:,3] = y+h/2
    return xyxy

def check_pc_within_bbox(x1, y1, x2, y2, pc):  
    flag = np.logical_and(pc[:, 0] > x1, pc[:, 0] < x2)
    flag = np.logical_and(flag, pc[:, 1] > y1)
    flag = np.logical_and(flag, pc[:, 1] < y2)
    return flag

def toDinoPrompt(metaData,className):
    listOfParts = metaData[className]
    prompt = ""
    partList = {}
    for i,part in enumerate(listOfParts):
        prompt += f"{className} {part}.".lower()
        partList[f"{className} {part}".lower()] = i
    return prompt,partList

def InferDINOSAMZeroShot(input_pc_file, category, modelDINO, predictorSAM, metaData, device,BOX_TRESHOLD = 0.2,
    TEXT_TRESHOLD = 0.3, SCORE_THRESHOLD=0.2, n_neighbors = 21, n_pass=3, save_dir="tmp"):
    
#     print("-----Zero-shot inference of %s-----" % input_pc_file)
    TEXT_PROMPT,partList = toDinoPrompt(metaData, category)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/rendered_img", exist_ok=True) #create the necessary save directories
    os.makedirs(f"{save_dir}/dino_pred", exist_ok=True)
    os.makedirs(f"{save_dir}/semantic_segDino_KNN", exist_ok=True)
    
    io = IO()
    xyz, rgb = normalize_pc(input_pc_file, save_dir, io, device) #read Point cloud and rgb in the format n,3
    img_dir, pc_idx, screen_coords = render_pc(xyz, rgb, save_dir, device) #create the rendered 2D images and return 
    # pc_idx = hxw where every pixel has a PC index correspondence   
    preds = []
    for i in range(pc_idx.shape[0]):
        image_source, image = load_image(f"{save_dir}/rendered_img/{i}.png") #load rgb images
        predictorSAM.set_image(image_source)
#         print("[dino inference...]")
        boxes, logits, phrases = predict(
                                        model=modelDINO,
                                        image=image,
                                        caption=TEXT_PROMPT,
                                        box_threshold=BOX_TRESHOLD,
                                        text_threshold=TEXT_TRESHOLD
                                    )
        phrases = np.array(phrases) #just to fix indexing

        xyxy = yolobbox2bbox(boxes)*image.shape[-1] #change bbox format to xyxy and scale with image size
        
        nms_mask = []
        for t,bbox in enumerate(xyxy): 
            if check_pc_within_bbox(bbox[0], bbox[1], bbox[2], bbox[3], screen_coords[i]).mean() < 0.95: 
                nms_mask.append(t)
        xyxy = xyxy[nms_mask]
        boxes = boxes[nms_mask]
        logits = logits[nms_mask]
        phrases = phrases[nms_mask]
        
        
        
        nms_indexes = nms(torch.tensor(xyxy) , logits, 0.5).numpy() #non maximum supression

        nms_mask = []
        for t,index in enumerate(nms_indexes):
            if phrases[index].lower() in partList.keys():
                nms_mask.append(t)
        nms_indexes = nms_indexes[nms_mask] #this is a temporary fix for DINO returning different classes that are not in the PROMPT
        # another fix is needed for this as this eleminates some important segments such as chair back as the phrase is not exact to PROMPT

        input_boxes = torch.tensor(xyxy[nms_indexes], device=predictorSAM.device)    
        transformed_boxes = predictorSAM.transform.apply_boxes_torch(input_boxes, image_source.shape[:2])
        

        if transformed_boxes.shape[0] > 0:
            masks, _, _ = predictorSAM.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )    #create segmentation masks with sam

            for index,j in enumerate(nms_indexes):
                preds.append({'image_id': i, 'category_id': phrases[j], 
                              'bbox': boxes[j]*image.shape[-1], 
                              'score': logits[j],
                              'mask':masks[index,0]   
                             }
                            )
#         annotated_frame = annotate(image_source=image_source, boxes=boxes[nms_indexes], logits=logits[nms_indexes], phrases=phrases[nms_indexes])
#         cv2.imwrite(f"{save_dir}/dino_pred/{i}.png", annotated_frame) #save an annotated image for DINO debugging
        
    pc_aggMask = torch.zeros((xyz.shape[0],len(partList)+1)) #this is a segment agg mask we sum all the scores from our bboxes 
    #into their own respective channel, the last channel is for unsegmented parts
    pc_aggMask[:,-1] = SCORE_THRESHOLD #we can set a confidence threshold by setting the unsegmented score
    for prediction in preds:
        maskedPC_idx = pc_idx[prediction["image_id"],prediction["mask"].cpu().numpy()] #this gives you the pc idx of the points that are inside the mask
        index_pcMasked = np.unique(maskedPC_idx)[1:] # we only need the unique idx and the first id is always -1 meaning not found
        pc_aggMask[index_pcMasked,partList[prediction["category_id"]]] += prediction["score"] #add up all the scores for each part
    pc_seg_classes = torch.argmax(pc_aggMask,dim=-1) #select the highest score as our segmentation class
    #if non of the part scores are over the SCORE_THRESHOLD it will be left unsegmented
    partColors = distinctipy.get_colors(len(partList))
    rgb_sem_merged = np.zeros((xyz.shape[0], 3))
    # since projections are not exact meaning not every PC point is rendered into our image our backprojections are not dense
    # use KNN to smooth these backprojections 
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(xyz) #create a knn
    
    results = {"partseg_rgbs":{}}
    for colorId,part in enumerate(partList):
        pc_part_idx = np.zeros((xyz.shape[0]),dtype=int)
        rgb_sem = np.zeros((xyz.shape[0],3))
        pc_part_idx[torch.where(pc_seg_classes==partList[part])] = 1
        
        for pass_ in range(n_pass):
            notColoredIndexes = torch.where(pc_seg_classes!=partList[part]) #find non segmented parts for smoothing

            n_indexes = nn.kneighbors(xyz[notColoredIndexes],n_neighbors+1,return_distance=False)
            n_indexes = n_indexes[:,1:] #get n_neighbors for the points, the first index is always the point itself so delete that
            #we have dense point clouds so distance based measures are not necessary and sometimes give worst results
            flag = pc_part_idx[n_indexes].mean(axis=1) 
            
            flag[np.where(flag>=0.4)] = 1 #and segmnent the points where the mean of neighbours are colored %40 or over
            flag[np.where(flag<0.4)] = 0
            pc_part_idx[notColoredIndexes] = flag
           
        rgb_sem[pc_part_idx.astype(bool)] = partColors[colorId]
        rgb_sem_merged += rgb_sem
        results["partseg_rgbs"][part] = rgb_sem
        save_colored_pc(f"{save_dir}/semantic_segDino_KNN/{part}.ply", xyz, rgb_sem)
        
    save_colored_pc(f"{save_dir}/semantic_segDino_KNN/{category}.ply", xyz, rgb_sem_merged)
    results["partList"] = partList
    results["xyz"] = xyz
    return results

if __name__ == "__main__":
    metaData = json.load(open("./PartNetE_meta.json")) 
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    modelDINO = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                       "../GroundingDINO/weights/groundingdino_swinb_cogcoor.pth",
                      device=device
                      )

    sam_checkpoint = "../SAM/weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"


    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda")

    predictorSAM = SamPredictor(sam)
    for category in ["Chair", "Suitcase", "Refrigerator", "Lamp", "Kettle"]: 
        input_pc_file = f"examples/{category}.ply"
        InferDINOSAMZeroShot(input_pc_file, category, modelDINO, predictorSAM, metaData, device, BOX_TRESHOLD = 0.2,
    TEXT_TRESHOLD = 0.3, SCORE_THRESHOLD=0.2, n_neighbors = 21, n_pass=5, save_dir=f'examples/zeroshot_{category}')