from demo import Infer
from demoDINO import InferDINOSAMZeroShot

def get_files_from_txt(file_path, categories, num_files):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    parsed_data = [(int(line.split()[0]), line.split()[1]) for line in lines]
    
    results = {}
    for category in categories:
        filtered_data = [f"data/{category}/{item[0]}" for item in parsed_data if item[1] == category]
        results[category] = filtered_data[:num_files]
    
    return results

def mIOU(label, preds):
    semantic_seg = np.array(label.item()['semantic_seg'])
    unique_parts = np.unique(semantic_seg)
    unique_parts = unique_parts[unique_parts != -1]
    partList = preds['partList']
    pred_partseg_rgbs = preds["partseg_rgbs"]

    ious = []

    for part, part_id in partList.items():
        if part_id not in unique_parts:
            # Part is not present in ground truth, skipping IoU calculation.
            continue

        gt_mask = semantic_seg == part_id

        pred_mask = np.any(pred_partseg_rgbs[part] != [0., 0., 0.], axis=-1)

        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()

        iou = intersection / union if union > 0 else 0
        ious.append(iou)

        print(f"IoU for {part}: {iou:.4f}")

    if ious:
        mean_iou = np.mean(ious)
        print(f"Mean IoU: {mean_iou:.4f}")
    else:
        print("No valid parts found in the ground truth. Mean IoU cannot be calculated.")
    return (ious, mean_iou)

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
    
    categories = ["Chair", "Suitcase", "Refrigerator", "Lamp", "Kettle"]
    
    files = get_files_from_txt("test_files.txt", categories, 10)
    for category, file in files.items():
        print("DINO INFERENCE, " + category + ":")
        preds = InferDINOSAMZeroShot(file + "/pc.ply", category, modelDINO, predictorSAM, metaData, device, BOX_TRESHOLD = 0.2,
    TEXT_TRESHOLD = 0.3, SCORE_THRESHOLD=0.2, n_neighbors = 21, n_pass=5, save_dir=f'examples/zeroshot_{category}')
        print("mIOU CALCULATION, " + category + ":")
        label = np.load(file + "/label.npy", allow_pickle=True)
        mIOU, partIOUs = mIOU(label, preds)
        
        