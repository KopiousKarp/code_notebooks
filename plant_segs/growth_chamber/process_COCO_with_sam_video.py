import torch
from torchvision import datasets, transforms
from torchvision.transforms import v2
import argparse
import json
import os
import numpy as np
from pycocotools.coco import COCO
from sam2.build_sam import build_sam2_video_predictor
import re
from datetime import datetime
import tempfile
import shutil

def load_coco_data(coco_root, coco_annFile):
    """
    Load COCO dataset with annotations.
    
    Args:
        coco_root: Path to the directory with images
        coco_annFile: Path to the COCO JSON annotation file
    
    Returns:
        dataset: COCO dataset with images and annotations
    """
    # Define minimal transform that preserves original dimensions
    minimal_transform = transforms.v2.Compose([
        v2.ToTensor()  # Convert to tensor
    ])
    
    # Load COCO dataset
    dataset = datasets.CocoDetection(root=coco_root, annFile=coco_annFile, transforms=minimal_transform)
    
    # Wrap dataset for handling masks properly
    dataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=("labels", "masks"))
    
    # Also load raw COCO API for direct access to annotations
    coco_api = COCO(coco_annFile)
    
    return dataset, coco_api

def process_with_sam2(coco_root, coco_api, sam2_checkpoint, model_cfg, device="cuda"):
    """
    Process COCO masks with SAM2 video predictor
    
    Args:
        coco_root: Path to images directory
        coco_api: COCO API for direct access to annotations
        sam2_checkpoint: Path to SAM2 checkpoint file
        model_cfg: SAM2 model configuration file
        device: Device to run inference on
    
    Returns:
        Dictionary mapping frame indices to object masks
    """
    # Build SAM2 video predictor
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cpu")
    
    # NOT SURE I NEED THIS
    # images = coco_api.loadImgs(img_ids)
    
    # Get frame names from the coco_root directory
    frame_names = {}
    file_names = [f for f in os.listdir(coco_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # Sort filenames chronologically based on timestamp in the filename (topview_YYYY-MM-DD_HH_MM_SS.jpg)

    # Extract timestamp from filename and use it for sorting
    timestamps_with_files = []
    for filename in file_names:
        # Extract timestamp pattern from filename
        match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})', filename)
        if match:
            timestamp_str = match.group(1)
            # Parse timestamp
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d_%H_%M_%S')
            timestamps_with_files.append((filename, timestamp))

    # Sort by timestamp
    timestamps_with_files.sort(key=lambda x: x[1])

    # Extract filenames in timestamp order
    frame_names = {filename: idx for idx, (filename, _) in enumerate(timestamps_with_files)}

    # Print first 10 items (or fewer if there are less than 10)
    for i, (filename, idx) in enumerate(list(frame_names.items())[:10]):
        print(f"{filename}: {idx}")

    # condition_frame_number = frame_names[coco_api.loadImgs(coco_api.getImgIds())[0]['file_name']]
    # print(f"Condition frame number: {condition_frame_number}")    

    temp_video_dir = tempfile.mkdtemp() 
    for i, (filename, _) in enumerate(timestamps_with_files):
        shutil.copy(os.path.join(coco_root, filename), os.path.join(temp_video_dir, f"{i:04d}.jpg"))
    print(f"Created temporary video directory at: {temp_video_dir}")

    # Initialize SAM2 state
    inference_state = predictor.init_state(video_path=temp_video_dir,
                                           offload_video_to_cpu=True,
                                           offload_state_to_cpu=True)
    

    print(f"Initialized SAM2 state with video at: {temp_video_dir}")
    # Process each annotation and add to SAM2
    img_ids = coco_api.getImgIds()
    for img_id in img_ids:
        # filename = frame_names[coco_api.loadImgs(img_id)['file_name']]
        # filename = coco_api.loadImgs(img_id['file_name'])
        
        # Skip if file not found in frame_names
        if filename not in frame_names:
            continue
            
        cond_frame_idx = frame_names[coco_api.loadImgs(img_id)[0]['file_name']]
        
        # Get annotations for this image
        ann_ids = coco_api.getAnnIds(imgIds=img_id)
        anns = coco_api.loadAnns(ann_ids)
        
        for ann in anns:
            # Get mask from annotation
            mask = coco_api.annToMask(ann)
            
            # Convert to torch tensor
            mask_tensor = torch.from_numpy(mask).float().to(device)
            
            # Add mask to SAM2 predictor
            _, out_obj_ids, _ = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=cond_frame_idx,
                obj_id=ann['id'],  # Use annotation ID as object ID
                mask=mask_tensor
            )
    
    # Propagate masks through all frames
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    return video_segments, frame_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process COCO annotations for SAM2 video prediction')
    parser.add_argument('--coco_root', type=str, required=True, help='Path to images directory')
    parser.add_argument('--coco_annFile', type=str, required=True, help='Path to COCO JSON annotation file')
    parser.add_argument('--sam2_checkpoint', type=str, default="/opt/sam2/checkpoints/sam2.1_hiera_tiny.pt", 
                        help='Path to SAM2 checkpoint file')
    parser.add_argument('--model_cfg', type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml", 
                        help='SAM2 model configuration file')
    parser.add_argument('--device', type=str, default="cuda", help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Load COCO dataset and API
    dataset, coco_api = load_coco_data(args.coco_root, args.coco_annFile)
    
    print(f"Successfully loaded COCO dataset with {len(dataset)} images")
    
    # Process masks with SAM2 video predictor
    video_segments, frame_names = process_with_sam2(
        args.coco_root, 
        coco_api, 
        args.sam2_checkpoint, 
        args.model_cfg, 
        args.device
    )
    
    ### TODO: add the masks to the annotations in the COCO format
    ### and save the updated COCO JSON file
    new_annotations = []
    for frame_idx, objs in video_segments.items():
        ann = {
            'id': 1000,  # Unique annotation ID
            'image_id': 1,  # ID of the image
            'category_id': 1,  # ID of the category
            'bbox': [20, 30, 50, 70],  # Bounding box [x, y, width, height]
            'area': 50 * 70,
            'iscrowd': 0
        }
    

    print(f"Successfully processed {len(video_segments)} frames with SAM2")
    print(f"Tracked {sum(len(objs) for objs in video_segments.values())} objects across all frames")