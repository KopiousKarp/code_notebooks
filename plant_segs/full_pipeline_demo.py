import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
from PIL import Image, ImageDraw
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
from PadSquare import PadSquare
import json
import os
import fnmatch
from plant_mask_measurements import find_rectangular_width, measure_root_mask, plot_measurements, aprox_count_and_width
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import cv2
import torch.nn as nn
preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')
model = smp.Unet('resnet34', encoder_weights='imagenet', in_channels=3, classes=4, activation=None)
model.load_state_dict(torch.load('/work/best_model.pth', map_location=device,weights_only=True))
model = model.to(device)
classifier = torch.load('/work/best_classifier.pth',weights_only=False)
def find_images(directory):
    image_files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if fnmatch.fnmatch(filename, '*.png') or fnmatch.fnmatch(filename, '*.jpg') or fnmatch.fnmatch(filename, '*.jpeg'):
                image_files.append(os.path.join(root, filename))
    return image_files

# predict file paths; from the sorting notebook
def predict_filepaths(
    job_dir,
    model_name,
    classifier_name, 
    model,           # Your trained UNET (or similar) model
    best_classifier, # Your trained logistic regression classifier
    device
):
    
    file_paths = find_images(job_dir)
    print(f"Predicting file paths in {job_dir} with {len(file_paths)} images")
    model.eval()
    
    transform = transforms.v2.Compose([
        PadSquare(0), 
        v2.Resize((512, 512)), 
        v2.ToImage()
        ])
    if job_dir.endswith('/'):
        job_dir = job_dir[:-1]
    json_file = f"{job_dir.split('/')[-1]}_{model_name}_{classifier_name}_{device}.json"
    print(f"Using json file: {json_file}")
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    video = []
    for index, path in enumerate(file_paths):
        if path in data:
            print(f"Skipping {path}")
            continue
        else:    
            print(f"Processing {path}")
            # 1) Load image
            try:
                image = Image.open(path).convert('RGB')
            except:
                print(f"Could not open {path}")
                continue                
            # 2) Apply transforms
            img_t = transform(image)
            img_pp = preprocess_input(img_t.permute(1, 2, 0).numpy()) 
            print(f"img_pp shape: {img_pp.shape}")
            img_pp = torch.from_numpy(img_pp).permute(2, 0, 1).float().unsqueeze(0)
            with torch.no_grad():
                # Extract features
                feats = model.encoder(img_pp.to(device))  # shape [B, C, H, W]
                feats_inter = feats[-1]
                feats_for_classifier = feats_inter.view(feats_inter.size(0), feats_inter.size(1), -1).mean(dim=-1).cpu().numpy()
                
                # 3) Make classifier prediction
                pred = best_classifier.predict(feats_for_classifier)[0]
                measurements = None
                pred = 1 #REMOVE THIS LINE TO USE CLASSIFIER PREDICTION
                if pred == 1:
                    #Proceed with decoder; might need a batch dimension added
                    # bottleneck = model.
                    outputs = model.decoder(feats)
                    mask = model.segmentation_head(outputs) 
                    # Apply softmax activation to convert logits to probabilities
                    mask = torch.nn.functional.softmax(mask, dim=1)
                    # Convert probability maps to binary masks
                    threshold = 0.5
                    binary_mask = (mask > threshold).float()
                    mask = binary_mask  # Replace the soft mask with binary mask
                    print(f"outputs shape: {mask.shape}")
                    mask = mask.squeeze(0)
                    # Convert the root mask to uint8 with values 0 and 255 for OpenCV compatibility
                    root_mask = (mask[2].cpu().numpy() * 255).astype(np.uint8)
                    # Pass Stalk mask for find_rectangular_width
                    stalk_bbox, width = find_rectangular_width(mask[3].cpu().numpy(), center=(250,250))
                    # Check if stalk_bbox is a single value and convert to expected format if needed
                    marker_bbox, marker_width = find_rectangular_width(mask[1].cpu().numpy(), center=None)
                    # Convert stalk_bbox from (x, y, width, height) to array of points
                    if isinstance(stalk_bbox, tuple) and len(stalk_bbox) == 4:
                        x, y, w, h = stalk_bbox
                        bbox_points = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
                    else:
                        bbox_points = stalk_bbox
                    if stalk_bbox is None:
                        print("No stalk found")
                        continue        
                    # Remove redundant call, only use root_mask which is already processed
                    measurements = measure_root_mask(root_mask, bbox_points)
                    measurements['marker_width'] = marker_width
                    measurements['stalk_width'] = width
                    # Convert img_pp from [1, 3, 512, 512] to [512, 512, 3]
                    img_np = img_pp.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    measurements["root_count"], measurements["root_width"] = aprox_count_and_width(root_mask, img_np, bbox_points)
                    print(measurements)
                              
                    # Retrieve frame from plot_measurements and save to video
                    frame = plot_measurements(img_t, bbox_points, measurements,mask, return_image=True)
                    video.append(frame)
                print(f"index: {index} Predicted {path} as {pred}")
                if measurements is None:
                    data[path] = {"measurable":int(pred), "measurements":None}
                else:
                    data[path] = {"measurable":int(pred), "measurements":measurements}# Save results to json             
        if index % 100 == 0:
            with open(json_file, 'w') as f:
                json.dump(data, f)
    # Save the frames as a video
    if video:
        fps = 1  # 1 frame per second (each frame displayed for 1 second)
        output_path = os.path.join(job_dir, f"{job_dir.split('/')[-1]}_{model_name}_{classifier_name}_results.mp4")
        
        # Get dimensions from the first frame
        height, width = np.array(video[0]).shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Creating video with {len(video)} frames at {fps} fps...")
        
        # Convert PIL images to OpenCV format and write to video
        for frame in tqdm(video):
            # Convert PIL Image to numpy array and from RGB to BGR (for OpenCV)
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out.write(frame_cv)
        
        # Release the video writer
        out.release()
        print(f"Video saved to {output_path}")
    else:
        print("No frames to save as video.")

    # Save final data to json
    with open(json_file, 'w') as f:
        json.dump(data, f)


predict_filepaths(
    "/opt/RootTaggingGUI/stalk_images/",
    "UNET",
    "logistic_regression", 
    model,           # trained UNET (or similar) model
    classifier, # trained logistic regression classifier
    device
)