import torch
from torch.utils.data import DataLoader, Subset, random_split # Added random_split
from torchvision import datasets, transforms
from torchvision.transforms import v2
# from sklearn.model_selection import KFold # Removed KFold
# from sklearn.model_selection import train_test_split # Using torch.utils.data.random_split instead
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch.nn as nn
from tqdm import tqdm as tdqm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
from PadSquare import PadSquare # Assuming PadSquare.py is in the same directory or PYTHONPATH
import time
import datetime
import json
import os
import pprint
import argparse

# Get the current time and store it to track execution time
start_time = time.time()
start_datetime = datetime.datetime.now()
print(f"Started execution at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

augmentations = transforms.v2.Compose([
    transforms.v2.RandomPerspective(distortion_scale=0.25, p=0.5),
    transforms.v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.v2.RandomHorizontalFlip(p=0.5),
    transforms.v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.v2.RandomRotation(degrees=5)
])

def custom_collate_fn(batch):
    # Unpack the batch - assuming each item is a (image, mask) tuple
    images, masks = zip(*batch)
    mask_batch = []
    for mask in masks:
        multiclass_mask = torch.zeros((4, mask["masks"].shape[1], mask["masks"].shape[2]), dtype=mask["masks"].dtype)
        for idx, label in enumerate(mask["labels"]):
            multiclass_mask[label] += mask["masks"][idx]
        multiclass_mask[0] = 1 - torch.max(multiclass_mask[1:], dim=0)[0]
        mask_batch.append(multiclass_mask)

    return images, mask_batch


# Preprocessing fn for the encoder
preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')


# Add argument parsing
parser = argparse.ArgumentParser(description="Train a U-Net model with parameter sweep.")
parser.add_argument('--coco_roots', nargs='+', required=True, help='List of COCO dataset root directories.')
parser.add_argument('--coco_annFiles', nargs='+', required=True, help='List of COCO annotation files.')
parser.add_argument('--learning_rates', nargs='+', type=float, default=[1e-4], help='List of learning rates to test.')
parser.add_argument('--weight_decays', nargs='+', type=float, default=[1e-5], help='List of weight decays to test.')
parser.add_argument('--train_split_ratio', type=float, default=0.8, help='Ratio of data to use for training (0.0 to 1.0).')
parser.add_argument('--epochs', type=int, default=25, help='Number of epochs for training each parameter combination.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation.')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader.')
parser.add_argument('--no_augmentations', action='store_false', default=True, help='Whether to apply augmentations to the training set.')

args = parser.parse_args()

coco_roots = args.coco_roots
coco_annFiles = args.coco_annFiles

# Standard transform for COCO dataset loading (base processing)
# This is applied to (image, target) by CocoDetection if target_keys are wrapped.
standard_transform = transforms.v2.Compose([
    PadSquare(padding_mode='symmetric'),
    v2.Resize((512, 512)), 
    v2.ToImage(), # Converts PIL/NumPy to tensor, CHW format
    v2.ToDtype(torch.float32, scale=True) # Normalize to [0,1] if input is uint8 & convert type
])

# Load full dataset
all_datasets = []
for coco_root, coco_annFile in zip(coco_roots, coco_annFiles):
    # Pass standard_transform to CocoDetection's `transforms` argument.
    # This will apply it to (image, target) pairs.
    seasonal_dataset = datasets.CocoDetection(root=coco_root, annFile=coco_annFile, transforms=standard_transform)
    # Wrap to ensure target keys like "masks" and "labels" are also transformed (e.g., resized).
    seasonal_dataset = datasets.wrap_dataset_for_transforms_v2(seasonal_dataset, target_keys=("labels", "masks"))
    all_datasets.append(seasonal_dataset)

if not all_datasets:
    raise ValueError("No datasets were loaded. Check COCO paths and annotation files.")

if len(all_datasets) == 1:
    full_dataset = all_datasets[0]
else:
    full_dataset = torch.utils.data.ConcatDataset(all_datasets)

print(f"Full dataset size: {len(full_dataset)}")

# Split dataset into training and validation
dataset_size = len(full_dataset)
if not (0 < args.train_split_ratio < 1):
    raise ValueError("train_split_ratio must be between 0 and 1 (exclusive of 0 and 1 for two splits).")
train_size = int(dataset_size * args.train_split_ratio)
val_size = dataset_size - train_size

if train_size == 0 or val_size == 0:
    raise ValueError(
        f"Train size ({train_size}) or validation size ({val_size}) is zero. "
        f"Adjust train_split_ratio or dataset size. Dataset size: {dataset_size}"
    )

generator = torch.Generator().manual_seed(69) # For reproducible split
train_subset_base, val_subset_base = random_split(full_dataset, [train_size, val_size], generator=generator)

print(f"Training subset size: {len(train_subset_base)}")
print(f"Validation subset size: {len(val_subset_base)}")
if args.no_augmentations:
    train_subset_base.transforms = transforms.v2.Compose([
            standard_transform,
            augmentations
        ])
else:
    print("Augmentations are disabled. Using only standard transforms for training.")
    train_subset_base.transforms = standard_transform
val_subset_base.transforms = standard_transform

# Create DataLoaders
train_dl = DataLoader(
    train_subset_base, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, collate_fn=custom_collate_fn, pin_memory=torch.cuda.is_available()
)
val_dl = DataLoader(
    val_subset_base, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, collate_fn=custom_collate_fn, pin_memory=torch.cuda.is_available()
)

# Calculate class weights for weighted cross entropy (once, on the training set)
print("Calculating class imbalances for the training set...")
class_counts = torch.zeros(4, device='cpu') 
# Iterate through train_dl once to get class counts from collated masks
for _, masks_batch in tdqm(train_dl, desc="Calculating class weights"): # masks_batch is a list of multiclass_mask tensors
    for mask_tensor in masks_batch: # mask_tensor is (4, H, W)
        if mask_tensor.device != class_counts.device: # Ensure devices match for sum
            mask_tensor = mask_tensor.to(class_counts.device)
        for i in range(4): # Iterate over classes
            class_counts[i] += mask_tensor[i].sum()

if class_counts.sum() == 0:
    print("Warning: No class pixels found in the training set based on collated masks. Using equal weights.")
    class_weights = (torch.ones(4) / 4).to(device)
else:
    class_proportions = class_counts / class_counts.sum()
    print(f"Class proportions (0,1,2,3): {class_proportions.numpy()}")
    # Inverse weighting: weight = 1 - proportion or 1 / proportion
    # Using 1 - proportion, then normalize or clamp
    class_weights_cpu = 1.0 - class_proportions
    class_weights_cpu = torch.clamp(class_weights_cpu, min=1e-6) # Avoid zero weights
    # class_weights_cpu /= class_weights_cpu.sum() # Optional: normalize weights to sum to 1
    class_weights = class_weights_cpu.float().to(device)
print(f"Using class weights for loss: {class_weights.cpu().numpy()}")


param_sweep_results = []
best_overall_val_loss = float('inf')
best_model_state_dict = None
best_params = {}
best_model_loss_curves = {'train': [], 'val': []}
for lr in args.learning_rates:
    for wd in args.weight_decays:
        print(f"\n===== Training with LR={lr}, WD={wd} =====")

        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=4,
            decoder_channels=(256,128,64,32,16)
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        current_best_val_loss_for_params = float('inf')
        current_best_state_for_params = None
        # Initialize structures to track loss curves for all models
        all_loss_curves = {'train': [], 'val': []}
        
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for imgs_tuple, masks_batch_list in tdqm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} Train LR={lr} WD={wd}"):
                # imgs_tuple: batch of image tensors (CHW, float, [0,1]) from TransformedDataset
                # masks_batch_list: list of [4, H, W] tensors from custom_collate_fn
                
                processed_imgs_list = []
                for img_tensor in imgs_tuple:
                    # Ensure img_tensor is on CPU for numpy conversion if it's not already
                    img_np_hwc = img_tensor.permute(1,2,0).cpu().numpy()
                    preprocessed_np = preprocess_input(img_np_hwc) # SMP preprocess
                    processed_imgs_list.append(torch.tensor(preprocessed_np))
                
                # Stack preprocessed images and move to device
                # preprocess_input output is HWC, permute to CHW for model
                processed_imgs = torch.stack(processed_imgs_list).permute(0,3,1,2).float().to(device)
                masks_stacked = torch.stack(masks_batch_list).to(device) # Batch, 4, H, W

                preds = model(processed_imgs) # Model output (B, C, H, W)
                # CrossEntropyLoss expects target (B,C,H,W) to be float for soft labels
                loss = criterion(preds, masks_stacked.float()) 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_dl)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs_tuple, masks_batch_list in tdqm(val_dl, desc=f"Epoch {epoch+1}/{args.epochs} Val   LR={lr} WD={wd}"):
                    processed_imgs_list = []
                    for img_tensor in imgs_tuple:
                        img_np_hwc = img_tensor.permute(1,2,0).cpu().numpy()
                        preprocessed_np = preprocess_input(img_np_hwc)
                        processed_imgs_list.append(torch.tensor(preprocessed_np))

                    processed_imgs = torch.stack(processed_imgs_list).permute(0,3,1,2).float().to(device)
                    masks_stacked = torch.stack(masks_batch_list).to(device)

                    preds = model(processed_imgs)
                    val_loss += criterion(preds, masks_stacked.float()).item()
            val_loss /= len(val_dl)

            print(f"LR={lr}, WD={wd} - Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            all_loss_curves['train'].append(train_loss)
            all_loss_curves['val'].append(val_loss)

            if val_loss < current_best_val_loss_for_params:
                current_best_val_loss_for_params = val_loss
                current_best_state_for_params = model.state_dict().copy()
                best_model_loss_curves = all_loss_curves.copy() # Save best loss curves
        
        # After all epochs for this (lr, wd) combination, evaluate on validation set
        # using the best model state for these parameters
        if current_best_state_for_params:
            model.load_state_dict(current_best_state_for_params)
        else: # Should not happen if epochs > 0, but as a fallback
            print(f"Warning: No best state found for LR={lr}, WD={wd}. Using last epoch model for metrics.")

        model.eval()
        current_metrics = {
            "iou_score": [], "f1_score": [], "f2_score": [], "accuracy": [], "recall": [], "precision": [],
            'tp' : [], 'fp' : [], 'fn' : [], 'tn' : [] # These will store (C,) tensors per sample
        }
        
        # val_dataset_transformed is TransformedDataset(val_subset_base, transforms_to_apply=None)
        # Its __getitem__ returns (img_tensor_chw_01, target_dict)
        with torch.no_grad():
            for img_tensor_chw_01, target_dict in tdqm(val_subset_base, desc=f"Metrics LR={lr} WD={wd}"):
                # img_tensor_chw_01 is (C,H,W), float, [0,1]
                img_np_hwc = img_tensor_chw_01.permute(1,2,0).cpu().numpy()
                img_preprocessed_np = preprocess_input(img_np_hwc) # HWC numpy
                img_for_model = torch.tensor(img_preprocessed_np).permute(2,0,1).unsqueeze(0).float().to(device) # 1,C',H',W'

                pred_logits = model(img_for_model) # Output: (1, 4, H, W)
                pred_probs = torch.softmax(pred_logits, dim=1).cpu() # (1, 4, H, W) on CPU

                # Process target_dict to ground truth mask (4, H, W) - similar to one sample in custom_collate_fn
                # This needs to be robust, using the image's actual H, W after transforms
                h, w = img_tensor_chw_01.shape[-2:]
                
                # Create gt_multiclass_mask from target_dict
                # (Reusing part of custom_collate_fn logic for a single sample)
                temp_batch = [(img_tensor_chw_01, target_dict)] # Create a dummy batch of 1
                _, single_mask_batch_list = custom_collate_fn(temp_batch) # Returns list with one [4,H,W] mask
                gt_multiclass_mask = single_mask_batch_list[0] # (4, H, W), on CPU by custom_collate_fn's design

                # Ensure gt_multiclass_mask is (1, 4, H, W) for get_stats with mode='multilabel'
                gt_multiclass_mask_batch = gt_multiclass_mask.unsqueeze(0) # (1, 4, H, W)
                
                # Calculate stats per class using 'multilabel' mode
                tp_c, fp_c, fn_c, tn_c = smp.metrics.get_stats(
                    pred_probs, # (1, C, H, W)
                    gt_multiclass_mask_batch, # (1, C, H, W)
                    mode='multilabel', 
                    threshold=0.5 
                ) # Output shape: (C,) for each stat tensor

                current_metrics['tp'].append(tp_c) # Appending (C,) tensor
                current_metrics['fp'].append(fp_c)
                current_metrics['fn'].append(fn_c)
                current_metrics['tn'].append(tn_c)
                
                current_metrics["iou_score"].append(smp.metrics.iou_score(tp_c, fp_c, fn_c, tn_c, reduction="none"))
                current_metrics["f1_score"].append(smp.metrics.f1_score(tp_c, fp_c, fn_c, tn_c, reduction="none"))
                current_metrics["f2_score"].append(smp.metrics.fbeta_score(tp_c, fp_c, fn_c, tn_c, beta=2, reduction="none"))
                current_metrics["accuracy"].append(smp.metrics.accuracy(tp_c, fp_c, fn_c, tn_c, reduction="none"))
                current_metrics["recall"].append(smp.metrics.recall(tp_c, fp_c, fn_c, tn_c, reduction="none"))
                current_metrics["precision"].append(smp.metrics.precision(tp_c, fp_c, fn_c, tn_c, reduction="none"))

        # Aggregate metrics for this (lr, wd)
        aggregated_metrics_for_params = {}
        for c in range(4):
        # print(f"son of a bitch: {torch.mean(current_metrics['tp'][:][0][0][c].float())}")
            aggregated_metrics_for_params[c] = {
                'tp': torch.mean(current_metrics['tp'][:][0][0][c].float()), 
                'fp': torch.mean(current_metrics['fp'][:][0][0][c].float()),
                'fn': torch.mean(current_metrics['fn'][:][0][0][c].float()),
                'tn': torch.mean(current_metrics['tn'][:][0][0][c].float()),
                'iou': torch.mean(current_metrics["iou_score"][:][0][0][c].float()),
                'f1': torch.mean(current_metrics["f1_score"][:][0][0][c].float()),
                'f2': torch.mean(current_metrics["f2_score"][:][0][0][c].float()),
                'precision': torch.mean(current_metrics["precision"][:][0][0][c].float()),
                'recall': torch.mean(current_metrics["recall"][:][0][0][c].float()),
                'accuracy': torch.mean(current_metrics["accuracy"][:][0][0][c].float())
            }
        # if len(current_metrics['tp']) > 0: # If validation set was not empty
        #     for c in range(4): aggregated_metrics_for_params[c] = {} # Initialize for 4 classes

        #     for metric_key in ["iou_score", "f1_score", "f2_score", "accuracy", "recall", "precision"]:
        #         # Stack list of (C,) tensors into (num_samples, C)
        #         stacked_metric_values = torch.stack(current_metrics[metric_key]) 
        #         mean_metric_per_class = torch.mean(stacked_metric_values, dim=0) # (C,)
        #         for c in range(4):
        #             print(f"metrics {metric_key}: {mean_metric_per_class[c]}")
        #             aggregated_metrics_for_params[c][metric_key] = float(mean_metric_per_class[c][c])
            
        #     for stat_key in ['tp', 'fp', 'fn', 'tn']:
        #         stacked_stat_values = torch.stack(current_metrics[stat_key]) # (num_samples, C)
        #         sum_stat_per_class = torch.sum(stacked_stat_values, dim=0) # (C,)
        #         for c in range(4):
        #             aggregated_metrics_for_params[c][stat_key] = sum_stat_per_class[c].item()
        # else: # Handle empty validation set case for metrics
        #      for c in range(4):
        #         aggregated_metrics_for_params[c] = {
        #             k: 0.0 for k in ['iou_score', 'f1_score', 'f2_score', 'accuracy', 'recall', 'precision', 
        #                              'tp', 'fp', 'fn', 'tn']}

        print(f"LR={lr}, WD={wd} - Aggregated Validation Metrics:")
        pprint.pprint(aggregated_metrics_for_params)

        param_sweep_results.append({
            'lr': lr,
            'wd': wd,
            'best_val_loss_for_params': current_best_val_loss_for_params, # Renamed for clarity
            'per_class_metrics': aggregated_metrics_for_params,
            'loss_curves': best_model_loss_curves # Loss curves from the best epoch for this LR/WD
        })

        if current_best_val_loss_for_params < best_overall_val_loss:
            best_overall_val_loss = current_best_val_loss_for_params
            best_model_state_dict = current_best_state_for_params # State dict from best epoch for this LR/WD
            best_params = {'lr': lr, 'wd': wd}


# After sweep, analyze results
print(f"\nParameter sweep finished.")
if best_params:
    print(f"Best overall validation loss: {best_overall_val_loss:.4f} achieved with LR={best_params['lr']} WD={best_params['wd']}")
else:
    print(f"Best overall validation loss: {best_overall_val_loss:.4f}. No specific best parameters identified (e.g. single run or all same loss).")


# Save the best overall model
if best_model_state_dict:
    torch.save(best_model_state_dict, "best_model_param_sweep.pth")
    print("Saved best model (from parameter sweep) to best_model_param_sweep.pth")
else:
    print("No best model state was recorded (e.g., validation loss was always inf or no training occurred).")

# Finalizing results for JSON
# Ensure all metric values are basic Python types for JSON serialization
results_to_dump_cleaned = []
for res in param_sweep_results:
    cleaned_metrics = {}
    for c_idx, metrics_val in res['per_class_metrics'].items():
        cleaned_metrics[str(c_idx)] = {k: float(v) if isinstance(v, (torch.Tensor, np.generic)) else v for k, v in metrics_val.items()}
    
    results_to_dump_cleaned.append({
        "lr": res["lr"],
        "wd": res["wd"],
        "best_val_loss": float(res["best_val_loss_for_params"]) if isinstance(res["best_val_loss_for_params"], (torch.Tensor, np.generic)) else res["best_val_loss_for_params"],
        "per_class_metrics": cleaned_metrics,
        "all_loss_curves": best_model_loss_curves
    })

# Calculate and print total execution time
end_time = time.time()
end_datetime = datetime.datetime.now()
execution_time = end_time - start_time
hours, remainder = divmod(execution_time, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Finished execution at: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} (H:M:S)")

# Dump training results to JSON
output_summary = {
    "script_start_time": start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
    "script_end_time": end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
    "total_execution_time_seconds": execution_time,
    "best_params_found": best_params,
    "best_overall_validation_loss": float(best_overall_val_loss) if isinstance(best_overall_val_loss, (torch.Tensor, np.generic)) else best_overall_val_loss,
    "parameter_sweep_runs": results_to_dump_cleaned
}

pp = pprint.PrettyPrinter(indent=4)
print("\nSummary of parameter sweep:")
pp.pprint(output_summary)

with open("param_sweep_results.json", "w") as f:
    json.dump(output_summary, f, indent=4)

print("Saved parameter sweep results to param_sweep_results.json")
# filepath: untitled:Untitled-1