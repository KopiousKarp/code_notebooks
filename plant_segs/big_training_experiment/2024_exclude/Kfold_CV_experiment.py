import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import v2
from sklearn.model_selection import KFold
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch.nn as nn
from tqdm import tqdm as tdqm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
from PadSquare import PadSquare
import time
import datetime
import json
import os
import pprint
import argparse # Add this import
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

def get_dataloaders(dataset, train_idx, val_idx, batch_size=16, num_workers=2):
    train_idx = [int(i) for i in train_idx]
    val_idx   = [int(i) for i in val_idx]

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    # Assign transforms
    train_subset.dataset.transforms = transforms.v2.Compose([
        PadSquare(padding_mode='symmetric'),
        v2.Resize((512, 512)),
        v2.ToImage(),
        augmentations
    ])
    val_subset.dataset.transforms = transforms.v2.Compose([
        PadSquare(padding_mode='symmetric'),
        v2.Resize((512, 512)),
        v2.ToImage()
    ])

    train_dl = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=custom_collate_fn
    )
    val_dl = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=custom_collate_fn
    )
    return train_dl, val_dl

# Prepare full_dataset as before (concatenate COCO datasets)
# ... [your dataset loading code here] ...
# Add argument parsing
parser = argparse.ArgumentParser(description="Train a U-Net model with K-fold cross-validation.")
parser.add_argument('--coco_roots', nargs='+', required=True, help='List of COCO dataset root directories.')
parser.add_argument('--coco_annFiles', nargs='+', required=True, help='List of COCO annotation files.')
args = parser.parse_args()

coco_roots = args.coco_roots
coco_annFiles = args.coco_annFiles

standard_transform = transforms.v2.Compose([
    PadSquare(padding_mode='symmetric'),
    v2.Resize((512, 512)), 
    v2.ToImage()
    ])
full_dataset = datasets.CocoDetection(root=coco_roots[0], annFile=coco_annFiles[0], transforms=standard_transform)
full_dataset = datasets.wrap_dataset_for_transforms_v2(full_dataset, target_keys=("labels", "masks"))
if len(coco_roots) > 1:
    for coco_root, coco_annFile in zip(coco_roots[1:], coco_annFiles[1:]):
        seasonal_dataset = datasets.CocoDetection(root=coco_root, annFile=coco_annFile, transforms=standard_transform)
        seasonal_dataset = datasets.wrap_dataset_for_transforms_v2(seasonal_dataset, target_keys=("labels", "masks"))
        full_dataset = torch.utils.data.ConcatDataset([full_dataset, seasonal_dataset])

# Define KFold
n_splits = 30
kf = KFold(n_splits=n_splits, shuffle=True, random_state=69)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
    print(f"\n===== Fold {fold + 1}/{n_splits} =====")

    # Get train and validation dataloaders
    train_dl, val_dl = get_dataloaders(full_dataset, train_idx, val_idx)

    # Calculate class weights for weighted cross entropy
    print("Calculating class imbalances for fold", fold+1)
    class_counts = torch.zeros(4, device=device)
    for imgs, masks in tdqm(train_dl):
        for mask in masks:
            for i in range(4):
                class_counts[i] += mask[i].sum()
    class_weights = class_counts / class_counts.sum()
    print(f"Class weights: {class_weights.cpu().numpy()}")
    class_weights = 1-class_weights
    # Initialize model, loss, optimizer
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=4,
        decoder_channels=(256,128,64,32,16)
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.float().to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Training loop per fold
    best_val_loss = float('inf')
    best_state = None
    patience_threshold = 5
    patience_counter = 0
    for epoch in range(25):
        # Training
        model.train()
        train_loss = 0.0
        for imgs, masks in tdqm(train_dl):
            # Preprocess and move to device
            imgs = torch.stack([
                torch.tensor(preprocess_input(img.permute(1,2,0).numpy()))
                for img in imgs
            ]).permute(0,3,1,2).float().to(device)
            masks = torch.stack(masks).long().to(device)

            preds = model(imgs)
            loss = criterion(preds, masks.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)
        # print(f"Fold {fold+1} Epoch {epoch+1}: train_loss={train_loss:.4f}")
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in tdqm(val_dl):
                imgs = torch.stack([
                    torch.tensor(preprocess_input(img.permute(1,2,0).numpy()))
                    for img in imgs
                ]).permute(0,3,1,2).float().to(device)
                masks = torch.stack(masks).long().to(device)

                preds = model(imgs)
                val_loss += criterion(preds, masks.float()).item()
            val_loss /= len(val_dl)

        # Save best model state
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
            if patience_counter >= patience_threshold:
                print(f"Early stopping at epoch {epoch+1} for fold {fold+1}")
                break

        print(f"Fold {fold+1} Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    metrics_dict = {
        "IoU Score": [],
        "F1 Score": [],
        "F2 Score": [],
        "Accuracy": [],
        "Recall": [],
        "Precision": [],
        'tp' : [],
        'fp' : [],
        'fn' : [],
        'tn' : []
        }
    for image, target in val_dl.dataset:
        image = torch.tensor(preprocess_input(image.permute(1, 2, 0).numpy()))
        image = image.permute(2, 0, 1).unsqueeze(0).float()
        result = model(image.to(device))
        result = torch.softmax(result, dim=1).cpu()
        multiclass_mask = torch.zeros((4, target["masks"].shape[1], target["masks"].shape[2]), dtype=target["masks"].dtype)
        for idx, label in enumerate(target["labels"]):
            multiclass_mask[label] += target["masks"][idx]*255
        multiclass_mask[0] = 255 - torch.max(multiclass_mask[1:], dim=0)[0]
    
        # import matplotlib.pyplot as plt

        # # Convert prediction to class labels
        # pred_mask = torch.argmax(result, dim=1).squeeze().cpu().numpy()

        # # Convert multiclass_mask from one-hot to class indices
        # true_mask = torch.argmax(multiclass_mask, dim=0).cpu().numpy()

        # # Create directory for visualizations if it doesn't exist
        # os.makedirs(f"fold_{fold+1}_visualizations", exist_ok=True)

        # # Get the original image and convert for display
        # img_display = image.squeeze().permute(1, 2, 0).cpu().numpy()
        # # Normalize for display if needed
        # img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

        # # Get current sample index
        # sample_idx = len(metrics_dict["IoU Score"])
        # if sample_idx < 10:  # Only save first 10 samples
        #     # Create a figure with 3 subplots
        #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
        #     # Plot original image
        #     axes[0].imshow(img_display)
        #     axes[0].set_title('Original Image')
        #     axes[0].axis('off')
            
        #     # Plot ground truth mask
        #     axes[1].imshow(true_mask, cmap='viridis', vmin=0, vmax=3)
        #     axes[1].set_title('Ground Truth Mask')
        #     axes[1].axis('off')
            
        #     # Plot predicted mask
        #     axes[2].imshow(pred_mask, cmap='viridis', vmin=0, vmax=3)
        #     axes[2].set_title('Predicted Mask')
        #     axes[2].axis('off')
            
        #     plt.tight_layout()
        #     plt.savefig(f"fold_{fold+1}_visualizations/sample_{sample_idx}.png")
        #     plt.close()

        # multiclass_mask_tensor = torch.from_numpy(multiclass_mask)
        tp, fp, fn, tn = smp.metrics.get_stats(result, multiclass_mask.unsqueeze(0), mode='binary', threshold=0.5)
        # print(tp.shape, fp.shape, fn.shape, tn.shape)
        # print(tp, fp, fn, tn)
        metrics_dict['tp'].append(tp)
        metrics_dict['fp'].append(fp)
        metrics_dict['fn'].append(fn)
        metrics_dict['tn'].append(tn)
        metrics_dict["IoU Score"].append(smp.metrics.iou_score(tp, fp, fn, tn, reduction="none"))
        metrics_dict["F1 Score"].append(smp.metrics.f1_score(tp, fp, fn, tn, reduction="none"))
        metrics_dict["F2 Score"].append(smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="none"))
        metrics_dict["Accuracy"].append(smp.metrics.accuracy(tp, fp, fn, tn, reduction="none"))
        metrics_dict["Recall"].append(smp.metrics.recall(tp, fp, fn, tn, reduction="none"))
        metrics_dict["Precision"].append(smp.metrics.precision(tp, fp, fn, tn, reduction="none"))
    
    per_class_metrics = {}
    
    
    for c in range(4):
        # print(f"son of a bitch: {torch.mean(metrics_dict['tp'][:][0][0][c].float())}")
        per_class_metrics[c] = {
            'tp': torch.mean(metrics_dict['tp'][:][0][0][c].float()),
            'fp': torch.mean(metrics_dict['fp'][:][0][0][c].float()),
            'fn': torch.mean(metrics_dict['fn'][:][0][0][c].float()),
            'tn': torch.mean(metrics_dict['tn'][:][0][0][c].float()),
            'iou': torch.mean(metrics_dict["IoU Score"][:][0][0][c].float()),
            'f1': torch.mean(metrics_dict["F1 Score"][:][0][0][c].float()),
            'f2': torch.mean(metrics_dict["F2 Score"][:][0][0][c].float()),
            'precision': torch.mean(metrics_dict["Precision"][:][0][0][c].float()),
            'recall': torch.mean(metrics_dict["Recall"][:][0][0][c].float()),
            'accuracy': torch.mean(metrics_dict["Accuracy"][:][0][0][c].float())
        }
    # Print per-class metrics for the fold
    print(f"Fold {fold+1} Per-Class Metrics: {per_class_metrics}")

   

    # print(f"Fold {fold+1} Epoch {epoch+1}: per_class_metrics={per_class_metrics}")

     # record results
    fold_results.append({
        'fold': fold+1,
        'best_val_loss': best_val_loss,
        'per_class_metrics': per_class_metrics,
        'state_dict': best_state
    })


# Compute average best validation loss across folds
avg_loss = np.mean([res['best_val_loss'] for res in fold_results])
print(f"\nAverage best validation loss across folds: {avg_loss:.4f}")
# Calculate average per_class_metrics across folds
avg_metrics = {c: {'iou': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'f2': 0, 'accuracy': 0} for c in range(4)}

for res in fold_results:
    for c in range(4):
        for metric in avg_metrics[c].keys():
            avg_metrics[c][metric] += res['per_class_metrics'][c][metric]

# Divide by the number of folds to get averages
for c in range(4):
    for metric in avg_metrics[c].keys():
        avg_metrics[c][metric] /= n_splits

print("\nAverage per-class metrics across folds:")
for c in range(4):
    print(f"Class {c}:")
    for metric, value in avg_metrics[c].items():
        print(f"  {metric}: {value:.4f}")
# Calculate and print total execution time
end_time = time.time()
end_datetime = datetime.datetime.now()
execution_time = end_time - start_time
hours, remainder = divmod(execution_time, 3600)
minutes, seconds = divmod(remainder, 60)



print(f"Finished execution at: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} (H:M:S)")
print(f"Total seconds: {execution_time:.2f}")
# Save the best overall model (lowest val loss) to disk
best_overall = min(fold_results, key=lambda x: x['best_val_loss'])
torch.save(best_overall['state_dict'], "best_model.pth")
print("Saved best model to best_model.pth")

# Prepare results for JSON (exclude state_dict)
results_to_dump = []
for res in fold_results:
    results_to_dump.append({
        "fold": res["fold"],
        "best_val_loss": res["best_val_loss"],
        "per_class_metrics": {c: {k: float(v) for k, v in metrics.items()} for c, metrics in res["per_class_metrics"].items()}
    })

# Dump training results to JSON
# Pretty print avg_metrics
pp = pprint.PrettyPrinter(indent=4)
print("\nAverage per-class metrics across folds (pretty print):")
# Convert tensor values to float for prettier printing
pretty_avg_metrics = {}
for c in range(4):
    pretty_avg_metrics[f"Class {c}"] = {metric: float(value) for metric, value in avg_metrics[c].items()}
pp.pprint(pretty_avg_metrics)
with open("training_results.json", "w") as f:
    json.dump({
        "start_time": start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "avg_best_val_loss": avg_loss,
        "folds": results_to_dump,
        "avg_per_class_metrics": pretty_avg_metrics
    }, f, indent=4)

print("Saved training results to training_results.json")
