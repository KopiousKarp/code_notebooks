import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
from PIL import Image, ImageDraw
from tqdm import tqdm
import random
import numpy as np
from PadSquare import PadSquare 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#ChatGPT code for extracting distributions and computing kl divergence
def compute_kl_divergence(p, q, epsilon=1e-10):
    """
    Compute the KL divergence between two probability distributions.
    """
    p = p + epsilon
    q = q + epsilon
    return F.kl_div(p.log(), q, reduction='batchmean')

def get_image_distribution(data, num_bins=32):
    """
    Compute a global histogram for the entire dataset (across all images).
    Input: Tensor of shape [N, C, H, W]
    Returns: 1D normalized histogram
    """
    if isinstance(data, list):
        data = [v2.ToTensor()(data[i][0]) for i in range(len(data))]
    flat_pixels = data.flatten()
    hist = torch.histc(flat_pixels, bins=num_bins, min=flat_pixels.min(), max=flat_pixels.max())
    hist = hist / hist.sum()
    return hist

def get_label_distribution(labels, num_classes):
    """
    Compute normalized histogram of label classes.
    """
    hist = torch.bincount(labels, minlength=num_classes).float()
    hist = hist / hist.sum()
    return hist

coco_roots = ['/work/2021_annot/images',
              '/work/2022_annot/images',
              '/work/2023_annot/images',
              '/work/2024_annot/images']

coco_annFiles = ['/work/2021_annot/2021_annotations.json',
                 '/work/2022_annot/2022_annotations.json',
                 '/work/2023_annot/2023_annotations_corrected.json',
                 '/work/2024_annot/2024_annotations.json']
standard_transform = transforms.v2.Compose([
    PadSquare(padding_mode='symmetric'),
    v2.Resize((512, 512)), 
    v2.ToImage()
    ])
full_dataset = datasets.CocoDetection(root=coco_roots[0], annFile=coco_annFiles[0], transforms=standard_transform)
full_dataset = datasets.wrap_dataset_for_transforms_v2(full_dataset, target_keys=("labels", "masks"))
for coco_root, coco_annFile in zip(coco_roots[1:], coco_annFiles[1:]):
    seasonal_dataset = datasets.CocoDetection(root=coco_root, annFile=coco_annFile, transforms=standard_transform)
    seasonal_dataset = datasets.wrap_dataset_for_transforms_v2(seasonal_dataset, target_keys=("labels", "masks"))
    full_dataset = torch.utils.data.ConcatDataset([full_dataset, seasonal_dataset])

train_size = int(0.8 * len(full_dataset))
test_size = int(0.1 * len(full_dataset))

val_size = len(full_dataset) - (train_size + test_size)
full_dataset.train, full_dataset.test, full_dataset.val = torch.utils.data.random_split(full_dataset, [train_size, test_size, val_size])
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

full_dataset.train.transforms = transforms.v2.Compose([standard_transform, augmentations])
full_dataset.test.transforms = standard_transform
full_dataset.val.transforms = standard_transform
train_dataloader = DataLoader(
    full_dataset.train, 
    batch_size=8, 
    shuffle=True, 
    num_workers=2,
    collate_fn=custom_collate_fn)
test_dataloader = DataLoader(
    full_dataset.test, 
    batch_size=8, 
    shuffle=True, 
    num_workers=2,
    collate_fn=custom_collate_fn)
#NEXT STEPS 

# get the distribution of the histograms for this data 
# get the distribution for the labels of this data 
#KL divergence of the distributions 
kl_image = compute_kl_divergence(
            get_image_distribution([img for img, _ in train_dataloader], num_bins = 32),
            get_image_distribution([test_dataloader[:][0]], num_bins=32)
        )

kl_label = compute_kl_divergence(
            get_label_distribution([train_dataloader[:][1]], num_classes = 2),
            get_label_distribution([test_dataloader[:][1]], num_classes = 2)
        )



# apply a threshold to the kl divergences 
kl_threshold_image = 0.05
kl_threshold_label = 0.05
if kl_image < kl_threshold_image and kl_label < kl_threshold_label:
            print(f"Split found with divergences: KL(image)={kl_image:.4f}, KL(label)={kl_label:.4f}")
#Pass or fail and repeat as needed










