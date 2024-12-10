import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import copy
from pycocotools.coco import COCO
from torchvision import transforms, models, tv_tensors, datasets
from torchvision.transforms import v2
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import numpy as np
import torch.nn.functional as F
from myCnns import *


torch.manual_seed(0)
class CustomCocoDetection(CocoDetection):
    def __init__(self, root, annFile, transform=None, num_classes=4):
        super(CustomCocoDetection, self).__init__(root, annFile, transform=transform)
        self.num_classes = num_classes
        # add a coco structure to create the mask properly
        self.coco = COCO(annFile)

    def __getitem__(self, index):
        # Load image and annotations
        img, target = super(CustomCocoDetection, self).__getitem__(index)
       
        _, width, height = img.shape
         
        mask = np.zeros((self.num_classes, width, height), dtype=np.uint8)
        
        # Parse annotations to draw masks
        for annotation in target:
            category_id = annotation['category_id'] # 1-3
            # print(category_id)
            if category_id <= self.num_classes:  # Ensure the category is within the range
                # Get segmentation data
                ann2mask = self.coco.annToMask(annotation)
                mask[category_id] += ann2mask 
        # Create background mask (mask[0]) as inverse of all other masks
        mask[0] = 1 - np.maximum.reduce([mask[i] for i in range(1, self.num_classes)])
         # Convert the mask to a tensor
        mask = torch.tensor(mask)
       
        # Apply transformations to the image if specified
        if self.transform:
            img = self.transform(img)
            # mask = self.transform(mask)
        # Center crop img and mask
        center_crop_size = 1024
        _, img_height, img_width = img.shape
        crop_top = (img_height - center_crop_size) // 2
        crop_left = (img_width - center_crop_size) // 2
        img = img[:, crop_top:crop_top + center_crop_size, crop_left:crop_left + center_crop_size]
        mask = mask[:, crop_top:crop_top + center_crop_size, crop_left:crop_left + center_crop_size]
        
        return img, mask

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])
dataset = CustomCocoDetection(
    root='./2024_annot/images',
    annFile='./2024_annot/2024_annotations.json',
    transform=transform,
    num_classes=4)
    # Calculate mean and std across the dataset
mean = torch.zeros(3)
std = torch.zeros(3)
nb_samples = 0.
print("Calculating Normalization factors")
for img, _ in tqdm(dataset):
    mean += img.mean(dim=[1,2])
    std += img.std(dim=[1,2])
    nb_samples += 1

mean /= nb_samples
std /= nb_samples

# Update transform to include normalization
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=mean, std=std),
    # v2.RandomHorizontalFlip(p=1),
    # v2.RandomPerspective(distortion_scale=0.5, p=0.5),
    # v2.RandomVerticalFlip(p=0.5),
    # v2.GaussianBlur(7,sigma=(0.1, 2.0)) 
])

dataset.transform = transform

train_size = int(0.8 * len(dataset))
test_size = int(0.2 * len(dataset))
test_size += len(dataset) - (train_size + test_size)
dataset.train, dataset.test = torch.utils.data.random_split(dataset, [train_size, test_size])

#Batch size will stay 1 for now. Images are just too damn big
dataloader_train = DataLoader(dataset.train, batch_size=1, shuffle=True,collate_fn=lambda batch: tuple(zip(*batch)))
dataloader_test = DataLoader(dataset.test, batch_size=1, shuffle=False,collate_fn=lambda batch: tuple(zip(*batch)))

# Calculate class weights for weighted cross entropy
class_counts = torch.zeros(4)  # 4 classes including background
total_pixels = 0
print("Calculating class imbalances")
for _, masks in tqdm(dataloader_train):
    masks = torch.cat(masks, dim=0)
    # Count pixels for classes 1-3
    for i in range(1, 4):
        class_counts[i] += (masks[i] > 0).sum().item()
    total_pixels += masks[0].numel()
    # Class 0 (background) is all remaining pixels
    class_counts[0] += total_pixels - sum(class_counts[1:])

class_weights = 1.0 / (class_counts + 1e-8)  # add small epsilon to avoid division by zero
class_weights = class_weights / class_weights.sum() * len(class_weights)  # normalize weights


print("Class weights:", class_weights)





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)
print(f'Using {device}')
for j in [U2Net, ResidualUNet, SegNet, UNet]:
    model = j()
    model_name = f'{j.__name__}_exp_2024'
    print(f'Training {j.__name__}')
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    # Define the training logic 
    model.to(device) # uses memory
    
    
    for epoch in range(100):  # example: train for 10 epochs
               
        for dataloader in [dataloader_train, dataloader_test]:
            epoch_loss = 0
            datacount = 0
            if dataloader == dataloader_train:
                model.train()
                train_flag = 1
            else:
                model.eval()
                train_flag= 0
            
            for inputs, labels in tqdm(dataloader):
                
                datacount += len(inputs)
                inputs = torch.cat(inputs, dim=0).to(device)
                labels = torch.cat(labels,dim=0).to(device)

                # print(f'labels {labels.shape} inputs {inputs.shape}') #pass
                if train_flag:
                    torch.set_grad_enabled(True)
                    optimizer.zero_grad()
                else:
                    torch.set_grad_enabled(False)
                outputs = model(inputs)
                     
                
                if labels.dim() == 3:  # If no batch dimension, add one
                    labels = labels.unsqueeze(0)
                if outputs.dim() == 3:  
                    outputs = outputs.unsqueeze(0)
            
                
                assert labels.shape == outputs.shape, f"Shape mismatch: labels shape {labels.shape}, outputs shape {outputs.shape}"
                # print(f"Maximum output value: {outputs.max()}")
                loss = criterion(outputs, labels.float())
                epoch_loss += loss.item()
                if train_flag:
                    loss.backward()
                    optimizer.step()
            
            if train_flag:
                print(f'training epoch {epoch} loss = {epoch_loss/datacount}')
            else:
                print(f'Testing epoch {epoch} loss = {epoch_loss/datacount}')
            # print(f'Model memory usage: {torch.cuda.memory_allocated(device) / (1024 ** 3):.2f} GB')
    # Save the trained model here
    torch.save(model.state_dict(), f'{model_name}.pth')





