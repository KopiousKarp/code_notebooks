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

def logits_to_mask(logits):
    mask = np.zeros((3,logits.shape[1],logits.shape[2]), dtype=np.uint8)
    print(logits.shape, mask.shape)
    for i in range(logits.shape[1]):
        for j in range(logits.shape[2]):
            if int(logits[0,i,j]) == 0:
                continue
            else:
                class_val = max(1, abs(int(logits[0,i,j])))
                mask[class_val-1, i, j] = 255
    
    return mask
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
            category_id = annotation['category_id']
            # print(category_id)
            if category_id <= self.num_classes:  # Ensure the category is within the range
                # Get segmentation data
                ann2mask = self.coco.annToMask(annotation)
                mask[category_id] += ann2mask*255 
         # Convert the mask to a tensor
        mask = torch.tensor(mask)#.permute(2, 0, 1)  # Change to (num_classes, height, width)
        #TODO: make sure this code outputs mask to be a tensor of the same size and shape as img
        # Apply transformations to the image if specified
        if self.transform:
            img = self.transform(img)
        # Center crop img and mask
        
        center_crop_size = 2440
        _, img_height, img_width = img.shape
        crop_top = (img_height - center_crop_size) // 2
        crop_left = (img_width - center_crop_size) // 2
        img = img[:, crop_top:crop_top + center_crop_size, crop_left:crop_left + center_crop_size]
        mask = mask[:, crop_top:crop_top + center_crop_size, crop_left:crop_left + center_crop_size]
        
        return img, mask

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])
dataset = CustomCocoDetection(
    root='./2024_annot/images',
    annFile='./2024_annot/2024_annotations.json',
    transform=transform,
    num_classes=4)


train_size = int(0.7 * len(dataset))
test_size = int(0.2 * len(dataset))
val_size = len(dataset) - train_size - test_size
dataset.train, dataset.test, dataset.val = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

#Batch size will stay 1 for now. Images are just too damn big
dataloader_train = DataLoader(dataset.train, batch_size=1, shuffle=True,collate_fn=lambda batch: tuple(zip(*batch)))
dataloader_test = DataLoader(dataset.test, batch_size=1, shuffle=False,collate_fn=lambda batch: tuple(zip(*batch)))
dataloader_val = DataLoader(dataset.val, batch_size=1, shuffle=False,collate_fn=lambda batch: tuple(zip(*batch)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')
for j in [UNet, SegNet, AttentionUNet, ResidualUNet]:
    model = j()
    model_name = f'{j.__name__}_exp_2024'
    print(f'Training {j.__name__}')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    # Define the training logic herusedse
    model.to(device) # uses memory
    
    testp = True
    for epoch in range(100):  # example: train for 10 epochs
        epoch_loss = 0
        datacount = 0
        for inputs, labels in tqdm(dataloader_val):

            datacount += len(inputs)
            #inputs is tuple with dimensions (batch, num_channels, height, width)
            # assert inputs[0].shape == labels[0].shape, f"Shape mismatch: inputs shape {inputs[0].shape}, labels shape {labels[0].shape}"
            inputs = torch.cat(inputs, dim=1).to(device)
            labels = torch.cat(labels,dim=1).to(device)
            # print(f'labels {labels.shape} inputs {inputs.shape}') #pass
            optimizer.zero_grad()
            outputs = model(inputs)# throws error here
            model.predict_mask()        
            labels = torch.argmax(labels, dim=0)

            # print(f'labels {labels.shape} inputs {inputs.shape}')
            # if testp:
            #     torch.set_printoptions(threshold=float('inf'))
            #     labels_t = labels.view(1, 1, *labels.shape).float()  # Reshape to (N, C, H, W)
            #     resized_labels = F.interpolate(labels_t, size=(16, 16), mode='nearest').squeeze(0).squeeze(0)  # Interpolate and then remove the added dimensions
            #     print(resized_labels)
                # testp = False           
            if labels.dim() == 2:  # If no batch dimension, add one
                labels = labels.unsqueeze(0)
            if outputs.dim() == 4:  # If no batch dimension, add one
                outputs = outputs.unsqueeze(0)
            
            assert labels.shape == outputs.shape, f"Shape mismatch: labels shape {labels.shape}, outputs shape {outputs.shape}"
        
            loss = criterion(outputs, labels.float())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        if epoch == 0 or epoch == 99:
            # outputs = F.softmax(outputs, dim=1)*255
            print("saving output")
         
            # labels = logits_to_mask(labels.cpu().detach().numpy())
            # labels = torch.from_numpy(labels)
        
            # transforms.ToPILImage()(labels).save(f'label_epoch{epoch}.png')
                
            # outputs = logits_to_mask(outputs)
            # transforms.ToPILImage()(outputs).save(f'output_epoch{epoch}.png')
        print(f'epoch {epoch} loss = {epoch_loss/datacount}')
        print(f'Model memory usage: {torch.cuda.memory_allocated(device) / (1024 ** 3):.2f} GB')
    # Save the trained model here
    torch.save(model.state_dict(), f'{model_name}.pth')





