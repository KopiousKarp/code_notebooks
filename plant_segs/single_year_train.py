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
from myCnns import *
torch.manual_seed(0)
class CustomCocoDetection(CocoDetection):
    def __init__(self, root, annFile, transform=None, num_classes=3):
        super(CustomCocoDetection, self).__init__(root, annFile, transform=transform)
        self.num_classes = num_classes

    def __getitem__(self, index):
        # Load image and annotations
        img, target = super(CustomCocoDetection, self).__getitem__(index)
        # Create an empty mask with shape (height, width, num_classes)
        # print(img.shape, target)
        _, width, height = img.shape
        mask = np.zeros((height, width, self.num_classes), dtype=np.uint8)
        
        # Parse annotations to draw masks
        for annotation in target:
            category_id = annotation['category_id']
            if category_id <= self.num_classes:  # Ensure the category is within the range
                # Get segmentation data
                segmentation = annotation.get('segmentation', [])
                if isinstance(segmentation, list):  # Polygon segmentation
                    for polygon in segmentation:
                        poly = np.array(polygon).reshape(-1, 2)
                        img_poly = Image.new('L', (width, height), 0)
                        ImageDraw.Draw(img_poly).polygon(poly, outline=1, fill=1)
                        mask[:, :, category_id - 1] |= np.array(img_poly, dtype=np.uint8)
         # Convert the mask to a tensor
        mask = torch.tensor(mask).permute(2, 0, 1)  # Change to (num_classes, height, width)
        #TODO: make sure this code outputs mask to be a tensor of the same size and shape as img
        # Apply transformations to the image if specified
        if self.transform:
            img = self.transform(img)
        
        return img, mask

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])
dataset = CustomCocoDetection(
    root='./2024_annot/images',
    annFile='./2024_annot/2024_annotations.json',
    transform=transform,
    num_classes=3)


train_size = int(0.7 * len(dataset))
test_size = int(0.2 * len(dataset))
val_size = len(dataset) - train_size - test_size
dataset.train, dataset.test, dataset.val = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

dataloader_train = DataLoader(dataset.train, batch_size=1, shuffle=True,collate_fn=lambda batch: tuple(zip(*batch)))
dataloader_test = DataLoader(dataset.test, batch_size=1, shuffle=False,collate_fn=lambda batch: tuple(zip(*batch)))
dataloader_val = DataLoader(dataset.val, batch_size=1, shuffle=False,collate_fn=lambda batch: tuple(zip(*batch)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')
for j in [UNet, U2Net, ResidualUNet, AttentionUNet, SegNet]:
    model = j()
    model_name = f'{j.__name__}_exp_2024'
    print(f'Training {j.__name__}')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # Define the training logic here
    model.to(device)
    for epoch in range(10):  # example: train for 10 epochs
        epoch_loss = 0
        datacount = 0
        for batch in tqdm(dataloader_train):
            inputs, labels = batch
            datacount += len(inputs)
            #inputs is tuple with dimensions (batch, num_channels, height, width)
            inputs = torch.cat(inputs, dim=1).to(device)
            labels = [label.to(device) for label in labels]
            optimizer.zero_grad()
            outputs = model(inputs) # throws error here
            # print(outputs.shape, labels[0].permute(0, 2, 1).shape)
            labels = [label.permute(0, 2, 1).float() for label in labels]
            outputs = outputs.float()
            loss = criterion(outputs, *labels)
            epoch_loss += loss
            loss.backward()
            optimizer.step()
        print(f'epoch loss = {epoch_loss/datacount}')
    # Save the trained model here
    torch.save(model.state_dict(), f'{model_name}.pth')





