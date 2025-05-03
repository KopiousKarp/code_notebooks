# My actual training script
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import copy


def train_model(model_name,model, dataloaders, criterion, optimizer, num_epochs=25, device='cuda'):
    pocket_dict = {
        'name': model_name, 
        'loss_train': [], 
        'loss_test': [], 
        'loss_val': float('inf'),
        'model': None,
        "best_loss": float('inf') 
        }
    model.to(device)
    for epoch in tqdm(range(num_epochs)):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        for phase in ['train','test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_datacount = 0
            for dataloader in dataloaders:
                for inputs, masks in dataloader[phase]:
                    inputs = inputs.to(device)
                    masks = masks.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs,masks)
                        running_loss += loss.item() * inputs.size(0)
                        running_datacount += inputs.size(0) 
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    # del inputs, masks
                    # torch.cuda.empty_cache()
            epoch_loss += (running_loss / running_datacount)
            if phase == 'train':
                pocket_dict['loss_train'].append(epoch_loss)
            else:
                pocket_dict['loss_test'].append(epoch_loss)
                if epoch_loss < pocket_dict['best_loss']:
                    pocket_dict['best_loss'] = epoch_loss
                    pocket_dict['model'] = copy.deepcopy(model.state_dict())
            print(f'{phase} Loss: {epoch_loss:.4f}')

            if epoch > 10 and pocket_dict['best_loss'] < pocket_dict['loss_test'][-10].any():
                print('Early stopping')
                model.load_state_dict(pocket_dict['model'])
                model.eval()
                with torch.set_grad_enabled(False):
                    for dataloader in dataloaders:
                        for inputs, masks in dataloader['val']:
                            inputs = inputs.to(device)
                            masks = masks.to(device)
                            outputs = model(inputs)
                            loss = criterion(outputs,masks)
                            pocket_dict['loss_val'] = loss.item()
                break

    return pocket_dict

from myCnns import UNet, U2Net, ResidualUNet, AttentionUNet, SegNet
from pycocotools.coco import COCO
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

coco_24 = COCO('/work/2024_annot/2024_annotations.json')
coco_23 = COCO('/work/2023_annot/2023_annotations.json')
coco_22 = COCO('/work/2022_annot/2022_annotations.json')
coco_21 = COCO('/work/2021_annot/2021_annotations.json')
    
class CocoDataset(Dataset):
    def __init__(self, coco, img_ids, transform=None):
        self.coco = coco
        self.img_ids = img_ids
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
    
        img_path = img_info['file_name']
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"File not found: {img_path}")
            # Return a placeholder or skip this item
            return None

        masks = np.zeros((3, img_info['height'], img_info['width']))
        for ann in anns:
            class_id = ann['category_id'] - 1
            masks[class_id] = np.maximum(masks[class_id], self.coco.annToMask(ann))
            categories = self.coco.loadCats(self.coco.getCatIds())
            if [category['name'] for category in categories] == ["Markers", "Roots", "Stalks"]:
                continue
            else: 
                masks[[0, 1]] = masks[[1, 0]]
        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
    #Possibly usefull for augmentations later
    # transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor()
    # ])
transform = transforms.ToTensor()
dataloaders = []
img_paths = [
    '/work/2021_annot/images',
    '/work/2022_annot/images',
    '/work/2023_annot/images',
    '/work/2024_annot/images'
]
ann_paths = [
    '/work/2021_annot/2021_annotations.json',
    '/work/2022_annot/2022_annotations.json',
    '/work/2023_annot/2023_annotations.json',
    '/work/2024_annot/2024_annotations.json'
]
# for coco in [coco_21, coco_22, coco_23, coco_24]:
for i in range(4):
    # img_ids = coco.getImgIds()
    # dataset = CocoDataset(coco, img_ids, transform=transform)
    dataset = CocoDetection(img_paths[i], ann_paths[i], transform=transform)
    train_size = int(0.7 * len(dataset))
    test_size = int(0.2 * len(dataset))
    val_size = len(dataset) - train_size - test_size
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])
    #Need to change the batch size for 2023 because input dimensions are different
    if i == 2:
        dataloader = {
            'train': DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1),
            'test': DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1),
            'val': DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
        }
    else:
        dataloader = {
            'train': DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1),
            'test': DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=1),
            'val': DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=1)
        }
    dataloaders.append(dataloader)
# Calculate class weights for weighted cross entropy loss
weight_tensors = []
for dataloader in dataloaders:
    class_counts = np.zeros(3)  # Assuming 3 classes
    for phase in ['train', 'test', 'val']:
        for item in dataloader[phase]:  #line throws error
            if item is None:
                continue
            image, annotations = item  # image is the input; annotations contain segmentation details
            height, width = image.shape[-2:]  # Get image dimensions
            multi_class_mask = np.zeros((3, height, width), dtype=np.uint8)  # Assuming 3 classes

            for annotation in annotations:
                class_id = annotation['category_id'] - 1  # Convert to zero-based index
                # Convert segmentation to binary mask
                multi_class_mask[class_id] = np.maximum(multi_class_mask[class_id], binary_mask)

            # Count pixels for each class
            for class_id in range(3):
                class_counts[class_id] += (multi_class_mask[class_id] == 1).sum()

    
    total_counts = class_counts.sum()
    class_weights = total_counts / (3 * class_counts)
    weight_tensors.append(torch.tensor(class_weights, dtype=torch.float))


for i, year in enumerate(["21","22","23","24"]): 
    for j in [UNet, U2Net, ResidualUNet, AttentionUNet, SegNet]:
        model = j()
        model_name = f'{j.__name__}_exp_{year}'
        criterion = nn.CrossEntropyLoss(weight=weight_tensors[i])
        # criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        pocket_dict = train_model(model_name, model, [dataloaders[i]], criterion, optimizer, num_epochs=1)
        torch.save(pocket_dict, f'{model_name}.pt')

for j in [UNet, U2Net, ResidualUNet, AttentionUNet, SegNet]:
    model = j()
    model_name = f'{j.__name__}_exp_multi'
    criterion = nn.CrossEntropyLoss(weight=weight_tensors.mean(dim=0))
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    pocket_dict = train_model(model_name, model, dataloaders, criterion, optimizer, num_epochs=10)
    torch.save(pocket_dict, f'{model_name}.pt')


