# My actual training script
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import copy
#TODO create dataloader dict with 70|20|10 split, normalized
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25 device='cuda'):
    pocket_dict = {
        'loss_train': [], 
        'loss_test': [], 
        'validation_loss': [],
        'model': copy.deepcopy(model.state_dict()),
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

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                #TODO: this needs to log loss
                running_loss += loss.item() * inputs.size(0)
                running_datacount += inputs.size(0)    
            epoch_loss = running_loss / running_datacount

        print(f'{phase} Loss: {epoch_loss:.4f}')
