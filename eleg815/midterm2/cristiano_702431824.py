"""
This template will help you solve the coding problem in the midterm.

This script contains a function to help you read the command-line 
arguments and some other to give you an initial structure with some
hints in the comments.

Import the necessary libraries and define any functions required for 
your implementation.

"""

import argparse
import torch
import torch.optim as optim
from torch import nn 
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader
# Create your model
class BiggerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiggerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout1 = nn.Dropout(p=0.25)  # Add dropout layer after first fully connected layer
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out
    

def read_args():
    """This function reads the arguments 
    from the command line. 

    Returns:
        list with the input arguments
    """
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("true"):
            return True
        elif v.lower() in ("false"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()

    parser.add_argument("is_testing", type=str2bool, help="Set Testing Stage")
    parser.add_argument("ckpt_path", nargs='?', type=str, help="Checkpoint path", default='./')
    parser.add_argument("testing_data_path", nargs='?', type=str, help="Testing data path", default='./')
    parser.add_argument("output_path", nargs='?', type=str, help="output path", default='./')

    return parser.parse_args()
input_size = 300  
hidden_size = 500  # Number of neurons in the hidden layer
output_size = 10 

def train():
    """
    Do everything related to the training of your model
    """
    # Create your data loader
    training_data = np.load('./midterm2_training_data.npy')
    training_labels = np.load('./midterm2_training_labels.npy')
    # Convert numpy arrays to torch tensors
    data_tensor = torch.tensor(training_data, dtype=torch.float64)
    labels_tensor = torch.tensor(training_labels, dtype=torch.uint8)
    # Create a TensorDataset
    dataset = TensorDataset(data_tensor, labels_tensor)
    # Calculate split lengths
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # Instantiate your model
    model = BiggerNN(input_size, hidden_size, output_size)
    # Setup your criterion
    criterion = nn.CrossEntropyLoss()
    # Setup your optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.00021, momentum=0.95)
    # Implement the training loop
    convergence_epoch = 0
    best_test_loss = float('inf')
    test_losses = []
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for points, labels in train_loader:
            points = points.float()
            labels = labels.long()
            # Forward pass
            outputs = model(points)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for points, labels in test_loader:
                points = points.float()
                labels = labels.long()
                outputs = model(points)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        # Save the model if test loss improves
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            # np.save('cristiano_702431824.npy', model.state_dict())
            torch.save(model.state_dict(), 'cristiano_702431824.pth')
        if epoch > 50:
            if convergence_epoch == 0 and best_test_loss < min(test_losses[-10:]):
                print(f'early stop at epoch {epoch+1} Loss: {best_test_loss:.4f}')
                convergence_epoch = epoch+1
                break
                
        if(epoch % 25 == 0 or epoch == num_epochs-1):
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
            if(epoch == num_epochs-1):
                print(f'Best Test Loss: {best_test_loss:.4f}')
                print(f'Convergence at epoch {convergence_epoch}')


def test(input_args):
    """
    Do everything related to the testing of your model
    """

    print('Start testing')
    # read data from input_args.testing_data_path
    testing_data = np.load(input_args.testing_data_path)
    data_tensor = torch.tensor(testing_data, dtype=torch.float64)
    test_loader = DataLoader(TensorDataset(data_tensor), batch_size=64, shuffle=False)
    # load model with weights from input_args.ckpt_path
    model = BiggerNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(input_args.ckpt_path, weights_only=True))
    model.eval()
    # compute model predictions
    predictions = []
    with torch.no_grad():
        for points in test_loader:
            points = points[0].float()
            outputs = model(points)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.numpy())
    # save [studentname_udid].npy file in input_args.output_path
    predictions = np.array(predictions)
    output_file = f"{input_args.output_path}/cristiano_702431824.npy"
    np.save(output_file, predictions)
   

def main():
    """This is the main function of yor script.
    From here you call your training or 
    testing functions.
    """
    input_args = read_args()

    if input_args.is_testing:
        test(input_args)
    else:
        train()

if __name__=="__main__":
    main()
