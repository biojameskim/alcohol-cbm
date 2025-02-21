import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.backends.cudnn as cudnn
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             accuracy_score, balanced_accuracy_score, roc_auc_score)

import os 

# Determine device for PyTorch (CUDA GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class E2EBlock(nn.Module):
    def __init__(self, in_planes, planes, example, bias=False):
        super().__init__() 
        # Use the 4th dimension of the example input to determine the kernel width/height.
        self.d = example.size(3)
        self.cnn1 = nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        # Explicit dimension specification for concatenation
        return torch.cat([a]*self.d, dim=3) + torch.cat([b]*self.d, dim=2)

class BrainNetCNN(nn.Module):
    def __init__(self, example):
        super().__init__()
        # Use the example input to determine the number of input channels and spatial dimensions.
        self.in_planes = example.size(1)
        self.d = example.size(3)
        
        self.E2Econv1 = E2EBlock(1, 32, example, bias=True)
        self.E2Econv2 = E2EBlock(32, 64, example, bias=True)
        self.E2N = nn.Conv2d(64, 1, (1, self.d))
        self.N2G = nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = nn.Linear(256, 128)
        self.dense2 = nn.Linear(128, 30)
        self.dense3 = nn.Linear(30, 1) # Output a single logit
        
    def forward(self, x):
        out = F.leaky_relu(self.E2Econv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.E2Econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(self.dense2(out), negative_slope=0.33), p=0.5)
        out = self.dense3(out) 
        return out

dir = "data/training_data/cnn/aligned"
# matrix_type = "SC"
# matrix_type = "FC"
matrix_type = "FCgsr"


# Dataset and DataLoader updates
class NCANDA_Dataset(Dataset):
    def __init__(self, directory=dir, matrix_type=matrix_type, mode="train", transform=None, class_balancing=False):
        """
        Args:
            directory (string): Path to the dataset.
            mode (str): "train" for training, "validation" for validation, "train+validation" for full training.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.mode = mode
        self.transform = transform
        
        x = np.load(os.path.join(directory, "X_" + matrix_type + "_control_moderate.npy"))
        y = np.load(os.path.join(directory, "y_aligned_control_moderate.npy"))
        print(f"Loaded {matrix_type} data with shape: {x.shape}, {y.shape}")
        print("Amount of 1s in y: ", np.sum(y == 1))
        
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.33, random_state=42
        ) # test is the validation set (for conditionals below)
        
        if mode == "train":
            x, y = X_train, y_train
            print("Amount of 1s in y (train): ", np.sum(y == 1))
        elif mode == "validation":
            x, y = X_test, y_test
            print("Amount of 1s in y (validation): ", np.sum(y == 1))
        elif mode == "train+validation":
            pass  # Use full dataset
        else:
            raise ValueError("Invalid mode specified")
        
        # NORMALIZE DATA
        if mode == "train":
            self.mean, self.std = x.mean(), x.std()
        x = (x - X_train.mean()) / X_train.std()  # Normalize using training statistics

        self.X = torch.FloatTensor(np.expand_dims(x, 1).astype(np.float32))
        self.Y = torch.FloatTensor(y).unsqueeze(1)
        
        print(f"{self.mode} dataset shape: {self.X.shape}, {self.Y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y  # Return tuple instead of list

# Training and evaluation updates
def train(epoch, net, trainloader, criterion, optimizer):
    net.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(trainloader)

def test(net, testloader, criterion):
    net.eval()
    test_loss = 0.0
    all_targets = []
    all_logits = []
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            all_logits.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate and compute metrics
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    y_prob = torch.sigmoid(all_logits).numpy()  # Convert logits to probabilities
    y_pred = (y_prob > 0.5).astype(int)  # Convert probabilities to binary labels

    y_true = all_targets.numpy()
    
    accuracy = accuracy_score(y_true, y_pred) * 100
    balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    avg_loss = test_loss / len(testloader)
    # print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    # print(f"Balanced Accuracy: {balanced_acc:.2f}%")
    # print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    # print(f"ROC-AUC: {roc_auc:.4f}")
    
    return avg_loss, accuracy, balanced_acc, precision, recall, f1, roc_auc

# Main execution flow
if __name__ == "__main__":
    # Initialize datasets and loaders
    trainset = NCANDA_Dataset(mode="train")
    testset = NCANDA_Dataset(mode="validation")
    print("Training on matrix type: ", matrix_type)
    
    trainloader = DataLoader(trainset, batch_size=20, shuffle=True, 
                           num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=20, shuffle=False,
                          num_workers=2, pin_memory=True)
    
    # Model setup
    # Initialize the network using an example input from the training set
    net = BrainNetCNN(trainset.X[0:1])
    net = net.to(device)
    if device.type == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    # Initialize weights using Kaiming initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu', a=0.33)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    net.apply(init_weights)
    
    # Training parameters
    # We have 280 of class 0 and 373 of class 1 (in training data)
    pos_weight = torch.tensor([280 / 373], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    
    # Training loop
    num_epochs = 200
    train_losses = []
    test_losses = []
    accuracies = []
    balanced_accs = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []

    for epoch in range(num_epochs):
        train_loss = train(epoch, net, trainloader, criterion, optimizer)
        test_metrics = test(net, testloader, criterion)
        test_loss, accuracy, balanced_acc, precision, recall, f1, roc_auc = test_metrics
        
        # Append metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)
        balanced_accs.append(balanced_acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        roc_aucs.append(roc_auc)
        
        # Print every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
            print(f"Accuracy: {accuracy:.2f}% | Balanced Acc: {balanced_acc:.2f}%")
            print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    
    # Save all metrics
    torch.save(net.state_dict(), "brainnetcnn_model.pth")
    np.savez("training_stats.npz", 
             train_losses=train_losses,
             test_losses=test_losses,
             accuracies=accuracies,
             balanced_accs=balanced_accs,
             precisions=precisions,
             recalls=recalls,
             f1_scores=f1_scores,
             roc_aucs=roc_aucs)