import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
import fla

def split_data(X,y, test_size, stratify, val_size, random_state, batch_size):
    X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=test_size, random_state=random_state)
    if val_size!=0:
        X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, stratify=y_train, test_size=val_size, random_state=seed)
            
    train_dataset = a.npDataset(X_train,y_train)
    test_dataset = a.npDataset(X_test,y_test)
    if val_size!=0:
        val_dataset = a.npDataset(X_val,y_val)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    if val_size!=0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, val_loader

#assumes y labels are ordinal
def make_model(X,y, hidden_dims, attn_heads):
    
    model_params = {'input_dim': X.shape[1],
                    'hidden_dims': hidden_dims,
                    'output_dim': len(np.unique(y)),
                    'attn_heads': attn_heads,
                    'activation': nn.ReLU()
                    }
    
    model = fla.FLANN(**model_params)
    
    if len(np.unique(y))==2:
        y_counts = np.unique(y, return_counts=True)[1]
        weight = torch.tensor([y_counts[0]/y_counts[1]], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    else:
        weight = torch.tensor(compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y), dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weight)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return model, criterion, optimizer

def train_with_val(train_loader, val_loader, model, criterion, optimizer, patience):
    num_epochs = 500
    best_val_loss = float('inf')
        best_model = None
        patience = 10
        early_stop_counter = 0
        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                labels = labels.unsqueeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            val_losses = []
            for inputs, labels in val_loader:
                with torch.no_grad():
                    outputs = model(inputs)
                    labels = labels.unsqueeze(1)
                    val_loss = criterion(outputs, labels)
                    val_losses.append(val_loss.item())
            
            avg_val_loss = np.mean(val_losses)
            print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}')
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= patience:
                print(f'Early stopping after epoch {epoch+1} with validation loss {best_val_loss:.4f}')
                break
        model.load_state_dict(best_model)
    return model

def train_no_val(num_epochs, train_loader, model, criterion, optimizer):
    for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                labels = labels.unsqueeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()


def eval():
    return None

def save_model():

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Configuration for training')
    parser.add_argument("--model_config", type=str, help='Path to json model config file')
    parser.add_argument('train_eval_config', type=str, help='Path to json training routine config')
    
    args = parser.parse_args()
