import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

import nnutils2

args = argparse.ArgumentParser(description='Train.py')
# Command Line ardguments

args.add_argument('--data_dir', dest="data_dir", action="store", default="./flowers/")
args.add_argument('--gpu', dest="gpu", action="store", default="gpu")
args.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
args.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
args.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
args.add_argument('--epochs', dest="epochs", action="store", type=int, default=4)
args.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
args.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

pa = args.parse_args()
where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs

print("***************Training Start complete***************")
print("Arguments:")
print(" data dir: ", pa.data_dir)
print(" save path: ", pa.save_dir)
print(" learning rate: ", pa.learning_rate)
print(" structure: ", pa.arch)
print(" dropout: ", pa.dropout)
print(" hidden layer: ", pa.hidden_units)
print(" power: ", pa.gpu)
print(" epoch: ", pa.epochs)
print("")

# Load Neural network
print("Loading Data-----------------------------------")
trainloader, v_loader, testloader, train_data = nnutils2.load_data(where)
print("Data Loaded-----------------------------------")
print("")

# Setup Neural network
print("Setting Neural Network-------------------------")
model, optimizer, criterion = nnutils2.nn_setup(structure,dropout,hidden_layer1,lr,power)
print("Neural Network Set-----------------------------")
print("")

# Train Neural network
print("Training Neural Network------------------------")
nnutils2.train_network(model, optimizer, criterion, epochs, 20, trainloader, power)
print("Neural Network Trained------------------------")
print("")

# Save
print("Saving Neural Network------------------------")
nnutils2.save_checkpoint(model,train_data, path,structure,hidden_layer1,dropout,lr)
print("Neural Network Saved-------------------------")
print("")


# load_data
print("Load Neural Network-------------------------")
nnutils2.load_checkpoint(path)
print()
print()
print("***************Training complete***************")
