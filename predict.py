import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

import nnutils2

#Command Line Arguments

args = argparse.ArgumentParser(description='predict-file')
args.add_argument('input_img', default='paind-project/flowers/test/1/image_06752.jpg', nargs='*', action="store", type = str)
args.add_argument('checkpoint', default='/home/workspace/paind-project/checkpoint.pth', nargs='*', action="store",type = str)
args.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
args.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
args.add_argument('--gpu', default="gpu", action="store", dest="gpu")

print("Arguments Loaded.")
pa = args.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
input_img = pa.input_img
path = pa.checkpoint

print("Loading data.")
training_loader, testing_loader, validation_loader = nnutils2.load_data()

nnutils2.load_checkpoint(path)

with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)

print("Start Prediction Code.")
probabilities = nnutils2.predict(path_image, model, number_of_outputs, power)

labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])

print("Print Probability.")
i=0
while i < number_of_outputs:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i += 1

print("Prediction completed.")
