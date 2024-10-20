'''
this script is for the training code of Project 2..

-------------------------------------------
INTRO:
You can change any parts of this code

-------------------------------------------

NOTE:
this file might be incomplete, feel free to contact us
if you found any bugs or any stuff should be improved.
Thanks :)

Email:
yliu9097@uni.sydney.edu.au, yili7216@uni.sydney.edu.au
'''

# import the packages
import argparse
import logging
import sys
import time
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from our_dataset import OurDataset

from network import Network # the network you used

parser = argparse.ArgumentParser(description= \
                                     'scipt for training of project 2')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Used when there are cuda installed.')
args = parser.parse_args()

# training process. 
def train_net(net, trainloader, valloader):
########## ToDo: Your codes goes below #######
    val_accuracy = 0
    # val_accuracy is the validation accuracy of each epoch. You can save your model base on the best validation accuracy.

    return val_accuracy

##############################################

############################################
# Transformation definition
# NOTE:
# Write the train_transform here. We recommend you use
# Normalization, RandomCrop and any other transform you think is useful.

train_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

####################################

####################################
# Define the training dataset and dataloader.
# You can make some modifications, e.g. batch_size, adding other hyperparameters, etc.

# Folder path containing the dataset (e.g., '../dataset/')
dataset_dir = '/train'  # Replace with the path to your dataset

# Get all class names (folder names) from the directory
class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])

# Create a class-to-number dictionary
class_to_number_dict = {class_name: i for i, class_name in enumerate(class_names)}

# Create an image path-to-class dictionary
image_path_to_class_dict = {}
for class_name in class_names:
    class_folder = os.path.join(dataset_dir, class_name)
    for image_file in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_file)
        if os.path.isfile(image_path):
            image_path_to_class_dict[image_path] = class_name


train_image_path = '../train/' 
validation_image_path = '../validation/' 


our_dataset = OurDataset(image_path_to_class_dict, class_to_number_dict, apply_resize=True)

# Split the dataset into train and validation sets (80% train, 20% val)
train_size = int(0.8 * len(our_dataset))
val_size = len(our_dataset) - train_size

trainset, valset = torch.utils.data.random_split(our_dataset, [train_size, val_size])


trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                         shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(valset, batch_size=32,
                                         shuffle=True, num_workers=4)
####################################

# ==================================
# use cuda if called with '--cuda'.

network = Network()
if args.cuda:
    network = network.cuda()

num_classes = len(class_names)  # Number of classes from the folder names
network = Network(num_classes)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(network.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train the model using the train_net function from the previous code
val_acc = train_net(network, trainloader, valloader, criterion, optimizer, num_epochs=10, device=device)


# train and eval your trained network
# you have to define your own 
val_acc = train_net(network, trainloader, valloader)


print("final validation accuracy:", val_acc)

# ==================================
