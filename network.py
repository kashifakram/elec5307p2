'''
this script is for the network of Project 2.

You can change any parts of this code

-------------------------------------------
'''

import torch
import torch.nn as nn
import timm

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        num_classes = 22
        # Load the pre-trained ViT model
        self.vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        
        # Replace the classification head (final layer) to match the number of classes
        self.vit_model.head = nn.Linear(self.vit_model.head.in_features, num_classes)

    def forward(self, x):
        # Forward pass through the Vision Transformer model
        return self.vit_model(x)
