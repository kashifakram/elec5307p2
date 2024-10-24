import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
import PIL

class OurDataset(Dataset):
    def __init__(self, image_path_to_class_dict, class_to_number_dict, apply_resize=False):
        super().__init__()
        self.image_path_to_class_dict = image_path_to_class_dict
        self.class_to_number_dict = class_to_number_dict
        self.image_paths = list(self.image_path_to_class_dict.keys())
        self.resize = transforms.Resize((224, 224))  # Resize to 256x256 for ViT input
        self.to_tensor = transforms.ToTensor()  # Converts images to tensor
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])  # Standard normalization for ViT
        self.apply_resize = apply_resize

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Open the image and convert to RGB
        image = PIL.Image.open(self.image_paths[index]).convert('RGB')
        
        # Apply resize if needed
        if self.apply_resize:
            image = self.resize(image)
        
        # Convert the image to a PyTorch tensor and normalize
        image = self.to_tensor(image)
        image = self.normalize(image)
        
        # Get the corresponding label
        class_name = self.image_path_to_class_dict[self.image_paths[index]]
        label = self.class_to_number_dict[class_name]
        
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label


# class OurDataset(Dataset):
#   def __init__(self, image_path_to_class_dict, class_to_number_dict, apply_resize=False):
#     super().__init__()
#     self.image_path_to_class_dict = image_path_to_class_dict
#     self.class_to_number_dict = class_to_number_dict
#     self.image_paths = list(self.image_path_to_class_dict.keys())
#     self.resize = torchvision.transforms.Resize((256, 256))
#     self.apply_resize = apply_resize

#   def __len__(self):   # we can get the size of this dataset by using len()
#     return len(self.image_paths)

#   def __getitem__(self, index):  # we can access those data with indexes
#     image = PIL.Image.open(self.image_paths[index]).convert('RGB')
#     if self.apply_resize:
#       image = np.array(self.resize(image))
#     else:
#       image = np.array(image)
#     class_name = self.image_path_to_class_dict[self.image_paths[index]]
#     label = self.class_to_number_dict[class_name]
#     return (image, label)
