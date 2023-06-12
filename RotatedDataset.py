import torch
import math
import torchvision.transforms as transforms
class RotatedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, use_rotation_label = False):
        self.dataset = dataset
        self.use_rotation_label = use_rotation_label

    def __getitem__(self, index):
        image, label = self.dataset[math.floor(index/4)]
        rotation_angle = index % 4 * 90 
        rotated_image = transforms.functional.rotate(image, rotation_angle)
        label = index % 4 if self.use_rotation_label else label
        return rotated_image, label

    def __len__(self):
        return len(self.dataset)*4
