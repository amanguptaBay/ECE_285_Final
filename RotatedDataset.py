import torch
import math
import torchvision.transforms as transforms
class RotatedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, use_rotation_label = False, use_both_labels = False):
        self.dataset = dataset
        self.use_rotation_label = use_rotation_label
        self.use_both_labels = use_both_labels

    def __getitem__(self, index):
        image, label = self.dataset[index]
        rotation_angle = index % 4 * 90 
        #TODO: Use_Rotation_Label logic has not been tested yet
        resized_image = transforms.functional.resize(image, (32,32))
        rotated_image = transforms.functional.rotate(resized_image, rotation_angle)
        labelImage = label
        labelRotation = index % 4
        label = (labelImage, labelRotation) if self.use_both_labels else (labelRotation if self.use_rotation_label else labelImage)
        return rotated_image, label

    def __len__(self):
        return len(self.dataset)
