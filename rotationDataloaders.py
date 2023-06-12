import torch
import torchvision
def RotatedDataset(dataloader):
    """
        Takes a sequence of images and rotates, outputs a list of 3 tensors:
            0: Input Images (all rotations)
            1: Image Classes from Original Dataloader
            2: Image Rotation Angles (multiple of 90 degrees)
    """
    for out in dataloader:
        outputImages = torch.cat((out[0],
                                torchvision.transforms.functional.rotate(out[0],90),
                                torchvision.transforms.functional.rotate(out[0],180),
                                torchvision.transforms.functional.rotate(out[0],270)
                                ),0)
        outputClasses = torch.cat((out[1],out[1],out[1],out[1]),0)
        zeroes = torch.zeros((out[1].shape))
        outputAngles = torch.cat((zeroes, zeroes+1, zeroes+2, zeroes+3),0)
        yield outputImages, outputClasses, outputAngles
def RotatedDatasetOriginalClasses(dataloader):
    gen = RotatedDataset(dataloader)
    for out in gen:
        yield out[0],out[1]
def RotatedDatasetRotationAngleAsClasses(dataloader):
    gen = RotatedDataset(dataloader)
    for out in gen:
        yield out[0],out[2]