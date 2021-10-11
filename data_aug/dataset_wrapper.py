import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets
from data_aug import square_noise
#import cv2
import torch

np.random.seed(0)


def get_outlier_from_seg(input_tensor):
    if torch.nonzero(input_tensor, as_tuple=False).sum() != 0:
        return 1 #torch.Tensor([1]) #, dtype=torch.LongTensor)
    else:
        return 0 #torch.Tensor([0]) #, dtype=torch.LongTensor)

def get_simclr_pipeline_transform(size, s, aug_type='standard'):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    if aug_type == 'standard':
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            # GaussianBlur(c), #TODO: Check in which versions of SimCLR this is used (also ood)
                                            transforms.ToTensor()])
    elif aug_type == 'standard-rot':
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.RandomRotation(180),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            transforms.ToTensor()])
    elif aug_type == 'ce-no_crop':
        data_transforms = transforms.Compose([transforms.Resize(size=size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            transforms.ToTensor(),
                                            square_noise.masking_trafo])
    elif aug_type == 'square_noise-nor_cop':
        data_transforms = transforms.Compose([transforms.Resize(size=size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            transforms.ToTensor(),
                                            square_noise.masking_trafo, 
                                            AddNoise(1e-2)])
    return data_transforms


def get_grey_pipeline_transform(size, aug_type='standard'):
    if aug_type == 'standard':
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomRotation(degrees=180),
                                              transforms.ToTensor()])
    elif aug_type == 'standard-blur':
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomRotation(degrees=180),
                                              transforms.ToTensor(), 
                                              SmoothTensor(kernel_size=int(0.1 * size), channels=1)])                            
    elif aug_type == 'ce':
        data_transforms = transforms.Compose([transforms.Resize(size), 
                                                transforms.ToTensor(), 
                                                square_noise.masking_trafo])
    elif aug_type == 'ce-no_crop':
        data_transforms = transforms.Compose([transforms.Resize(size), 
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomRotation(degrees=180),
                                                transforms.ToTensor(), 
                                                square_noise.masking_trafo])
    elif aug_type == 'ce-blur':
        data_transforms = transforms.Compose([transforms.Resize(size), 
                                                transforms.ToTensor(), 
                                                SmoothTensor(kernel_size=int(0.1 * size), channels=1), 
                                                square_noise.masking_trafo])
    elif aug_type == 'random_crop':
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                                transforms.ToTensor()])
    elif aug_type == 'random_crop-ce':
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                                transforms.ToTensor(),
                                                square_noise.masking_trafo])


    else:
        raise NotImplementedError

    return data_transforms



class DoubleHeadedDataTransform(object):
    def __init__(self, transform_i, transform_j):
        self.transform_i = transform_i
        self.transform_j = transform_j

    def __call__(self, sample):

        xi = self.transform_i(sample)
        xj = self.transform_j(sample)
        return xi, xj


class AddNoise(object):
    def __init__(self, eps):
        self.eps = eps 
    
    def __call__(self, sample):
        return torch.clamp(sample+torch.randn(sample.shape, device=sample.device)*self.eps, min=0, max=1)

class SmoothTensor(object):
    def __init__(self, kernel_size, channels=1, min=0.1, max=2.0):
        self.min = min
        self.max = max 
        self.channels = channels
        if kernel_size % 2 == 0:
            kernel_size -= 1
        self.kernel_size = kernel_size
        self.gaussian_filter = torch.nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=False,
            padding=kernel_size // 2,
        )

    def __call__(self, sample):
        x_cord = torch.arange(self.kernel_size)
        x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (self.kernel_size - 1) / 2.0
        sigma = (self.max - self.min) * np.random.random_sample() + self.min
        variance = sigma ** 2.0

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        import math

        gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2.0 * variance)
        )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

        return self.gaussian_filter(sample[None])[0]



# class GaussianBlur(object):
#     # Implements Gaussian blur as described in the SimCLR paper
#     def __init__(self, kernel_size, min=0.1, max=2.0):
#         self.min = min
#         self.max = max
#         # kernel size is set to be 10% of the image height/width
#         self.kernel_size = kernel_size
#
#     def __call__(self, sample):
#         sample = np.array(sample)
#
#         # blur the image with a 50% chance
#         prob = np.random.random_sample()
#
#         if prob < 0.5:
#             sigma = (self.max - self.min) * np.random.random_sample() + self.min
#             sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
#
#         return sample