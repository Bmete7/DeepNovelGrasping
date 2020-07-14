# -*- coding: utf-8 -*-
"""
Created on Tue May 22 22:14:08 2020

@author: burak
"""




from __future__ import print_function, division
import os
import torch
import torchvision.transforms.functional as F
import pandas as pd
from skimage import io, transform

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# Ignore warnings
import warnings
import matplotlib.patches as patches
from PIL import Image

warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


class GraspDataset(Dataset):
    """Grasp dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.grasp_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.grasp_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.grasp_frame.iloc[idx, 1])
        image = io.imread(img_name)
        #image = Image.fromarray(image)
        #mage = Image.open(img_name)
        #img = Image.open(img_name)
        
        grasp = self.grasp_frame.iloc[idx, 2:].as_matrix()
        grasp = grasp.astype('float').reshape(-1, 5)
        sample = {'image': image, 'grasp': grasp}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

def show_grasps(ax,image, grasp):
    """Show image with grasp"""
    plt.imshow(image)
    rect = patches.Rectangle((grasp[0,0],grasp[0,1]),grasp[0,2],grasp[0,3],linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.pause(0.001)  # pause a bit so that plots are updated


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, grasp = sample['image'], sample['grasp']

        h, w = image.shape[:2]


        img = transform.resize(image, (self.output_size, self.output_size))
       

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        #grasp = grasp * [self.output_size / 1024]

        return {'image': img, 'grasp': grasp}




class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, grasp = sample['image'], sample['grasp']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'grasp': torch.from_numpy(grasp)}




class Normalize(object):


    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        im = tensor['image']
        im = F.to_pil_image(im)
        
        return F.normalize(im, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    


def show_grasps_batch(sample_batched):
    """Show image with grasp for a batch of samples."""
    images_batch, grasp_batch = \
    sample_batched['image'], sample_batched['grasp']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    ax = plt.gca()
    for i in range(batch_size):
        
        rect = patches.Rectangle((grasp_batch[i,0,0] + (i*224),grasp_batch[i,0,1]),grasp_batch[i,0,2],grasp_batch[i,0,3],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        

        plt.title('Batch from dataloader')
