import torch
import os
import cv2
import numpy as np
import torch.utils.data as data
from glob import glob
from configs import all_config
import albumentations
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from scipy.ndimage.morphology import distance_transform_edt
import matplotlib.pyplot as plt


class ValDataset(data.Dataset):
    def __init__(self,data,**config):
        super().__init__()
        self.data = data
        self.image_path = config['image_path']
        # self.mask_path = config['mask_path']
        self.num_classes = config['num_classes']
        self.transform = get_transform('val', **config)
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data = self.data[idx]
        image = cv2.imread(data)
        # print(type(image))
        augmented=self.transform(image=image)
        image = augmented['image']
        image = image.astype('float32') / 255
        image = image.transpose(2, 0, 1)
        return image,{'data': data}

        # mask = []

        # for i in range(self.num_classes):
        #     mask.append(cv2.imread(os.path.join(self.mask_path, str(i), data), cv2.IMREAD_GRAYSCALE)[..., None])
        # mask = np.dstack(mask)

def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == (i + 1) for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)

    """
    
    if radius < 0:
        return mask
    
    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    
    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)    
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap


class EdgeDataset(data.Dataset):
    def __init__(self,data,flag,**config) -> None:
        super().__init__()
        self.data = data
        self.flag = flag
        self.image_path = config['image_path']
        self.mask_path = config['mask_path']
        self.num_classes = config['num_classes']
        self.transform = get_transform(flag, **config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        image = cv2.imread(os.path.join(self.image_path, data))
        mask = []

        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_path, str(i), data), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)
        mask = mask.astype('float32') / 255.0

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        _edgemap = mask_to_onehot(mask[:,:,0], 1)

        _edgemap = onehot_to_binary_edges(_edgemap, 2, 1)

        edgemap = torch.from_numpy(_edgemap).float()

        image = image.astype('float32')
        image = image.transpose(2, 0, 1)
        
        mask = mask.transpose(2, 0, 1)

        return image, mask, edgemap

class Dataset(data.Dataset):
    def __init__(self, data, flag, **config):
        self.data = data
        self.flag = flag
        self.image_path = config['image_path']
        self.mask_path = config['mask_path']
        self.num_classes = config['num_classes']
        self.transform = get_transform(flag, **config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        image = cv2.imread(os.path.join(self.image_path, data))
        
        mask = []

        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_path, str(i), data), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = image.astype('float32') / 255
        image = image.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        return image, mask, {'data': data}

def get_transform(flag, **config):

    if flag == 'train':
        transform = Compose([transforms.RandomRotate90(), transforms.Flip(),
            OneOf([transforms.HueSaturationValue(), transforms.RandomBrightness(), transforms.RandomContrast()], p=1),
            transforms.Resize(config['height'], config['width']), transforms.Normalize()])
    elif flag == 'val' or flag == 'test':
        transform = Compose([transforms.Resize(config['height'], config['width']), transforms.Normalize( max_pixel_value=255.0 )])

    return transform


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

config = all_config('train', 'vessel', 'UNet')
total_data = [os.path.basename(i) for i in glob(config['image_path'] + "*.png")]
# train_data, val_data = train_test_split(total_data, test_size=0.2, random_state=41)


