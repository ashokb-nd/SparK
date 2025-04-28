# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Callable, Optional, Tuple
import random
import pandas as pd
import mlflow

import PIL.Image as PImage
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torchvision.io import read_image

from config import S3_CSV_PATH, LOCAL_DATA_ROOT, S3_DATASET_PATH_PREFIX

try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC




def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f: img: PImage.Image = PImage.open(f).convert('RGB')
    return img

class ImageNetDataset(Dataset):
    def __init__(
            self,
            local_root_dir: str,
            csv_path: str,  # column having paths should be named 'path'
            transform: Callable,
            s3_path_prefix: str,
            loader: Callable = pil_loader,
    ):
        """
        First look for the image in local_root_dir. If not found, then look for it in S3.
        Cache it to local_root_dir if found in S3.

        Note: This class assumes that the CSV file has a header and the column with image paths is named 'path'.
        """
        self.root_dir = local_root_dir
        self.transform = transform
        self.s3_path_prefix = s3_path_prefix
        self.loader = loader

        self.dataframe = pd.read_csv(csv_path, header=0)  # Assuming the CSV has a header

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Any:
        img_path = self.dataframe.loc[index, 'path']  # Assuming the column with image paths is named 'path'
        img_local_path = os.path.join(self.root_dir, img_path)

        # Attempt to load the image locally
        img = self._load_local_image(img_local_path)
        if img is not None:
            return self.transform(img)

        # If local loading fails, attempt to download from S3
        img = self._download_and_load_image(img_path, img_local_path)
        if img is not None:
            return self.transform(img)

        # If all else fails, pick another random image
        print(f"Failed to load image at index {index}. Picking a random image.")
        random_index = random.randint(0, len(self.dataframe) - 1)
        return self.__getitem__(random_index)

    def _load_local_image(self, img_local_path: str) -> Optional[Any]:
        """Try to load an image from the local file system."""
        try:
            return self.loader(img_local_path)
        except Exception as e:
            print(f"Error loading local image {img_local_path}: {e}")
            return None

    def _download_and_load_image(self, img_path: str, img_local_path: str) -> Optional[Any]:
        """Try to download an image from S3 and load it."""
        img_s3_path = os.path.join(self.s3_path_prefix, img_path)
        try:
            mlflow.artifacts.download_artifacts(img_s3_path, dst_path=img_local_path)
            return self.loader(img_local_path)
        except Exception as e:
            print(f"Error downloading image {img_s3_path}: {e}")
            return None
        


def build_dataset_to_pretrain(input_size) -> Dataset:
    """ 
    You may need to modify this function to return your own dataset.
    Define a new class, a subclass of `Dataset`, to replace our ImageNetDataset.
    Use dataset_path to build your image file path list.
    Use input_size to create the transformation function for your images, can refer to the `trans_train` blow. 
    
    :param dataset_path: the folder of dataset
    :param input_size: the input size (image resolution)
    :return: the dataset used for pretraining
    """
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.67, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    csv_path_local = os.path.join(os.path.abspath(LOCAL_DATA_ROOT), os.path.basename(S3_CSV_PATH))

    dataset_train = ImageNetDataset(
        local_root_dir = os.path.abspath(LOCAL_DATA_ROOT),
        csv_path = csv_path_local,
        transform = train_transforms,
        s3_path_prefix = S3_DATASET_PATH_PREFIX,
    )

    print_transform(train_transforms, '[pre-train]')
    return dataset_train

def print_transform(transform, s):
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')
