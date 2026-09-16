from pathlib import Path

from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random
import itertools
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def img_cut(img):
    img[img < 0] = 0
    img[img > 255] = 255
    return img.astype(np.uint8)


class SemiDataset(Dataset):
    def __init__(self, cfg, name, root, mode, size=None, id_path=None, nsample=None):
        self.cfg = cfg
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        elif mode == 'val':
            with open(f'splits/%s/greatwall_beijing_{self.mode}.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()


        elif mode == 'test':
            root_path = Path(self.root)

            # 递归收集所有目录下的 tif/tiff
            tif_files = []
            for dirpath, _, filenames in os.walk(root_path, followlinks=True):
                d = Path(dirpath)
                for fn in filenames:
                    p = d / fn
                    if p.suffix.lower() in {".tif", ".png"}:
                        tif_files.append(p)

            # 存成相对路径，兼容中文目录名
            self.ids = [str(p.relative_to(root_path)).replace("\\", "/") for p in sorted(tif_files)]

            if len(self.ids) == 0:
                raise RuntimeError(f"No tif/tiff found under {self.root}.")
        else:
            raise ValueError(f'Unsupported mode: {mode}')

    def __getitem__(self, item):
        id = self.ids[item]

        if self.mode == 'test':
            img = Image.open(os.path.join(self.root, id)).convert('RGB')
            fre = np.fft.fft2(np.array(img), axes=(0, 1))
            fre_a = np.abs(fre)
            fre_p = np.angle(fre)

            constant = fre_p.mean()
            fre_ = fre_a * np.e ** (1j * constant)
            img_a = np.abs(np.fft.ifft2(fre_, axes=(0, 1)))
            img_a = Image.fromarray(img_cut(img_a))

            img = normalize(img)
            img_a = normalize(img_a)
            return img, img_a, id

        elif self.mode == 'val':
            img = Image.open(os.path.join(self.root, 'greatwall_beijing/', id.split(' ')[0])).convert('RGB')
            mask = np.array(Image.open(os.path.join(self.root, 'greatwall_beijing/', id.split(' ')[1])).convert('L'))
        else:
            if self.mode == 'train_l':
                img = Image.open(os.path.join(self.root, 'greatwall_beijing/', id.split(' ')[0])).convert('RGB')
                mask = np.array(Image.open(os.path.join(self.root, 'greatwall_beijing/', id.split(' ')[1])).convert('L'))
            else:
                img = Image.open(os.path.join(self.root, 'greatwall_beijing/', id.split(' ')[0])).convert('RGB')
                mask = np.zeros((img.height, img.width), dtype=np.uint8)

        # get Fourier amplitude and phase
        fre = np.fft.fft2(np.array(img), axes=(0, 1))
        fre_a = np.abs(fre)
        fre_p = np.angle(fre)

        # set the amplitude a constant
        constant = fre_p.mean()
        fre_ = fre_a * np.e ** (1j * constant)
        img_a = np.abs(np.fft.ifft2(fre_, axes=(0, 1)))
        img_a = Image.fromarray(img_cut(img_a))

        # mask = mask / 255
        mask = Image.fromarray(mask.astype(np.uint8))

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            img_a = normalize(img_a)
            return img, img_a, mask, id

        img, img_a, mask = crop(img, img_a, mask, self.size, 255)
        img, img_a, mask = hflip(img, img_a, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask), normalize(img_a)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)
        img_a_w, img_a_s1, img_a_s2 = deepcopy(img_a), deepcopy(img_a), deepcopy(img_a)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1, img_a_s1 = random_grayscale(img_s1, img_a_s1)
        img_s1, img_a_s1 = blur(img_s1, img_a_s1, p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2, img_a_s2 = random_grayscale(img_s2, img_a_s2)
        img_s2, img_a_s2 = blur(img_s2, img_a_s2, p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)
        img_a_s1 = normalize(img_a_s1)
        img_a_s2 = normalize(img_a_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255
        return normalize(img_w), normalize(img_a_w), img_s1, img_a_s1, img_s2, img_a_s2, ignore_mask

    def __len__(self):
        return len(self.ids)
