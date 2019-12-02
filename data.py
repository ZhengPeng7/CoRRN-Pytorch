import os
from random import shuffle
import cv2
from PIL import Image
import numpy as np
from skimage import io, color
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


def get_paths_sir2(dir_sir2='datasets'):
    paths_1 = []
    root_path_1 = os.path.join(dir_sir2, 'SIR2/1. Solid_Object(13,520 KB)/SolidObjectDataset')
    for rp in sorted([t for t in os.listdir(root_path_1) if os.path.isdir(os.path.join(root_path_1, t))], key=lambda x: int(x)):
        path_num = os.path.join(root_path_1, rp)
        for p in os.listdir(path_num):
            path_focus_thick = os.path.join(path_num, p)
            for p in sorted(os.listdir(path_focus_thick), key=lambda x: int(x)):
                path_num2 = os.path.join(path_focus_thick, p)
                for p in os.listdir(path_num2):
                    paths_1.append(os.path.join(path_num2, p).replace('\\', '/'))
    paths_2 = []
    root_path_2 = os.path.join(dir_sir2, 'SIR2/2. Postcard(173,538 KB)/Postcard Dataset')
    for rp in [t for t in os.listdir(root_path_2) if os.path.isdir(os.path.join(root_path_2, t))]:
        path_focus_thick = os.path.join(root_path_2, rp)
        for p in os.listdir(path_focus_thick):
            path_alphabet = os.path.join(path_focus_thick, p)
            for p in sorted(os.listdir(path_alphabet), key=lambda x: int(x)):
                path_num = os.path.join(path_alphabet, p)
                for p in os.listdir(path_num):
                    paths_2.append(os.path.join(path_num, p).replace('\\', '/'))
    paths_3 = []
    root_path_3 = os.path.join(dir_sir2, 'SIR2/3. Wild_Scene(4,154 KB)/withgt')
    for rp in sorted(os.listdir(root_path_3), key=lambda x: int(x)):
        path_num = os.path.join(root_path_3, rp)
        for p in os.listdir(path_num):
            paths_3.append(os.path.join(path_num, p).replace('\\', '/'))
    paths_b, paths_m, paths_r = [], [], []
    for paths in [paths_1, paths_2, paths_3]:
        for p in paths:
            file_name = p.split('/')[-1].split('.')[0]
            if 'g' in file_name.replace('rs', ''):
                paths_b.append(p)
            if 'm' in file_name.replace('rs', ''):
                paths_m.append(p)
            if 'r' in file_name.replace('rs', ''):
                paths_r.append(p)
    return paths_r, paths_b, paths_m


def get_paths_coco(dir_coco='../COCO_Rimage_dataset_Good'):
    paths_b = [
        os.path.join(dir_coco, 'B', f) for f in sorted(
            os.listdir(os.path.join(dir_coco, 'B')),
            key=lambda x: int(x[:-4].split('_')[0])*4+int(x[:-4].split('_')[1])
        )]
    paths_m = [
        os.path.join(dir_coco, 'M', f) for f in sorted(
            os.listdir(os.path.join(dir_coco, 'M')),
            key=lambda x: int(x[:-4].split('_')[0])*4+int(x[:-4].split('_')[1])
        )]
    paths_r = [
        os.path.join(dir_coco, 'R', f) for f in sorted(
            os.listdir(os.path.join(dir_coco, 'R')),
            key=lambda x: int(x[:-4].split('_')[0])*4+int(x[:-4].split('_')[1])
        )]
    return paths_r, paths_b, paths_m


def gradient(image):
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    magnitude = np.sqrt(gradient_x**2.0 + gradient_y**2.0)

    return magnitude


def cuda2numpy(tensor):
    array = tensor.detach().cpu().squeeze().numpy()
    return array


class ImageTransformer(object):
    def __init__(self, config):
        self.normalization_mean, self.normalization_std = config.normalization_mean, config.normalization_std
        self.image2tensor = transforms.ToTensor()
        self.normalizer = transforms.Normalize(config.normalization_mean, config.normalization_std)
        # Normalization matrix for fast denormalization
        self.normalization_mean_matrix, self.normalization_std_matrix = None, None

    def normalize(self, image):
        return self.normalizer(self.image2tensor(image))

    def denormalize(self, tensor):
        if self.normalization_mean_matrix is None or self.normalization_mean_matrix.shape != tensor.shape:
            # Initialization on the norm matrices
            self.normalization_mean_matrix = torch.zeros_like(tensor)
            self.normalization_std_matrix = torch.zeros_like(tensor)
            for idx_channel in range(self.normalization_mean_matrix.shape[0]):
                self.normalization_mean_matrix[idx_channel, ...] += self.normalization_mean[idx_channel]
                self.normalization_std_matrix[idx_channel, ...] += self.normalization_std[idx_channel]
        tensor = tensor * self.normalization_std_matrix + self.normalization_mean_matrix
        array = cuda2numpy(tensor).transpose(1, 2, 0) * 255
        return array


class DataLoaderTrain(data.Dataset):
    def __init__(self, config):
        self.paths_r, self.paths_b, self.paths_m = get_paths_coco(config.dir_coco)  # get_paths_sir2(config.dir_sir2)
        self.data_len = len(self.paths_b)

        self.image_size = config.preproc['resize'] if 'resize' in config.preproc else (96, 128)
        self.random_hsi = config.preproc['random_hsi']
        self.if_hflip = 'hflip' in config.preproc and config.preproc['hflip']
        self.if_vflip = 'vflip' in config.preproc and config.preproc['vflip']
        self.transformer_resize = transforms.Resize(self.image_size)
        self.transformer_image2tensor = transforms.ToTensor()
        self.transformer_norm = transforms.Normalize(config.normalization_mean, config.normalization_std)

        self.permulation = np.random.permutation(self.data_len)
        self.images_mbgr = []
        self.load_all = self.data_len < 500

        if self.load_all:
            for i in range(self.data_len):
                # No augmentation on g and b
                path_r, path_b, path_m = self.paths_r[i], self.paths_b[i], self.paths_m[i]

                image_m = Image.fromarray(io.imread(path_m).astype(np.uint8)).convert('RGB')
                image_b = Image.fromarray(io.imread(path_b).astype(np.uint8)).convert('RGB')
                image_r = Image.fromarray(io.imread(path_r).astype(np.uint8)).convert('RGB')
                image_g = Image.fromarray(gradient(io.imread(path_b, as_gray=True)))

                image_m = self.transformer_resize(image_m)
                image_b = self.transformer_resize(image_b)
                image_r = self.transformer_resize(image_r)
                image_g = self.transformer_resize(image_g)

                self.images_mbgr.append([image_m, image_b, image_g, image_r])

    def __getitem__(self, index):
        if self.load_all:
            image_m, image_b, image_g, image_r = self.images_mbgr[self.permulation[index]]
        else:
            path_r, path_b, path_m = self.paths_r[self.permulation[index]], self.paths_b[self.permulation[index]], self.paths_m[self.permulation[index]]

            image_m = Image.fromarray(io.imread(path_m).astype(np.uint8)).convert('RGB')
            image_b = Image.fromarray(io.imread(path_b).astype(np.uint8)).convert('RGB')
            image_r = Image.fromarray(io.imread(path_r).astype(np.uint8)).convert('RGB')
            image_g = Image.fromarray(gradient(io.imread(path_b, as_gray=True)))

            image_m = self.transformer_resize(image_m)
            image_b = self.transformer_resize(image_b)
            image_r = self.transformer_resize(image_r)
            image_g = self.transformer_resize(image_g)
        if self.random_hsi:
            image_m = transforms.ColorJitter(brightness=self.random_hsi, contrast=self.random_hsi, saturation=self.random_hsi, hue=self.random_hsi)(image_m)
        if self.if_hflip and np.random.random() > 0.5:
            image_m = F.hflip(image_m)
            image_b = F.hflip(image_b)
            image_g = F.hflip(image_g)
            image_r = F.hflip(image_r)
        if self.if_vflip and np.random.random() > 0.5:
            image_m = F.vflip(image_m)
            image_b = F.vflip(image_b)
            image_g = F.vflip(image_g)
            image_r = F.vflip(image_r)
        image_m = self.transformer_norm(self.transformer_image2tensor(image_m)).float().cuda()
        image_b = self.transformer_norm(self.transformer_image2tensor(image_b)).float().cuda()
        image_r = self.transformer_norm(self.transformer_image2tensor(image_r)).float().cuda()
        image_g = self.transformer_image2tensor(image_g).float().cuda()
        return image_m, image_b, image_g, image_r

    def __len__(self):
        return self.data_len

    def shuffle(self):
        self.permulation = np.random.permutation(self.data_len)


class DataLoaderValidate(data.Dataset):
    def __init__(self, config):
        self.paths_r, self.paths_b, self.paths_m = get_paths_sir2(config.dir_sir2)
        self.paths_rbm = list(zip(self.paths_r, self.paths_b, self.paths_m))
        shuffle(self.paths_rbm)
        self.paths_r, self.paths_b, self.paths_m = zip(*self.paths_rbm)
        self.data_len = 10
        self.image_size = (384, 512)

        self.transformer_resize = transforms.Resize(self.image_size)
        self.transformer_image2tensor = transforms.ToTensor()
        self.transformer_norm = transforms.Normalize(config.normalization_mean, config.normalization_std)

        self.images_mbgr = []

        for i in range(self.data_len):
            # No augmentation on g and b
            path_r, path_b, path_m = self.paths_r[i], self.paths_b[i], self.paths_m[i]

            image_m = Image.fromarray(io.imread(path_m).astype(np.uint8)).convert('RGB')
            image_b = Image.fromarray(io.imread(path_b).astype(np.uint8)).convert('RGB')
            image_r = Image.fromarray(io.imread(path_r).astype(np.uint8)).convert('RGB')
            image_g = Image.fromarray(gradient(io.imread(path_b, as_gray=True)))

            image_m = self.transformer_resize(image_m)
            image_b = self.transformer_resize(image_b)
            image_r = self.transformer_resize(image_r)
            image_g = self.transformer_resize(image_g)

            self.images_mbgr.append([image_m, image_b, image_g, image_r])

    def __getitem__(self, index):
        image_m, image_b, image_g, image_r = self.images_mbgr[index]
        image_m = self.transformer_norm(self.transformer_image2tensor(image_m)).float().cuda()
        image_b = self.transformer_norm(self.transformer_image2tensor(image_b)).float().cuda()
        image_r = self.transformer_norm(self.transformer_image2tensor(image_r)).float().cuda()
        image_g = self.transformer_image2tensor(image_g).float().cuda()
        return image_m, image_b, image_g, image_r

    def __len__(self):
        return self.data_len
