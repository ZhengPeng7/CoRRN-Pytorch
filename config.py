import torch


class Config(object):
    def __init__(self):
        self.dir_sir2 = 'datasets'
        self.dir_coco='../COCO_Rimage_dataset_Good'
        self.dir_test = ''
        self.epochs = 200
        self.learning_rate = 1e-4
        self.batch_size = 8
        self.batch_size_test = 1
        self.preproc = {'resize': (384, 512), 'random_hsi': 0.0, 'hflip': True, 'vflip': True}
        self.normalization_mean = (0, 0, 0)
        self.normalization_std = (1, 1, 1)
