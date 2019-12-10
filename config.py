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
        self.normalization_mean, self.normalization_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.num_val_coco = 1000
        self.losses_train = []
        self.losses_validate_coco = []
        self.losses_validate_sir2 = []
