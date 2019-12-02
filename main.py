import os
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from config import Config
from data import DataLoaderTrain, ImageTransformer, cuda2numpy, DataLoaderValidate
from CoRRN import CoRRN
from loss import TotalLoss


cudnn.benchmark = True
config = Config()

model = CoRRN().cuda()
criterion = TotalLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
image_transformer = ImageTransformer(config)


data_loader_train = DataLoaderTrain(config)
print('Data length:', data_loader_train.data_len)
data_loader_train = torch.utils.data.DataLoader(dataset=data_loader_train, batch_size=config.batch_size)
data_loader_validate = DataLoaderValidate(config)
data_loader_validate = torch.utils.data.DataLoader(dataset=data_loader_validate, batch_size=config.batch_size_test)

for epoch in range(1, config.epochs+1):
    model.train()
    for idx_load, (image_m, image_b, image_g, image_r) in enumerate(data_loader_train):
        estimations = model(image_m)
        loss = criterion(estimations, {'r': image_r, 'g': image_g, 'b': image_b})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 1 == 0:
        print('epoch-{}, loss-{:.2e}.'.format(epoch, loss.item()))
        torch.save(model.state_dict(), './weights_CoRRN_datasetcoco.pth')
        # validate
        for idx_load, (image_m, image_b, image_g, image_r) in enumerate(data_loader_validate):
            model.eval()
            with torch.no_grad():
                estimations = model(image_m)
            for idx_in_batch in range(estimations['b'].shape[0]):
                est_r, est_g, est_b = estimations['r'][idx_in_batch], estimations['g'][idx_in_batch], estimations['b'][idx_in_batch]
                est_r, est_b = image_transformer.denormalize(est_r).astype(np.uint8), image_transformer.denormalize(est_b).astype(np.uint8)
                est_g = cuda2numpy(est_g*255).astype(np.uint8)
                image_m = image_transformer.denormalize(image_m[idx_in_batch]).astype(np.uint8)
                image_b = image_transformer.denormalize(image_b[idx_in_batch]).astype(np.uint8)
                image_r = image_transformer.denormalize(image_r[idx_in_batch]).astype(np.uint8)
                image_g = cuda2numpy(image_g[idx_in_batch]*255).astype(np.uint8)
                cv2.imwrite(os.path.join('val_images', str(idx_load), 'b_est.png'), cv2.cvtColor(est_b, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join('val_images', str(idx_load), 'g_est.png'), cv2.cvtColor(est_g, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join('val_images', str(idx_load), 'r_est.png'), cv2.cvtColor(est_r, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join('val_images', str(idx_load), 'm_image.png'), cv2.cvtColor(image_m, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join('val_images', str(idx_load), 'b_image.png'), cv2.cvtColor(image_b, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join('val_images', str(idx_load), 'g_image.png'), cv2.cvtColor(image_g, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join('val_images', str(idx_load), 'r_image.png'), cv2.cvtColor(image_r, cv2.COLOR_RGB2BGR))
