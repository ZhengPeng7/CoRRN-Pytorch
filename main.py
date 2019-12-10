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
model.load_state_dict(torch.load('weights/CoRRN_coco_ep8_loss[train:2.228e+00-val_coco:2.219e+00-val_sir2:2.363e+00].pth'))
criterion = TotalLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate/10, betas=(0.9, 0.999))
image_transformer = ImageTransformer(config)


data_loader_train = DataLoaderTrain(config)
print('Data length:', data_loader_train.data_len)
data_loader_train = torch.utils.data.DataLoader(dataset=data_loader_train, batch_size=config.batch_size)
data_loader_validate = DataLoaderValidate(config)
data_loader_validate = torch.utils.data.DataLoader(dataset=data_loader_validate, batch_size=config.batch_size_test)
data_loader_validate_sir2 = DataLoaderValidate(config, dataset='sir2')
data_loader_validate_sir2 = torch.utils.data.DataLoader(dataset=data_loader_validate_sir2, batch_size=config.batch_size_test)


if not os.path.exists('weights'):
    os.makedirs('weights')

for epoch in range(9, config.epochs+1):
    model.train()
    loss_train = []
    for idx_load, (image_m, image_b, image_g, image_r) in enumerate(data_loader_train):
        estimations = model(image_m)
        loss = criterion(estimations, {'r': image_r, 'g': image_g, 'b': image_b})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
    if epoch % 1 == 0:
        print('epoch:{}, loss_train:{:.3e}.'.format(epoch, np.mean(loss_train)))
        # validate
        model.eval()
        loss_validate_coco = []
        for idx_load, (image_m, image_b, image_g, image_r) in enumerate(data_loader_validate):
            with torch.no_grad():
                estimations = model(image_m)
                loss = criterion(estimations, {'r': image_r, 'g': image_g, 'b': image_b})
            loss_validate_coco.append(loss.item())
            for idx_in_batch in range(estimations['b'].shape[0]):
                est_r, est_g, est_b = estimations['r'][idx_in_batch], estimations['g'][idx_in_batch], estimations['b'][idx_in_batch]
                est_r, est_b = image_transformer.denormalize(est_r).astype(np.uint8), image_transformer.denormalize(est_b).astype(np.uint8)
                est_g = cuda2numpy(est_g*255).astype(np.uint8)
                image_m = image_transformer.denormalize(image_m[idx_in_batch]).astype(np.uint8)
                image_b = image_transformer.denormalize(image_b[idx_in_batch]).astype(np.uint8)
                image_r = image_transformer.denormalize(image_r[idx_in_batch]).astype(np.uint8)
                image_g = cuda2numpy(image_g[idx_in_batch]*255).astype(np.uint8)
                if not os.path.exists(os.path.join('val_images_coco', str(idx_load))):
                    os.makedirs(os.path.join('val_images_coco', str(idx_load)))
                cv2.imwrite(os.path.join('val_images_coco', str(idx_load), 'b_est.png'), cv2.cvtColor(est_b, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join('val_images_coco', str(idx_load), 'g_est.png'), cv2.cvtColor(est_g, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join('val_images_coco', str(idx_load), 'r_est.png'), cv2.cvtColor(est_r, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join('val_images_coco', str(idx_load), 'm_image.png'), cv2.cvtColor(image_m, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join('val_images_coco', str(idx_load), 'b_image.png'), cv2.cvtColor(image_b, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join('val_images_coco', str(idx_load), 'g_image.png'), cv2.cvtColor(image_g, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join('val_images_coco', str(idx_load), 'r_image.png'), cv2.cvtColor(image_r, cv2.COLOR_RGB2BGR))
        loss_validate_sir2 = []
        for idx_load, (image_m, image_b, image_g, image_r) in enumerate(data_loader_validate_sir2):
            with torch.no_grad():
                estimations = model(image_m)
                loss = criterion(estimations, {'r': image_r, 'g': image_g, 'b': image_b})
            loss_validate_sir2.append(loss.item())
            for idx_in_batch in range(estimations['b'].shape[0]):
                est_r, est_g, est_b = estimations['r'][idx_in_batch], estimations['g'][idx_in_batch], estimations['b'][idx_in_batch]
                est_r, est_b = image_transformer.denormalize(est_r).astype(np.uint8), image_transformer.denormalize(est_b).astype(np.uint8)
                est_g = cuda2numpy(est_g*255).astype(np.uint8)
                image_m = image_transformer.denormalize(image_m[idx_in_batch]).astype(np.uint8)
                image_b = image_transformer.denormalize(image_b[idx_in_batch]).astype(np.uint8)
                image_r = image_transformer.denormalize(image_r[idx_in_batch]).astype(np.uint8)
                image_g = cuda2numpy(image_g[idx_in_batch]*255).astype(np.uint8)
                if not os.path.exists(os.path.join('val_images_sir2', str(idx_load))):
                    os.makedirs(os.path.join('val_images_sir2', str(idx_load)))
                cv2.imwrite(os.path.join('val_images_sir2', str(idx_load), 'b_est.png'), cv2.cvtColor(est_b, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join('val_images_sir2', str(idx_load), 'g_est.png'), cv2.cvtColor(est_g, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join('val_images_sir2', str(idx_load), 'r_est.png'), cv2.cvtColor(est_r, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join('val_images_sir2', str(idx_load), 'm_image.png'), cv2.cvtColor(image_m, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join('val_images_sir2', str(idx_load), 'b_image.png'), cv2.cvtColor(image_b, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join('val_images_sir2', str(idx_load), 'g_image.png'), cv2.cvtColor(image_g, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join('val_images_sir2', str(idx_load), 'r_image.png'), cv2.cvtColor(image_r, cv2.COLOR_RGB2BGR))
        config.losses_train.append(np.mean(loss_train))
        config.losses_validate_coco.append(np.mean(loss_validate_coco))
        config.losses_validate_sir2.append(np.mean(loss_validate_sir2))
        print('\t\tloss_validate_[coco/sir2]:{:.3e}/{:.3e}.'.format(config.losses_validate_coco[-1], config.losses_validate_sir2[-1]))
        torch.save(model.state_dict(), './weights/CoRRN_coco_ep{}_loss[train:{:.3e}-val_coco:{:.3e}-val_sir2:{:.3e}].pth'.format(
            epoch, config.losses_train[-1], config.losses_validate_coco[-1], config.losses_validate_sir2[-1]
        ))
        np.savetxt('weights/losses_train.txt', config.losses_train)
        np.savetxt('weights/losses_validate_coco.txt', config.losses_validate_coco)
        np.savetxt('weights/losses_validate_sir2.txt', config.losses_validate_sir2)
