import torch.utils.data as data
import torchvision.transforms as tfs
from PIL import Image
from torchvision.transforms import functional as FF
import torchvision.transforms as transforms
import sys
sys.path.append('.')
sys.path.append('..')
import random
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from option import opt
from os import listdir
from os.path import isfile
import numpy as np
from random import sample

BS = opt.bs
print(BS)
crop_size = 'whole_img'
if opt.crop:
    crop_size = opt.crop_size


def tensorShow(tensors, titles=None):
    '''
        t:BCWH
        '''
    fig = plt.figure()
    for tensor, tit, i in zip(tensors, titles, range(len(tensors))):
        img = make_grid(tensor)
        npimg = img.numpy()
        ax = fig.add_subplot(211 + i)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(tit)
    plt.show()


class RESIDE_Dataset(data.Dataset):
    def __init__(self, path, train, size=crop_size, format=format):
        super(RESIDE_Dataset, self).__init__()
        self.size = size
        print('crop size', size)
        self.train = train
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'haze'))
        self.haze_dir = os.path.join(path, 'haze')
        self.haze_imgs = [os.path.join(path, 'haze', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'clear')
        if opt.trainset == "ots_inst_train":
            self.transforms = transforms.Compose([
                transforms.Resize((opt.fineSize, opt.fineSize), interpolation=2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])

            ])
    def __getitem__(self, index):
        if opt.trainset == "its_inst_train":
            haze, clear = self.get_inst_dataset(index)
            return haze, clear
        haze = Image.open(self.haze_imgs[index])
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                haze = Image.open(self.haze_imgs[index])

        img = self.haze_imgs[index]
        id = img.split('/')[-1].split('_')[0]
        clear_name = id + self.format
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        clear = tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        return haze, clear

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        data = tfs.ToTensor()(data)
        data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        target = tfs.ToTensor()(target)
        return data, target

    def get_inst_dataset(self, index):
        img = self.haze_imgs[index]
        id = img.split('/')[0].split('_')[0]
        clear_name = id + self.format
        clear_image_path = os.path.join(os.path.join(self.clear_dir, clear_name))
        npz_path = os.path.join(self.clear_dir + '_bbox')
        pred_info_path = os.path.join(npz_path + "/" + id + '.npz')
        pred_bbox = gen_maskrcnn_bbox_fromPred(pred_info_path)

        index_list = range(len(pred_bbox))
        if len(pred_bbox) == 0:
            startx = 0
            starty = 0
            endx = 530
            endy = 413
        else:
            index_list = sample(index_list, 1)
            startx, starty, endx, endy = pred_bbox[index_list[0]]

        haze_image_path = self.haze_imgs[index]
        rgb_img = Image.open(haze_image_path)
        inst_img = rgb_img.crop((startx, starty, endx, endy))
        haze = inst_img

        rgb_img = Image.open(clear_image_path)
        clear = rgb_img.crop((startx, starty, endx, endy))
        haze = self.transforms(haze.convert("RGB"))
        clear = self.transforms(clear.convert("RGB"))
        return haze, clear

    def __len__(self):
        return len(self.haze_imgs)

def gen_maskrcnn_bbox_fromPred(pred_data_path, box_num_upbound=-1):
    '''
    ## Arguments:
    - pred_data_path: Detectron2 predict results
    - box_num_upbound: object bounding boxes number. Default: -1 means use all the instances.
    '''
    pred_data = np.load(pred_data_path)
    assert 'bbox' in pred_data
    assert 'scores' in pred_data
    pred_bbox = pred_data['bbox'].astype(np.int32)
    if box_num_upbound > 0 and pred_bbox.shape[0] > box_num_upbound:
        pred_scores = pred_data['scores']
        index_mask = np.argsort(pred_scores, axis=0)[pred_scores.shape[0] - box_num_upbound: pred_scores.shape[0]]
        pred_bbox = pred_bbox[index_mask]
    return pred_bbox


def get_box_info(pred_bbox, original_shape, final_size):
    assert len(pred_bbox) == 4
    resize_startx = int(pred_bbox[0] / original_shape[0] * final_size)
    resize_starty = int(pred_bbox[1] / original_shape[1] * final_size)
    resize_endx = int(pred_bbox[2] / original_shape[0] * final_size)
    resize_endy = int(pred_bbox[3] / original_shape[1] * final_size)
    rh = resize_endx - resize_startx
    rw = resize_endy - resize_starty
    if rh < 1:
        if final_size - resize_endx > 1:
            resize_endx += 1
        else:
            resize_startx -= 1
        rh = 1
    if rw < 1:
        if final_size - resize_endy > 1:
            resize_endy += 1
        else:
            resize_starty -= 1
        rw = 1
    L_pad = resize_startx
    R_pad = final_size - resize_endx
    T_pad = resize_starty
    B_pad = final_size - resize_endy
    return [L_pad, R_pad, T_pad, B_pad, rh, rw]


def read_to_pil(img_path):
    '''
    return: pillow image object HxWx3
    '''
    out_img = Image.open(img_path)
    if len(np.asarray(out_img).shape) == 2:
        out_img = np.stack([np.asarray(out_img), np.asarray(out_img), np.asarray(out_img)], 2)
        out_img = Image.fromarray(out_img)
    return out_img


import os
pwd = os.getcwd()
print(pwd)
path = '/home/amax/Desktop/hra(fusion)/data'  # path to your 'data' folder

ITS_train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/ITS/',train=False,size=crop_size, format='.jpg'),batch_size=BS,shuffle=True)
ITS_test_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/SOTS/',train=False,size=crop_size,format='.png'),batch_size=1,shuffle=False)
OTS_train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/OTS/',train=True,size=crop_size,format='.jpg'),batch_size=BS,shuffle=True)
OTS_test_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/SOTS/',train=True,size=crop_size,format='.png'),batch_size=1,shuffle=True)

# OTS_inst_train_loader = DataLoader(dataset=RESIDE_Dataset(path + '/RESIDE/OTS', train=True, format='.jpg'),
#                                    batch_size=BS, shuffle=True)
# OTS_inst_test_loader = DataLoader( dataset=RESIDE_Dataset(path + '/RESIDE/SOTS', train=False, size='whole img', format='.jpg'), batch_size=1,
# #     shuffle=False)
# ITS_inst_train_loader = DataLoader(dataset=RESIDE_Dataset(path + '/RESIDE/OTS', train=True, format='.jpg'),
#                                    batch_size=BS, shuffle=True)
# ITS_inst_test_loader = DataLoader( dataset=RESIDE_Dataset(path + '/RESIDE/SOTS', train=False, size='whole img', format='.jpg'), batch_size=1,
# #     shuffle=False)


if __name__ == "__main__":
    pass
