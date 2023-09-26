import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import torchvision.transforms as tfs

import numpy as np
import torch
from skimage import img_as_ubyte, io
from torch import nn
from torch.backends import cudnn
from torch.optim import lr_scheduler

from net.models.HRA import HRA_INST, HRA_Fusion, HRA
from net.util import util

class FusionModel(nn.Module):
    def initialize(self, opt):
        opt.isTrain = True
        self.isTrain = opt.isTrain
        self.model_names = ['G', 'GF', 'GComp']
        self.optimizers = []
        self.device = opt.device
        # load/define networks
        num_in = opt.input_nc + opt.output_nc + 1

        self.visual_names = []
        self.blocks = opt.blocks

        self.netG = HRA_INST( blocks=opt.blocks).to(opt.device)
        self.netG = self.netG.cuda()
        device_ids = [0, 1]  # id为0和1的两块显卡
        self.netG = torch.nn.DataParallel(self.netG)
        cudnn.benchmark = True
        self.netG.eval()

        self.netGF = HRA_Fusion( blocks=opt.blocks).to(opt.device)
        self.netGF = self.netGF.cuda()
        self.netGF = torch.nn.DataParallel(self.netGF)
        self.netGF.eval()

        self.optimizer_G = torch.optim.Adam(
            params=filter(lambda x: x.requires_grad, self.netGF.parameters()),
            lr=opt.lr,
            betas=(0.9, 0.999),
            eps=1e-08)

        self.optimizers.append(self.optimizer_G)
        self.avg_losses = OrderedDict()
        self.avg_loss_alpha = opt.avg_loss_alpha
        self.error_cnt = 0
        for loss_name in self.loss_names:
            self.avg_losses[loss_name] = 0

        self.transforms = tfs.Compose([
            tfs.Resize((opt.fineSize, opt.fineSize), interpolation=2),
            tfs.ToTensor(),
        ])

    def setup(self, opt, parser=None):
        if self.isTrain:
            self.schedulers = [self.get_scheduler(optimizer, opt) for optimizer in self.optimizers]


        # 1、加载实例模型的权重
        load_path = "加载实例模型的权重地址"
        state_dict = torch.load(load_path, map_location=str(opt.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        self.netG.load_state_dict(state_dict, strict=False)
        # 2、加载全图模型的权重
        load_path = "加载全图模型的权重地址"
        state_dict = torch.load(load_path, map_location=str(opt.device))
        self.netGF.load_state_dict(state_dict, strict=False)

    def set_input(self, input):
        self.haze = input['cropped_haze'].to(self.device)[0]
        self.clear = input['cropped_clear'].to(self.device)[0]

    def set_fusion_input(self, input, box_info):

        self.full_haze = input['full_haze'].to(self.device)[0]
        self.full_clear = input['full_clear'].to(self.device)[0]
        self.box_info_list = box_info
    def set_forward_without_box(self, input):
        self.full_haze = input['full_haze'].to(self.device)
        self.full_clear = input['full_clear'].to(self.device)
        self.comp_B_reg = self.netGComp(self.full_haze)
        self.fake_B_reg = self.comp_B_reg
    def forward(self):
        # self.comp_B_reg = self.netGComp(self.full_haze)
        (_, feature_map) = self.netG(self.haze)
        self.fake_B_reg = self.netGF(self.full_haze, feature_map,
                                     self.box_info_list)
    def get_scheduler(self, optimizer, opt):
        if opt.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler
    def get_current_visuals(self):
        from collections import OrderedDict
        visual_ret = OrderedDict()
        visual_ret['haze'] = self.haze
        visual_ret['clear'] = self.clear
        visual_ret['full_haze'] = self.full_haze
        visual_ret['full_clear'] = self.full_clear
        visual_ret['result'] = self.fake_B_reg
        self.instance_mask = torch.nn.functional.interpolate(torch.zeros([1, 1, 176, 176]),
                                                             size=visual_ret['haze'].shape[2:], mode='bilinear').type(
            torch.cuda.FloatTensor)
        visual_ret['box_mask'] = torch.cat((self.instance_mask, self.instance_mask, self.instance_mask), 1)
        return visual_ret
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_fusion_epoch(self, epoch):
        path = '{0}/{1}_net_GF.pth'.format(os.path.join(self.opt.checkpoints_dir, self.opt.name), epoch)
        latest_path = '{0}/latest_net_GF.pth'.format(os.path.join(self.opt.checkpoints_dir, self.opt.name))
        torch.save(self.netGF.state_dict(), path)
        torch.save(self.netGF.state_dict(), latest_path)

    def save_current_imgs(self, path):
        a = self.haze.type(torch.cuda.FloatTensor)
        b = self.fake_B_reg.type(torch.cuda.FloatTensor)
        out_img = torch.clamp(util.lab2rgb(
            torch.cat((a,
                       b),
                      dim=1), self.opt), 0.0, 1.0)

        out_img = np.transpose(out_img.cpu().data.numpy()[0], (1, 2, 0))
        io.imsave(path, img_as_ubyte(out_img))

    def setup_to_test(self, fusion_weight_path):
        G_path = 'checkpoints/{0}/latest_net_G.pth'.format(fusion_weight_path)
        G_state_dict = torch.load(G_path)
        self.netG.module.load_state_dict(G_state_dict, strict=False)
        self.netG.eval()

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)


