import torch.nn as nn
import torch
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)
    
class downS(nn.Module):
    def __init__(self,inchannel,kernel_size):
        super(downS,self).__init__()
        self.downSam = nn.Conv2d(inchannel,inchannel*2,kernel_size=kernel_size,stride = 2,padding = 1)
    def forward(self,x):
        x = self.downSam(x)
        return x
class upS(nn.Module):
    def __init__(self,inchannel,kernel_size):
        super(upS,self).__init__()
        self.upSam = nn.Conv2d(inchannel,inchannel * 2,kernel_size =kernel_size,padding=kernel_size//2,padding_mode='reflect' )
        self.pixelS = nn.PixelShuffle(2)
    def forward(self,x):
        x = self.upSam(x)
        x = self.pixelS(x)
        return x
class SALayer(nn.Module):
    def __init__(self, channel):
        super(SALayer, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.sa = nn.Sequential(
                nn.Conv2d(2, 1, 3, padding=1, bias=True)
        )
    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        y = self.sigmoid(self.sa(torch.cat([max_out, avg_out], dim=1)))
        return x * y
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.FCL = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.FCL(self.max_pool(x))
        avg_out = self.FCL(self.avg_pool(x))
        y = self.sigmoid(max_out + avg_out)
        return x * y

class HRA_block(nn.Module):
    def __init__(self,dim):
        super(HRA_block, self).__init__()
        self.CAU=CALayer(dim)
        self.SAU=SALayer(dim)
    def forward(self, x):
        res=self.CAU(x)
        res=self.SAU(res)
        return res
class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size,):
        super(Block, self).__init__()
        self.conv1=conv(dim, dim, kernel_size, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=conv(dim,dim,kernel_size,bias=True)
        self.HRA_block=HRA_block(dim)

    def forward(self, x):
        res=self.act1(self.conv1(x))
        res=res+x 
        res=self.conv2(res)
        res=self.HRA_block(res)
        res += x 
        return res

class DCF(nn.Module):
     def __init__(self,dim,  in_channel=512, depth=256):
         super(DCF, self).__init__()
         self.dim=dim
         self.mean = nn.AdaptiveAvgPool2d((1, 1))
         self.conv = nn.Conv2d(in_channel, depth, 1, 1)
         self.dilated_block1 = nn.Conv2d(in_channel, depth, 1, 1)
         self.dilated_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
         self.dilated_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
         self.dilated_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
         self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
         self.salayer = SALayer(self.dim)
     def forward(self, x):
          size = x.shape[2:]
          image_features = self.mean(x)
          image_features = self.conv(image_features)
          image_features = F.upsample(image_features, size=size, mode='bilinear')
          dilated_block1 = self.dilated_block1(x)
          dilated_block6 = self.dilated_block6(x)
          dilated_block12 = self.dilated_block12(x)
          dilated_block18 = self.dilated_block18(x)
          net = self.conv_1x1_output(torch.cat([image_features, dilated_block1, dilated_block6,
                                                          dilated_block12, dilated_block18], dim=1))
          net=self.salayer(net)
          return net
class HRA_Module(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(HRA_Module, self).__init__()
        modules = [ Block(conv, dim, kernel_size)  for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)
    def forward(self, x):
        res = self.gp(x)
        res += x
        return res

class HRA(nn.Module):
    def __init__(self,blocks,conv=default_conv):
        super(HRA, self).__init__()
        self.dim=64
        kernel_size=3
        pre_process = [conv(3, self.dim, kernel_size)]
        self.downS1 = downS(self.dim,3)
        self.downS2 = downS(self.dim*2,3)
        self.upS1 = upS(self.dim * 4, 1)
        self.upS2 = upS(self.dim * 2, 1)
        self.h1= HRA_Module(conv, self.dim, kernel_size,blocks=blocks)
        self.h2 = HRA_Module(conv, self.dim*2, kernel_size, blocks=blocks)
        self.h3 = HRA_Module(conv, self.dim*4, kernel_size, blocks=blocks)
        self.h4 = HRA_Module(conv, self.dim*2, kernel_size, blocks=blocks)
        self.h5 = HRA_Module(conv, self.dim, kernel_size, blocks=blocks)
        self.DCF = DCF(dim=self.dim,in_channel=self.dim, depth=self.dim)
        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]
        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)
	    
    def forward(self, x1):
        x = self.pre(x1)
        res1=self.h1(x)# h w c
        Res1 = res1
        res1=self.downS1(res1)# h/2 w/2 2c
        res2=self.h2(res1)# h/2 w/2 2c
        res2 = self.downS2(res2)# h/4 w/4 4c
        res3=self.h3(res2)# h/4 w/4 4c
        res3 = self.upS1(res3)# h/2 w/2 2c
        res4=self.h4(res3)# h/2 w/2 2c
        res4 = self.upS2(res4)# h w c
        res5 = self.h5(res4)# h w c
        res6 = res5+Res1
        x = self.DCF(res6)
        x = self.post(x)
        return x + x1
	    
class HRA_INST(nn.Module):
    def __init__(self, blocks, conv=default_conv):
        super(HRA_INST, self).__init__()
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        self.downS1 = downS(self.dim, 3)
        self.downS2 = downS(self.dim * 2, 3)
        self.upS1 = upS(self.dim * 4, 1)
        self.upS2 = upS(self.dim * 2, 1)
        self.h1 = HRA_Module(conv, self.dim, kernel_size, blocks=blocks)
        self.h2 = HRA_Module(conv, self.dim * 2, kernel_size, blocks=blocks)
        self.h3 = HRA_Module(conv, self.dim * 4, kernel_size, blocks=blocks)
        self.h4 = HRA_Module(conv, self.dim * 2, kernel_size, blocks=blocks)
        self.h5 = HRA_Module(conv, self.dim, kernel_size, blocks=blocks)
        self.DCF = DCF(dim=self.dim,in_channel=self.dim, depth=self.dim)
        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        feature_map = {}
        x = self.pre(x1)
        feature_map['x'] = x
        res1 = self.h11(x)  # h w c
        Res1 = res1
        feature_map['res1'] = res1
        res1 = self.downS1(res1)  # h/2 w/2 2c
        res2 = self.h2(res1)  # h/2 w/2 2c
        Res2 = res2
        res2 = self.downS2(res2)  # h/4 w/4 4c
        res3 = self.h3(res2)  # h/4 w/4 4c
        # print("1",res3.size())
        res3 = self.upS1(res3)  # h/2 w/2 2c
        # print("2",res3.size())
        feature_map['res2'] = res3
        res4 = self.h4(res3)  # h/2 w/2 2c
        feature_map['res3'] = res4
        res4 = res4 + Res2
        res4 = self.upS2(res4)  # h w c
        res5 = self.h5(res4)  # h w c
        res6 = res5 + Res1
        out = self.DCF(res6)
        x = self.post(out)
        feature_map['x2'] = x
        return x + x1,feature_map
	    
class HRA_Fusion(nn.Module):
    def __init__(self, blocks, conv=default_conv):
        super(HRA_Fusion, self).__init__()
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        self.downS1 = downS(self.dim, 3)
        self.downS2 = downS(self.dim * 2, 3)
        self.upS1 = upS(self.dim * 4, 1)
        self.weight_layer3 = GFF_subnet(2 * self.dim)
        self.upS2 = upS(self.dim * 2, 1)
        self.h1 = HRA_Module(conv, self.dim, kernel_size, blocks=blocks)
        self.weight_layer2 = GFF_subnet(self.dim)
        self.h2 = HRA_Module(conv, self.dim * 2, kernel_size, blocks=blocks)
        self.h3 = HRA_Module(conv, self.dim * 4, kernel_size, blocks=blocks)
        self.h4 = HRA_Module(conv, self.dim * 2, kernel_size, blocks=blocks)
        self.weight_layer4 = GFF_subnet(2 * self.dim)
        self.h5 = HRA_Module(conv, self.dim, kernel_size, blocks=blocks)
        self.DCF= DCF(dim=self.dim,in_channel=self.dim, depth=self.dim)
        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]
        self.pre = nn.Sequential(*pre_process)
        self.weight_layer1 = GFF_subnet(self.dim)
        self.post = nn.Sequential(*post_precess)
        self.weight_layer5 = GFF_subnet(3)

    def forward(self, x1, instance_feature, box_info_list):
         x = self.pre(x1)
         x = self.weight_layer1(instance_feature['x'], x, box_info_list[0])
         res1 = self.h1(x)  # h w c
         res1 = self.weight_layer2(instance_feature['res1'], res1, box_info_list[0])
         Res1 = res1
         res1 = self.downS1(res1)  # h/2 w/2 2c
         res2 = self.h2(res1)  # h/2 w/2 2c
         Res2 = res2
         res2 = self.downS2(res2)  # h/4 w/4 4c
         res3 = self.h3(res2)  # h/4 w/4 4c
         res3 = self.upS1(res3)  # h/2 w/2 2c
         res3 = self.weight_layer3(instance_feature['res2'], res3, box_info_list[1])
         res4 = self.h4(res3)  # h/2 w/2 2c
         res4 = self.weight_layer4(instance_feature['res3'], res4, box_info_list[1])
         res4 = res4 + Res2
         res4 = self.upS2(res4)  # h w c
         res5 = self.h5(res4)  # h w c
         res6 = res5 + Res1
         out = self.DCF(res6)
         x = self.post(out)
         x = self.weight_layer5(instance_feature['x2'], x, box_info_list[0])

         return x + x1

class GFF_subnet(nn.Module):
    def __init__(self, input_ch, inner_ch=16):
        super(GFF_subnet, self).__init__()
        self.simple_instance_conv = nn.Sequential(
            nn.Conv2d(input_ch, inner_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(inner_ch, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.simple_bg_conv = nn.Sequential(
            nn.Conv2d(input_ch, inner_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(inner_ch, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
	self.post =  nn.Sequential(
            nn.Conv2d(inner_ch, inner_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.normalize = nn.Softmax(1)

    def resize_and_pad(self, feature_maps, info_array):
        feature_maps = torch.nn.functional.interpolate(feature_maps, size=(info_array[5], info_array[4]),
                                                           mode='bilinear')
        feature_maps = torch.nn.functional.pad(feature_maps,
                                               (info_array[0], info_array[1], info_array[2], info_array[3]),
                                               "constant", 0)
        return feature_maps

    def forward(self, instance_feature, bg_feature, box_info):
        mask_list = []
        featur_map_list = []
        mask_sum_for_pred = torch.zeros_like(bg_feature)[:1, :1]
        for i in range(instance_feature.shape[0]):
            tmp_crop = torch.unsqueeze(instance_feature[i], 0)
            conv_tmp_crop = self.simple_instance_conv(tmp_crop)
            pred_mask = self.resize_and_pad(conv_tmp_crop, box_info[i])
            tmp_crop = self.resize_and_pad(tmp_crop, box_info[i])
            mask = torch.zeros_like(bg_feature)[:1, :1]
            mask[0, 0, box_info[i][2]:box_info[i][2] + box_info[i][5],
            box_info[i][0]:box_info[i][0] + box_info[i][4]] = 1.0
            device = mask.device
            mask = mask.type(torch.FloatTensor).to(device)
            mask_sum_for_pred = torch.clamp(mask_sum_for_pred + mask, 0.0, 1.0)
            mask_list.append(pred_mask)
            featur_map_list.append(tmp_crop)
        pred_bg_mask = self.simple_bg_conv(bg_feature)
        if instance_feature.shape[-1] == 1:
            mask_list.append(mask_list[0] + (1 - mask_sum_for_pred) * 100000.0)
        else:
            mask_list.append(pred_bg_mask + (1 - mask_sum_for_pred) * 100000.0)
	mask_list = torch.cat(mask_list, dim = 1)
        mask_list_maskout = mask_list.clone()
        if instance_feature.shape[-1] == 1:
            featur_map_list = []
            featur_map_list.append(bg_feature)
            featur_map_list.append(bg_feature)
        else:
            featur_map_list.append(bg_feature)

        featur_map_list = torch.cat(featur_map_list, 0)
 
        mask_list_maskout = mask_list_maskout.permute(1, 0, 2, 3).contiguous()

        out = featur_map_list * mask_list_maskout
        out = self.normalize(out)
        out = torch.sum(out, 0, keepdim=True)
	out = self.post(out)
        return out  # , instance_mask, torch.clamp(mask_list, 0.0, 1.0)


if __name__ == "__main__":
    net=HRA(blocks=13)
    print(net)
