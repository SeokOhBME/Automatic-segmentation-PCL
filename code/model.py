import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    #if gpu_ids:
    #    assert(torch.cuda.is_available())
    #    net.to(gpu_ids)
    #    net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class R2U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, t=2):
        super(R2U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=32, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=32, ch_out=64, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.Up5 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=64, ch_out=32, t=t)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        sig = nn.Sigmoid()
        d1 = sig(d1)

        return d1



class R2AttU_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, t=2):
        super(R2AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=32, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=32, ch_out=64, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.Up5 = up_conv(ch_in=512, ch_out=256)
        self.Att5 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_RRCNN5 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Att4 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN4 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Att3 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN3 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Att2 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Up_RRCNN2 = RRCNN_block(ch_in=64, ch_out=32, t=t)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        sig = nn.Sigmoid()
        d1 = sig(d1)

        return d1



## 네트워크 구축하기
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr
        nb_filter = [32, 64, 128, 256, 512]

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=1, out_channels=nb_filter[0])
        self.enc1_2 = CBR2d(in_channels=nb_filter[0], out_channels=nb_filter[0])

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=nb_filter[0], out_channels=nb_filter[1])
        self.enc2_2 = CBR2d(in_channels=nb_filter[1], out_channels=nb_filter[1])

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=nb_filter[1], out_channels=nb_filter[2])
        self.enc3_2 = CBR2d(in_channels=nb_filter[2], out_channels=nb_filter[2])

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=nb_filter[2], out_channels=nb_filter[3])
        self.enc4_2 = CBR2d(in_channels=nb_filter[3], out_channels=nb_filter[3])

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=nb_filter[3], out_channels=nb_filter[4])

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=nb_filter[4], out_channels=nb_filter[3])

        self.unpool4 = nn.ConvTranspose2d(in_channels=nb_filter[3], out_channels=nb_filter[3],
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * nb_filter[3], out_channels=nb_filter[3])
        self.dec4_1 = CBR2d(in_channels=nb_filter[3], out_channels=nb_filter[2])

        self.unpool3 = nn.ConvTranspose2d(in_channels=nb_filter[2], out_channels=nb_filter[2],
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * nb_filter[2], out_channels=nb_filter[2])
        self.dec3_1 = CBR2d(in_channels=nb_filter[2], out_channels=nb_filter[1])

        self.unpool2 = nn.ConvTranspose2d(in_channels=nb_filter[1], out_channels=nb_filter[1],
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * nb_filter[1], out_channels=nb_filter[1])
        self.dec2_1 = CBR2d(in_channels=nb_filter[1], out_channels=nb_filter[0])

        self.unpool1 = nn.ConvTranspose2d(in_channels=nb_filter[0], out_channels=nb_filter[0],
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * nb_filter[0], out_channels=nb_filter[0])
        self.dec1_1 = CBR2d(in_channels=nb_filter[0], out_channels=nb_filter[0])

        self.fc = nn.Conv2d(in_channels=nb_filter[0], out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)
        sig = nn.Sigmoid()
        x = sig(x)


        return x
import torch
import torch.nn as nn

class NestedUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, deep_supervision=False, **kwargs):
        super().__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr


        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Contracting path
        self.conv0_0_1 = CBR2d(in_channels=1, out_channels=nb_filter[0])
        self.conv0_0_2 = CBR2d(in_channels=nb_filter[0], out_channels=nb_filter[0])
        self.conv1_0_1 = CBR2d(in_channels=nb_filter[0], out_channels=nb_filter[1])
        self.conv1_0_2 = CBR2d(in_channels=nb_filter[1], out_channels=nb_filter[1])
        self.conv2_0_1 = CBR2d(in_channels=nb_filter[1], out_channels=nb_filter[2])
        self.conv2_0_2 = CBR2d(in_channels=nb_filter[2], out_channels=nb_filter[2])
        self.conv3_0_1 = CBR2d(in_channels=nb_filter[2], out_channels=nb_filter[3])
        self.conv3_0_2 = CBR2d(in_channels=nb_filter[3], out_channels=nb_filter[3])
        self.conv4_0_1 = CBR2d(in_channels=nb_filter[3], out_channels=nb_filter[4])
        self.conv4_0_2 = CBR2d(in_channels=nb_filter[4], out_channels=nb_filter[4])

        self.conv0_1_1 = CBR2d(in_channels=nb_filter[0]+nb_filter[1], out_channels=nb_filter[0])
        self.conv0_1_2 = CBR2d(in_channels=nb_filter[0], out_channels=nb_filter[0])
        self.conv1_1_1 = CBR2d(in_channels=nb_filter[1]+nb_filter[2], out_channels=nb_filter[1])
        self.conv1_1_2 = CBR2d(in_channels=nb_filter[1], out_channels=nb_filter[1])
        self.conv2_1_1 = CBR2d(in_channels=nb_filter[2]+nb_filter[3], out_channels=nb_filter[2])
        self.conv2_1_2 = CBR2d(in_channels=nb_filter[2], out_channels=nb_filter[2])
        self.conv3_1_1 = CBR2d(in_channels=nb_filter[3]+nb_filter[4], out_channels=nb_filter[3])
        self.conv3_1_2 = CBR2d(in_channels=nb_filter[3], out_channels=nb_filter[3])

        self.conv0_2_1 = CBR2d(in_channels=nb_filter[0]*2 + nb_filter[1], out_channels=nb_filter[0])
        self.conv0_2_2 = CBR2d(in_channels=nb_filter[0], out_channels=nb_filter[0])
        self.conv1_2_1 = CBR2d(in_channels=nb_filter[1]*2 + nb_filter[2], out_channels=nb_filter[1])
        self.conv1_2_2 = CBR2d(in_channels=nb_filter[1], out_channels=nb_filter[1])
        self.conv2_2_1 = CBR2d(in_channels=nb_filter[2]*2 + nb_filter[3], out_channels=nb_filter[2])
        self.conv2_2_2 = CBR2d(in_channels=nb_filter[2], out_channels=nb_filter[2])

        self.conv0_3_1 = CBR2d(in_channels=nb_filter[0]*3 + nb_filter[1], out_channels=nb_filter[0])
        self.conv0_3_2 = CBR2d(in_channels=nb_filter[0], out_channels=nb_filter[0])
        self.conv1_3_1 = CBR2d(in_channels=nb_filter[1]*3 + nb_filter[2], out_channels=nb_filter[1])
        self.conv1_3_2 = CBR2d(in_channels=nb_filter[1], out_channels=nb_filter[1])

        self.conv0_4_1 = CBR2d(in_channels=nb_filter[0]*4 + nb_filter[1], out_channels=nb_filter[0])
        self.conv0_4_2 = CBR2d(in_channels=nb_filter[0], out_channels=nb_filter[0])
        self.conv1_4_1 = CBR2d(in_channels=nb_filter[1]*4 + nb_filter[2], out_channels=nb_filter[1])
        self.conv1_4_2 = CBR2d(in_channels=nb_filter[1], out_channels=nb_filter[1])


        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0_1(input)
        x0_0 = self.conv0_0_2(x0_0)

        x1_0 = self.conv1_0_1(self.pool(x0_0))
        x1_0 = self.conv1_0_2(x1_0)
        x0_1 = self.conv0_1_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x0_1 = self.conv0_1_2(x0_1)

        x2_0 = self.conv2_0_1(self.pool(x1_0))
        x2_0 = self.conv2_0_2(x2_0)
        x1_1 = self.conv1_1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x1_1 = self.conv1_1_2(x1_1)
        x0_2 = self.conv0_2_1(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x0_2 = self.conv0_2_2(x0_2)

        x3_0 = self.conv3_0_1(self.pool(x2_0))
        x3_0 = self.conv3_0_2(x3_0)
        x2_1 = self.conv2_1_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x2_1 = self.conv2_1_2(x2_1)
        x1_2 = self.conv1_2_1(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x1_2 = self.conv1_2_2(x1_2)
        x0_3 = self.conv0_3_1(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x0_3 = self.conv0_3_2(x0_3)

        x4_0 = self.conv4_0_1(self.pool(x3_0))
        x4_0 = self.conv4_0_2(x4_0)
        x3_1 = self.conv3_1_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x3_1 = self.conv3_1_2(x3_1)
        x2_2 = self.conv2_2_1(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x2_2 = self.conv2_2_2(x2_2)

        x1_3 = self.conv1_3_1(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x1_3 = self.conv1_3_2(x1_3)

        x0_4 = self.conv0_4_1(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        x0_4 = self.conv0_4_2(x0_4)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            sig = nn.Sigmoid()
            output1 = sig(output1)
            output2 = sig(output2)
            output3 = sig(output3)
            output4 = sig(output4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            sig = nn.Sigmoid()
            output = sig(output)
            return output



class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
class AttU_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        nb_filter = [32, 64, 128, 256, 512]

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=nb_filter[0])
        self.Conv2 = conv_block(ch_in=nb_filter[0], ch_out=nb_filter[1])
        self.Conv3 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[2])
        self.Conv4 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[3])
        self.Conv5 = conv_block(ch_in=nb_filter[3], ch_out=nb_filter[4])

        self.Up5 = up_conv(ch_in=nb_filter[4], ch_out=nb_filter[3])
        self.Att5 = Attention_block(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[2])
        self.Up_conv5 = conv_block(ch_in=nb_filter[4], ch_out=nb_filter[3])

        self.Up4 = up_conv(ch_in=nb_filter[3], ch_out=nb_filter[2])
        self.Att4 = Attention_block(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])
        self.Up_conv4 = conv_block(ch_in=nb_filter[3], ch_out=nb_filter[2])

        self.Up3 = up_conv(ch_in=nb_filter[2], ch_out=nb_filter[1])
        self.Att3 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Up_conv3 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[1])

        self.Up2 = up_conv(ch_in=nb_filter[1], ch_out=nb_filter[0])
        self.Att2 = Attention_block(F_g=nb_filter[0], F_l=nb_filter[0], F_int=16)
        self.Up_conv2 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[0])

        self.Conv_1x1 = nn.Conv2d(nb_filter[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        sig = nn.Sigmoid()
        d1 = sig(d1)

        return d1


import torch
import torch.nn as nn

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class ResUnet(nn.Module):
    def __init__(self, channel = 1, filters=[32, 64, 128, 256,512]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        ##
        #
#
        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.bridge = ResidualConv(filters[3], filters[4], 2, 1)

        ##
        #
        # self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
#       self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        ##

        self.upsample_1 = Upsample(filters[4], filters[4], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[3], filters[3], 1, 1)

        self.upsample_2 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_3 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_4 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv4 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)



        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)

        x4 = self.residual_conv_3(x3)  #########################################3
        # Bridge
        x5 = self.bridge(x4) ####################################################
        # Decode
        x5 = self.upsample_1(x5)
        x6 = torch.cat([x5, x4], dim=1)

        x7 = self.up_residual_conv1(x6)

        x7 = self.upsample_2(x7)
        x8 = torch.cat([x7, x3], dim=1)

        x9 = self.up_residual_conv2(x8)

        x9 = self.upsample_3(x9)
        x10 = torch.cat([x9, x2], dim=1)
        x11 = self.up_residual_conv3(x10)

        x11 = self.upsample_4(x11)
        x12 = torch.cat([x11, x1], dim=1)
        x13 = self.up_residual_conv4(x12)



        output = self.output_layer(x13)

        return output



import torch.nn.functional as F
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_norm, is_groupnorm=True):
        super(unetConv2, self).__init__()

        if is_norm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1),
                # nn.BatchNorm2d(out_size),
                nn.GroupNorm(min(8, out_size), out_size) if is_groupnorm else nn.BatchNorm2d(out_size),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1),
                # nn.BatchNorm2d(out_size),
                nn.GroupNorm(min(8, out_size), out_size) if is_groupnorm else nn.BatchNorm2d(out_size),
                nn.ReLU(),
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU())

    def forward(self, inputs):
        # print('inputs device', inputs.device)
        # print('unetConv2 device as follows: ... ')
        # for name, param in self.conv1.state_dict().items():
        #     print(name, param.device)
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=False):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2, output_size=inputs1.size())
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        # print(outputs1.device)
        # print(outputs2.device)
        # print(self.conv)
        outputs = self.conv(torch.cat([outputs1, outputs2], 1))
        return outputs


class sru(nn.Module):
    def __init__(self,

                 n_classes=1,
                 initial=1,
                 steps=4,
                 gate=3,
                 hidden_size=128,
                 feature_scale=1,
                 is_deconv=True,
                 in_channels=1,
                 is_batchnorm=True,
                 ):
        super(sru, self).__init__()

        self.steps = steps
        self.feature_scale = feature_scale
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.is_deconv = is_deconv

        filters = [32, 64, 128, 256, 512,1024]  # [8, 16, 32, 64, 128] [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels + self.n_classes, filters[0], self.is_batchnorm)
        self.conv1_start = unetConv2(self.in_channels , filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(self.hidden_size, filters[5], self.is_batchnorm)

        # upsampling
        self.up_concat5 = unetUp(filters[5], filters[4], self.is_deconv)
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # convgru
        #assert(self.hidden_size == filters[4])
        self.gru = ConvSRU(filters[4], self.hidden_size)

        # final conv (without any concat)
        self.conv_down = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs, h = 128):
        list_st = []
        for i in range(self.steps):
            if i == 0:
                stacked_inputs = inputs
                conv1 = self.conv1_start(stacked_inputs)


            else:
                stacked_inputs = torch.cat([inputs, s], dim=1)
                conv1 = self.conv1(stacked_inputs)




            maxpool1 = self.maxpool1(conv1)

            conv2 = self.conv2(maxpool1)
            maxpool2 = self.maxpool2(conv2)

            conv3 = self.conv3(maxpool2)
            maxpool3 = self.maxpool3(conv3)

            conv4 = self.conv4(maxpool3)
            maxpool4 = self.maxpool4(conv4)

            conv5 = self.conv5(maxpool4)
            maxpool5 = self.maxpool5(conv5)

            h = self.gru(maxpool5, h)
            dt = self.center(h)

            up5 = self.up_concat5(conv5, dt)
            up4 = self.up_concat4(conv4, up5)
            up3 = self.up_concat3(conv3, up4)
            up2 = self.up_concat2(conv2, up3)
            up1 = self.up_concat1(conv1, up2)

            s = self.conv_down(up1)
            sig = nn.Sigmoid()
            s = sig(s)
            list_st += [s]


        return list_st


class ConvSRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 ):
        super(ConvSRU, self).__init__()

        self.update_gate = unetConv2(input_size, hidden_size, True)
        self.out_gate = unetConv2(input_size, hidden_size, True)

    def forward(self, input_, h=None):
        # batch_size = input_.data.size()[0]
        # spatial_size = input_.data.size()[2:]

        # data size is [batch, channel, height, width]
        # print('input_.type', input_.data.type())
        # print('prev_state.type', prev_state.data.type())

        update = torch.sigmoid(self.update_gate(input_))
        # print('input_, reset, h, shape ', input_.shape, reset.shape, h.shape)
        # stacked_inputs_ = torch.cat([input_, h * reset], dim=1)

        out_inputs = torch.tanh(self.out_gate(input_))
        h_new = h * (1 - update) + out_inputs * update
        return h_new

class InceptionModule(nn.Module):
    def __init__(self,in_channels,num_filter):
        super().__init__()

        self.incep1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=num_filter, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(num_features=num_filter),
        nn.ReLU())


        self.incep2 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=num_filter, kernel_size=1, stride=1, padding=0),
         nn.BatchNorm2d(num_features=num_filter),
         nn.ReLU(),
         nn.Conv2d(in_channels=num_filter, out_channels=num_filter, kernel_size=3, stride = 1, padding=1),
         nn.BatchNorm2d(num_features=num_filter),
         nn.ReLU())

        self.incep3 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=num_filter, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(num_features=num_filter),
        nn.ReLU(),
        nn.Conv2d(in_channels=num_filter, out_channels=num_filter, kernel_size=3, stride = 1, padding=1),
        nn.Conv2d(in_channels=num_filter, out_channels=num_filter, kernel_size=3, stride = 1, padding=1),
        nn.BatchNorm2d(num_features=num_filter),
        nn.ReLU())

        self.incep4 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=num_filter, kernel_size=1, stride = 1, padding=0 ),
        nn.BatchNorm2d(num_features=num_filter),
        nn.ReLU(),
        nn.Conv2d(in_channels=num_filter, out_channels=num_filter, kernel_size=3, stride = 1, padding=1),
        nn.Conv2d(in_channels=num_filter, out_channels=num_filter, kernel_size=3, stride = 1, padding=1),
        nn.BatchNorm2d(num_features=num_filter),
        nn.ReLU())

    def forward(self, x):

        inception1 = self.incep1(x)
        inception2 = self.incep2(x)
        inception3 = self.incep3(x)
        inception4 = self.incep4(x)
        output = torch.cat([inception1,inception2,inception3,inception4], 1)
        #print(output.size())

        return output




class InceptionUnet(nn.Module):
    def __init__(self,num_classes = 1, input_channels = 1):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]

        def CBR_final(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=num_classes ,
                                 kernel_size=1, stride=1, padding=0,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr
        self.conv1 = InceptionModule(input_channels, nb_filter[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = InceptionModule(4*nb_filter[0],nb_filter[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = InceptionModule(4*nb_filter[1],nb_filter[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = InceptionModule(4*nb_filter[2],nb_filter[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = InceptionModule(4*nb_filter[3],nb_filter[4])


        self.unpool4 = nn.ConvTranspose2d(in_channels=4*nb_filter[4], out_channels=4*nb_filter[4],
                                          kernel_size=2, stride=2, padding=0, bias=True)
        self.dec_conv4 = InceptionModule(4*nb_filter[4],nb_filter[3])

        self.unpool3 = nn.ConvTranspose2d(in_channels=2*4*nb_filter[3], out_channels=2*4*nb_filter[3],
                                          kernel_size=2, stride=2, padding=0, bias=True)
        self.dec_conv3 = InceptionModule(2*4*nb_filter[3],nb_filter[2])

        self.unpool2 = nn.ConvTranspose2d(in_channels=2*4*nb_filter[2], out_channels=2*4*nb_filter[2],
                                          kernel_size=2, stride=2, padding=0, bias=True)
        self.dec_conv2 = InceptionModule(2*4*nb_filter[2],nb_filter[1])

        self.unpool1 = nn.ConvTranspose2d(in_channels=2*4*nb_filter[1], out_channels=2*4*nb_filter[1],
                                          kernel_size=2, stride=2, padding=0, bias=True)
        self.dec_conv1 = InceptionModule(2*4*nb_filter[1],nb_filter[0])

        self.final =CBR_final(2*4*nb_filter[0],1)
    def forward(self, x):

        enc_1_1 = self.conv1(x)
        enc_1_2 = self.pool1(enc_1_1)

        enc_2_1 = self.conv2(enc_1_2 )
        enc_2_2 = self.pool2(enc_2_1)

        enc_3_1 = self.conv3(enc_2_2 )
        enc_3_2 = self.pool3(enc_3_1)

        enc_4_1 = self.conv4(enc_3_2)
        enc_4_2 = self.pool4(enc_4_1)

        enc_5_2 = self.conv5(enc_4_2)

        unpool4 = self.unpool4(enc_5_2 )
        #print(unpool4.size())
        dec_4_1 = self.dec_conv4(unpool4 )
        #print(dec_4_1.size())
        dec_4_2 = torch.cat([dec_4_1,enc_4_1  ], 1)

        unpool3 = self.unpool3(dec_4_2 )
        dec_3_1 = self.dec_conv3(unpool3 )
        dec_3_2 = torch.cat([dec_3_1,enc_3_1  ], 1)

        unpool2 = self.unpool2(dec_3_2 )
        dec_2_1 = self.dec_conv2(unpool2 )
        dec_2_2 = torch.cat([dec_2_1,enc_2_1  ], 1)

        unpool1 = self.unpool1(dec_2_2 )
        dec_1_1 = self.dec_conv1(unpool1 )
        dec_1_2 = torch.cat([dec_1_1,enc_1_1  ], 1)
        y = self.final(dec_1_2)

        sig = nn.Sigmoid()
        y = sig(y)

        return y
