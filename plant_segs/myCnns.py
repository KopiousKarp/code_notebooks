# List of Fully Convolutional Neural Networks
import torch
import torch.nn as nn
import torch.nn.functional as F

# U-net
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return F.softmax(x3, dim=1)


# U-net++
class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetPlusPlus, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.middle = conv_block(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        m = self.middle(self.pool(e4))
        
        d4 = self.upconv4(m)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)
        
        out = self.final_conv(d1)
        return F.softmax(out, dim=1)

# SegNet
class SegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(SegNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        
        self.decoder4 = conv_block(512, 256)
        self.decoder3 = conv_block(256, 128)
        self.decoder2 = conv_block(128, 64)
        self.decoder1 = conv_block(64, out_channels)
        
    def forward(self, x):
        x1, idx1 = self.pool(self.encoder1(x))
        x2, idx2 = self.pool(self.encoder2(x1))
        x3, idx3 = self.pool(self.encoder3(x2))
        x4, idx4 = self.pool(self.encoder4(x3))
        
        x4 = self.unpool(x4, idx4)
        x4 = self.decoder4(x4)
        
        x3 = self.unpool(x4, idx3)
        x3 = self.decoder3(x3)
        
        x2 = self.unpool(x3, idx2)
        x2 = self.decoder2(x2)
        
        x1 = self.unpool(x2, idx1)
        x1 = self.decoder1(x1)
        
        return F.softmax(x1, dim=1)

# Residual U-net
class ResidualUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(ResidualUNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels)
            )
        
        def residual_block(in_channels, out_channels):
            return nn.Sequential(
                conv_block(in_channels, out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.encoder1 = residual_block(in_channels, 64)
        self.encoder2 = residual_block(64, 128)
        self.encoder3 = residual_block(128, 256)
        self.encoder4 = residual_block(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.middle = residual_block(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = residual_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = residual_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = residual_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = residual_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        m = self.middle(self.pool(e4))
        
        d4 = self.upconv4(m)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)
        
        out = self.final_conv(d1)
        return F.softmax(out, dim=1)

# U2-net
class U2Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(U2Net, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.middle = conv_block(512, 1024)
        
        self.upconv4 = up_block(1024, 512)
        self.decoder4 = conv_block(1024, 512)
        
        self.upconv3 = up_block(512, 256)
        self.decoder3 = conv_block(512, 256)
        
        self.upconv2 = up_block(256, 128)
        self.decoder2 = conv_block(256, 128)
        
        self.upconv1 = up_block(128, 64)
        self.decoder1 = conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        m = self.middle(self.pool(e4))
        
        d4 = self.upconv4(m)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)
        
        out = self.final_conv(d1)
        return F.softmax(out, dim=1)

# Swin U-net
class SwinUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(SwinUNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.middle = conv_block(512, 1024)
        
        self.upconv4 = up_block(1024, 512)
        self.decoder4 = conv_block(1024, 512)
        
        self.upconv3 = up_block(512, 256)
        self.decoder3 = conv_block(512, 256)
        
        self.upconv2 = up_block(256, 128)
        self.decoder2 = conv_block(256, 128)
        
        self.upconv1 = up_block(128, 64)
        self.decoder1 = conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        m = self.middle(self.pool(e4))
        
        d4 = self.upconv4(m)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)
        
        out = self.final_conv(d1)
        return F.softmax(out, dim=1)

# Attention U-net
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(AttentionUNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def attention_block(g, x, inter_channels):
            theta_x = nn.Conv2d(x.size(1), inter_channels, kernel_size=2, stride=2, padding=0)(x)
            phi_g = nn.Conv2d(g.size(1), inter_channels, kernel_size=1, stride=1, padding=0)(g)
            f = F.relu(theta_x + phi_g, inplace=True)
            psi_f = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)(f)
            rate = torch.sigmoid(psi_f)
            return x * rate
        
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.middle = conv_block(512, 1024)
        
        self.upconv4 = up_block(1024, 512)
        self.att4 = attention_block
        self.decoder4 = conv_block(1024, 512)
        
        self.upconv3 = up_block(512, 256)
        self.att3 = attention_block
        self.decoder3 = conv_block(512, 256)
        
        self.upconv2 = up_block(256, 128)
        self.att2 = attention_block
        self.decoder2 = conv_block(256, 128)
        
        self.upconv1 = up_block(128, 64)
        self.att1 = attention_block
        self.decoder1 = conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        m = self.middle(self.pool(e4))
        
        d4 = self.upconv4(m)
        d4 = torch.cat((d4, self.att4(d4, e4, 512)), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, self.att3(d3, e3, 256)), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, self.att2(d2, e2, 128)), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, self.att1(d1, e1, 64)), dim=1)
        d1 = self.decoder1(d1)
        
        out = self.final_conv(d1)
        return F.softmax(out, dim=1)