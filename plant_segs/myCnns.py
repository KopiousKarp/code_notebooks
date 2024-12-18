# List of Fully Convolutional Neural Networks
import torch
import torch.nn as nn
import torch.nn.functional as F

# U-net
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super(UNet, self).__init__()
        self.k = 7
        self.p = 3
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=self.k, padding=self.p),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=self.k, padding=self.p),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder2 = nn.Sequential(  
            nn.Conv2d(64, 128, kernel_size=self.k, padding=self.p),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=self.k, padding=self.p),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=self.k, padding=self.p),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=self.k, padding=self.p),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(  
            nn.Conv2d(256 + 128, 128, kernel_size=self.k, padding=self.p), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=self.k, padding=self.p),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=self.k, padding=self.p),  # ADD: Skip connection from encoder1
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=self.k, padding=self.p),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
      x1 = self.encoder1(x)  # Encoder Level 1
      x2 = self.encoder2(x1)  # Encoder Level 2
      x3 = self.bottleneck(x2)  # ADD: Bottleneck
      # Decoder Level 2 with skip connection
      x4 = self.decoder2(torch.cat([x3, x2], dim=0))  # ADD: Concatenation for skip connection
      # Decoder Level 1 with skip connection
      x5 = self.decoder1(torch.cat([x4, x1], dim=0))  # ADD: Concatenation for skip connection
    #   print(x5.shape)  
    #   return F.softmax(x5, dim=0)  # Output
      return x5
    
    # def predict_mask(self,img,merge=True):
    #     '''
    #     img : numpy array 
    #     merge : boolean
    #         True : output is the original image with mask overlayed
    #         False: output is soley the mask
    #     '''
    #     #convert img to tensor
    #     if isinstance(img, torch.Tensor):
    #         img_tensor = img
    #     else:
    #         img_tensor = torch.from_numpy(img).float()
        
    #     # run forward on img_tensor
    #     mask_logits =
    

        # torch.set_printoptions(threshold=float('inf'))
        # labels_t = mask_logits.view(1, 1, *mask_logits.shape).float()  # Reshape to (N, C, H, W)
        # resized_labels = F.interpolate(labels_t, size=(16, 16), mode='nearest').squeeze(0).squeeze(0)  # Interpolate and then remove the added dimensions
        # print(resized_labels)          
         


# SegNet
class SegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super(SegNet, self).__init__()
        # Encoder Block 1
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # ADD: Return indices for unpooling
        # Encoder Block 2
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # ADD: Return indices for unpooling
        # Decoder Block 2
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)  # ADD: Unpool layer
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Decoder Block 1
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)  # ADD: Unpool layer
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Encoder Block 1
        x1 = self.encoder1(x)
        x1_pooled, indices1 = self.pool1(x1)  # ADD: Save pooling indices

        # Encoder Block 2
        x2 = self.encoder2(x1_pooled)
        x2_pooled, indices2 = self.pool2(x2)  # ADD: Save pooling indices

        # Decoder Block 2
        x2_unpooled = self.unpool2(x2_pooled, indices2)  # ADD: Use pooling indices for unpooling
        x2_decoded = self.decoder2(x2_unpooled)

        # Decoder Block 1
        x1_unpooled = self.unpool1(x2_decoded, indices1)  # ADD: Use pooling indices for unpooling
        x1_decoded = self.decoder1(x1_unpooled)

        # return F.softmax(x1_decoded, dim=1)  # Final softmax activation for multi-class segmentation
        return x1_decoded


# Residual U-net
class ResidualUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super(ResidualUNet, self).__init__()
        # Residual Encoder Block 1
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.residual_conv1 = nn.Conv2d(in_channels, 64, kernel_size=1)

        # Residual Encoder Block 2
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.residual_conv2 = nn.Conv2d(64, 128, kernel_size=1)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
        )
        self.residual_conv_bottleneck = nn.Conv2d(128, 256, kernel_size=1)

        # Residual Decoder Block 2
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Skip connection from encoder2
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        # Residual Decoder Block 1
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Skip connection from encoder1
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Final Output Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder Level 1
        x1 = self.encoder1(x)
        x1_res = x1 + self.residual_conv1(x)  # Residual connection
        x1_pooled = self.pool1(x1_res)

        # Encoder Level 2
        x2 = self.encoder2(x1_pooled)
        x2_res = x2 + self.residual_conv2(x1_pooled)  # Residual connection
        x2_pooled = self.pool2(x2_res)

        # Bottleneck
        x3 = self.bottleneck(x2_pooled)
        x3_res = x3 + self.residual_conv_bottleneck(x2_pooled)  # Residual connection

        # Decoder Level 2
        x4_up = self.upconv2(x3_res)
        x4 = self.decoder2(torch.cat([x4_up, x2_res], dim=0))  # Skip connection
        x4_res = x4 + x4_up  # Residual connection

        # Decoder Level 1
        x5_up = self.upconv1(x4_res)
        x5 = self.decoder1(torch.cat([x5_up, x1_res], dim=0))  # Skip connection
        x5_res = x5 + x5_up  # Residual connection

        # Final Output
        out = self.final_conv(x5_res)
        # return F.softmax(out, dim=1)
        return out


# U2-net   Note: the original paper used Dilations INSTEAD OF DOWNSAMPLING
class U2Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super(U2Net, self).__init__()
        # Nested U-Nets as Residual U-Blocks
        self.encoder1 = UNet(in_channels, 64)  # First nested U-Net
        self.encoder2 = UNet(64, 128)         # Second nested U-Net

        # Bottleneck
        self.bottleneck = UNet(128, 256)     # Bottleneck nested U-Net

        # Decoder
        self.decoder2 = UNet(256 + 128, 128)  # Decoder with skip connections
        self.decoder1 = UNet(128 + 64, 64)    # Decoder with skip connections

        # Final output layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)  # First nested U-Net
        x2 = self.encoder2(x1)  # Second nested U-Net

        # Bottleneck
        x3 = self.bottleneck(x2)

        # Decoder with skip connections
        x4 = self.decoder2(torch.cat([x3, x2], dim=0))  # Skip connection from encoder2
        x5 = self.decoder1(torch.cat([x4, x1], dim=0))  # Skip connection from encoder1

        # Final output
        output = self.final(x5)
        # return F.softmax(output, dim=1)
        return output

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)


# Attention U-net
class AttentionUNet(UNet):
    def __init__(self, in_channels=3, out_channels=4):
        super(AttentionUNet, self).__init__()
        
        self.att2 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.att1 = Attention_block(F_g=128,F_l=128,F_int=64)
        
        
    def forward(self, x):
        x1 = self.encoder1(x)  # Encoder Level 1
        x2 = self.encoder2(x1)  # Encoder Level 2
        x3 = self.bottleneck(x2)  # Bottleneck
        print(x3.shape)
        # Apply attention block to the skip connection before passing to decoder
        a2 = self.att2(g=x2,x=x3)
        x4 = self.decoder2(torch.cat([x3, a2], dim=1))  # Decoder Level 2 with attention
        x5 = self.decoder1(torch.cat([x4, self.att1(g=x4,x=x1)], dim=1))  # Decoder Level 1 with attention

        # return F.softmax(x5, dim=1)  # Output
        return(x5)