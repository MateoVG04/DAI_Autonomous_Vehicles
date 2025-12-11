import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """A block of two convolutional layers, each followed by BatchNorm and ReLU activation."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block: a maxpool followed by a double convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block: an up-convolution followed by a double convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Use a transposed convolution to upsample the feature map
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: The feature map from the previous layer in the decoder (to be upsampled).
            x2: The corresponding feature map from the encoder (the skip connection).
        """
        # Upsample x1 to match the spatial dimensions of x2
        x1 = self.up(x1)

        # Concatenate the upsampled feature map (x1) with the skip connection (x2)
        # The skip connection provides high-resolution features to the decoder.
        x = torch.cat([x2, x1], dim=1)

        # Apply the double convolution block
        return self.conv(x)


class UNet(nn.Module):
    """
    The main U-Net architecture.
    The number of output classes is configurable, making this model flexible
    for both multi-class and binary segmentation.
    """

    def __init__(self, n_channels, n_classes):
        """
        Args:
            n_channels (int): Number of channels in the input image (e.g., 3 for RGB).
            n_classes (int): Number of classes for the output segmentation map.
                             (e.g., 23 for full semantic, 1 for binary lane detection).
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- Encoder Path ---
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # --- Decoder Path ---
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # --- Final Output Layer ---
        # A 1x1 convolution that maps the 64-channel feature map to the desired number of classes.
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # The forward pass defines the flow of data through the network.

        # Encoder
        x1 = self.inc(x)  # Initial convolution
        x2 = self.down1(x1)  # Downsample 1
        x3 = self.down2(x2)  # Downsample 2
        x4 = self.down3(x3)  # Downsample 3
        x5 = self.down4(x4)  # Bottom of the "U"

        # Decoder with Skip Connections
        # The output of each encoder layer is passed to the corresponding decoder layer.
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Final output
        logits = self.outc(x)
        return logits