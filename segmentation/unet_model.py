"""U-Net architecture for lesion segmentation (ISIC dataset).

Input : RGB photo, shape (batch, 3, 512, 512)
Output: raw logits (NOT probabilities yet), shape (batch, 1, 512, 512)
        - apply sigmoid outside this file to get a probability per pixel
        - threshold at 0.5 to get a binary mask

Standard U-Net shape: an encoder that shrinks the image while learning WHAT
is in it, then a decoder that grows it back while learning WHERE it is,
using skip connections to recover fine detail the encoder throws away.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x2. The basic repeated block in U-Net.

    Two conv layers instead of one gives the network more capacity to learn
    at each spatial resolution before the next up/downsample.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """One encoder step: halve spatial size (MaxPool), then DoubleConv.

    Going 512 -> 256 -> 128 -> 128 -> 64 -> 32 across the encoder lets deeper
    layers see a wider area of the original image ("larger receptive field"),
    which is how the network learns shape/context, not just local texture.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """One decoder step: upsample, concatenate with the matching encoder
    feature map (the "skip connection"), then DoubleConv.

    WHY THE SKIP CONNECTION MATTERS: the encoder's downsampling throws away
    precise pixel locations to gain a broader view. The decoder alone can't
    recover that precision. Concatenating the encoder's feature map at the
    same resolution hands that fine spatial detail straight back, which is
    exactly what a segmentation mask needs at its edges.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # if sizes are off by a pixel (can happen with odd dimensions),
        # pad so the concatenation below doesn't crash
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Full U-Net for binary lesion segmentation.

    in_channels=3  -> RGB photo
    out_channels=1 -> one mask channel (lesion vs background)
    base_channels  -> width of the first layer; doubles at each Down step.
                      64 is the standard U-Net default. Lower it (e.g. 32)
                      if training is too slow or runs out of GPU memory.

    Returns LOGITS, not probabilities - pair this with
    nn.BCEWithLogitsLoss during training (numerically more stable than
    doing sigmoid + BCELoss separately).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 64):
        super().__init__()

        # encoder (contracting path)
        self.in_conv = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)  # bottleneck

        # decoder (expanding path) - each Up takes the previous decoder
        # output PLUS the matching encoder skip connection
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)

        # 1x1 conv: collapse features down to out_channels, no spatial change
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder - save each step's output, we need it later as a skip connection
        x1 = self.in_conv(x)   # 512x512, base_channels
        x2 = self.down1(x1)    # 256x256, base_channels*2
        x3 = self.down2(x2)    # 128x128, base_channels*4
        x4 = self.down3(x3)    # 64x64,   base_channels*8
        x5 = self.down4(x4)    # 32x32,   base_channels*16 (bottleneck)

        # decoder - each step upsamples AND merges with the matching x_i above
        x = self.up1(x5, x4)   # -> 64x64
        x = self.up2(x, x3)    # -> 128x128
        x = self.up3(x, x2)    # -> 256x256
        x = self.up4(x, x1)    # -> 512x512

        return self.out_conv(x)  # (batch, 1, 512, 512), raw logits


if __name__ == "__main__":
    # quick sanity check: does a dummy batch flow through without shape errors?
    model = UNet(in_channels=3, out_channels=1)
    dummy_input = torch.randn(2, 3, 512, 512)  # batch of 2 fake RGB images
    output = model(dummy_input)
    print(f"input shape : {dummy_input.shape}")
    print(f"output shape: {output.shape}")
    assert output.shape == (2, 1, 512, 512), "output shape is wrong - check Up/Down blocks"
    print("shape check passed")