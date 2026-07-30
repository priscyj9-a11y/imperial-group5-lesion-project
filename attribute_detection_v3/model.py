"""
U-Net model with a pretrained VGG16 encoder and 5 independent output heads for Task 2.

The actual neural network that uses VGG16 to squish down the image and then rebuild it, and ends with 5 separate small output heads
(one detector per attribute, since an image can have multiple attributes present at once)
"""

#Explanation: What is a channel? think of a color photo. It has 3 channels: red, green, blue. Three separate grids of numbers, stacked on top of each other, describing the same image. 
#As data flows deeper into a neural network, it stops literally being "red/green/blue" and instead becomes abstract learned features, but it's still organized the same way: a stack of grids, and "channels" just means "how many grids are stacked."

import torch
import torch.nn as nn #PyTorch's toolbox specifically for building neural network layers (e.g. Conv2d, Linear, BatchNorm2d)
import torchvision.models as models  #library of pretrained computer vision models (we use VGG16 already trained on ImageNet)


#We define a new custom class of neural network layer/block which inherits from nn.Module (Pytorch's base class for all layers)
class DoubleConv(nn.Module):
    """This is the bottleneck block of U-Net: it refines the compressed features from the encoder before we start decoding them back up to full resolution. 
    It does this by running two consecutive 3×3 convolutions, each followed by batch normalization and ReLU activation."""

    def __init__(self, in_ch, out_ch): #we have to define how many input channels DoubleConv receives and how many output channels it should produce
        super().__init__() #calls nn.Module's constructor (required for every Pytorch layer)
        self.block = nn.Sequential( #nn.Sequential chains a list of layers together so data flows through them one after another automatically, in order. We're building that chain and storing it as self.block.

            #A convolution layer. Picture a small 3×3 (kernel_size=3) window sliding across the image, one small step at a time. At each position, it takes the 9 numbers under that window, multiplies them by 9 learned weights, adds them up into one number, and that becomes one pixel of the output. 
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), #padding=1 adds a 1-pixel border of zeros around the edges before sliding. Without this, each 3×3 pass would shrink the image slightly (lose 1 pixel off each edge); the padding exactly compensates, so output width/height equals input width/height.

            #Batch normalization. After the convolution,the numbers flowing through the network can end up wildly different in scale from layer to layer. This rescales the numbers so they have roughly zero mean and unit variance across the batch, for each channel. (keeps numbers stable -> training faster and more reliable)
            nn.BatchNorm2d(out_ch), 

            #activation function: ReLU - any negative number becomes 0, positive numbers stay the same. This is what lets the network learn non-linear patterns (without it, stacking convolutions would just be equivalent to one big linear operation - just 1 convolution). inplace=True is a small memory optimization (it modifies the tensor directly instead of making a copy).
            nn.ReLU(inplace=True), 

            #The same three steps again — conv, batch norm, ReLU — but this time both input and output are out_ch channels (this is why it's DOUBLE Conv)
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x): #Every nn.Module needs a forward method: defines what happens when data actually passes through the layer. Here: take input x, run it through the sequential chain we built, return the result.
        return self.block(x)

"""Why do we use 2 consecutive convolutions instead of just 1? The receptive field of a single 3×3 convolution is 3×3 pixels. Two consecutive 3×3 convolutions have a receptive field of 5×5 pixels (the first conv sees 3×3, the second conv sees 3×3 of that, which covers a total of 5×5 in the original image). So two convs can learn more complex patterns than one conv, but with fewer parameters than a single 5×5 convolution would require. This is standard practice in U-Net and many other segmentation networks."""



#Another custom block for the U-Net decoder: it upsamples the feature map, concatenates it with the matching skip connection from the encoder, and then runs a DoubleConv to refine the combined features.
class UpBlock(nn.Module):
    """Upsample, concatenate with skip connection, then DoubleConv."""
    def __init__(self, in_ch, skip_ch, out_ch): #in_ch = channels coming in from below (the previous, smaller-resolution stage); skip_ch = channels of the matching skip connection from the encoder we'll be combining with; out_ch = channels we want to output.
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2) #transposed convolution: increases spatial resolution (the opposite of a normal convolution, which usually decreases or keeps it the same). kernel_size=2, stride=2 doubles the height and width. It also changes the channel count from in_ch down to in_ch // 2
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch) #After upsampling, we're going to concatenate our upsampled features with the skip connection. So the input to this DoubleConv will have (in_ch // 2) channels from upsampling plus skip_ch channels from the skip connection

    def forward(self, x, skip): #This block's forward pass takes two inputs: x (the feature map coming up from the deeper/smaller stage) and skip (the matching feature map saved earlier from the encoder, at this same resolution).
        x = self.up(x) #Upsample x: double its height and width, halve its channels
        if x.shape[-2:] != skip.shape[-2:]: #.shape[-2:] grabs the last two dimensions of the tensor — height and width (PyTorch tensors here are shaped [batch, channels, height, width], so index -2 is height, -1 is width). This line checks: does our newly-upsampled x have the exact same height and width as skip? Sometimes, due to rounding when images have odd dimensions, they can differ by 1 pixel.
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False) #If they don't match, force x to be resized to exactly skip's height/width using bilinear interpolation (smooth pixel blending).
        x = torch.cat([x, skip], dim=1) #Concatenate x and skip along dimension 1, which is the channel dimension. So if x has 256 channels and skip has 256 channels, the result has 512 channels, same height/width as both.
        return self.conv(x) #Pass that combined tensor through the DoubleConv we built in __init__, which learns how to blend the two information sources into out_ch channels, and return the result.


#VGG16Encoder
class VGG16Encoder(nn.Module):
    """
    Wraps torchvision's VGG16 features to expose intermediate activations
    at each of the 5 pooling stages, needed for U-Net skip connections.
    """
    def __init__(self, pretrained=True): #flag pretrained (default True) for whether to load ImageNet-trained weights or start random.
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None #None means random initialization of the weights
        vgg = models.vgg16(weights=weights).features #builds the full VGG16 model (loading those pretrained weights if given). .features then grabs only the convolutional/pooling part of VGG16 — not its final classification layers (which were for ImageNet's 1000 classes, which we don't want). vgg is now just a flat sequence of conv/relu/maxpool layers.
        self.stage1 = vgg[0:4] #vgg behaves like a Python list of layers here. This line slices out layers at index 0, 1, 2, 3 (index 4 excluded) and stores them as stage1. Looking at VGG16's known structure, that's 2 conv layers + 2 ReLUs — before any downsampling happens.
        self.stage2 = vgg[4:9] #Same idea — each slice grabs the next chunk of VGG16's layers, with each chunk starting right after a maxpool (which halves resolution). # pool + conv2_1, conv2_2 -> 128 ch
        self.stage3 = vgg[9:16] # pool + conv3_1..3      -> 256 ch
        self.stage4 = vgg[16:23] # pool + conv4_1..3      -> 512 ch
        self.stage5 = vgg[23:30] # pool + conv5_1..3      -> 512 ch
        self.final_pool = vgg[30] #VGG16's very last maxpool layer, which we'll apply after stage5 to get the smallest, most compressed representation (the bottleneck).

    def forward(self, x):
        s1 = self.stage1(x)   #Run x through stage1. Because no pooling has happened yet in this stage, the output resolution is the same as the input
        s2 = self.stage2(s1)  #Each stage takes the previous stage's output as input, and each one contains a maxpool at its start, so resolution keeps halving: 1/2, 1/4, 1/8, 1/16 of the original height/width.
        s3 = self.stage3(s2)  # H/4
        s4 = self.stage4(s3)  # H/8
        s5 = self.stage5(s4)  # H/16
        bottleneck = self.final_pool(s5)  # One more maxpool applied to s5, halving resolution again to 1/32 — this is the deepest, most "squished" point of the whole network.
        return bottleneck, [s5, s4, s3, s2, s1] #Return two things: the bottleneck (to be fed into the decoder next), and a list of all 5 intermediate outputs — but notice the order is reversed (s5 first, s1 last). That's deliberate, because the decoder will need them in that order (deepest/smallest skip first, since that's what it meets right after the bottleneck).

#Main Model
class UNetMultiHead(nn.Module):
    """
    U-Net with a pretrained VGG16 encoder and 5 independent output heads
    for multi-attribute segmentation (Task 2).
    """
    def __init__(self, pretrained=True, freeze_encoder=False, num_attributes=5):
        super().__init__()
        self.encoder = VGG16Encoder(pretrained=pretrained) #create the encoder and store it
        if freeze_encoder:
            for p in self.encoder.parameters(): #self.encoder.parameters() gives every learnable weight/bias inside the encoder.
                p.requires_grad = False  #requires_grad = False tells PyTorch "don't compute gradients for this — don't update it during training." So if freeze_encoder=True, the pretrained VGG16 weights stay exactly as they are; only the decoder and heads will learn.
        self.bottleneck_conv = DoubleConv(512, 512) #After the encoder's bottleneck (which has 512 channels, matching VGG16's deepest layer), run it through one DoubleConv to further process those features before starting to decode upward. Channels stay 512 in, 512 out.
        self.up1 = UpBlock(in_ch=512, skip_ch=512, out_ch=512) #Five UpBlocks chained conceptually (though not yet connected — that happens in forward). Each one's in_ch matches the previous block's out_ch . Each skip_ch matches the channel count of the encoder stage it will be paired with
        self.up2 = UpBlock(in_ch=512, skip_ch=512, out_ch=256)
        self.up3 = UpBlock(in_ch=256, skip_ch=256, out_ch=128)
        self.up4 = UpBlock(in_ch=128, skip_ch=128, out_ch=64)
        self.up5 = UpBlock(in_ch=64, skip_ch=64, out_ch=32)
        self.heads = nn.ModuleList([nn.Conv2d(32, 1, kernel_size=1) for _ in range(num_attributes)]) #nn.ModuleList is a container that holds a list of layers and properly registers them with PyTorch (a plain Python list wouldn't). Inside, we create num_attributes (5, by default) separate 1×1 convolutions, each taking 32 input channels and producing 1 output channel

    def forward(self, x):
        input_size = x.shape[-2:] #Save the input's height and width, so we can guarantee the output matches it later.
        bottleneck, skips = self.encoder(x)  #Run the image through the encoder. This gives us two things back (matching what VGG16Encoder.forward returns): the bottleneck tensor, and the list of skip tensors (already in the deep-to-shallow order we need). (skips = [s5, s4, s3, s2, s1])
        x = self.bottleneck_conv(bottleneck) #Process the bottleneck through that extra DoubleConv layer. We reuse the variable name x here for the "current tensor as it flows through the decoder."
        x = self.up1(x, skips[0]) #Feed x (currently the processed bottleneck) and skips[0] (which is s5, the deepest skip) into up1. Remember UpBlock.forward upsamples x, concatenates it with the skip, and runs it through a DoubleConv. The result — now bigger, with up1's out_ch=512 channels — becomes the new x.
        x = self.up2(x, skips[1]) #Repeat that same process 4 more times, each time doubling resolution again and combining with the next skip connection
        x = self.up3(x, skips[2])
        x = self.up4(x, skips[3])
        x = self.up5(x, skips[4]) #By the end of up5, x should be back at (or very close to) the original input resolution, with 32 channels (matching up5's out_ch=32).
        if x.shape[-2:] != input_size: #Same safety check as before: if for any reason the decoder's output size doesn't exactly match the original input size, force-resize it
            x = nn.functional.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        outputs = [torch.sigmoid(head(x)) for head in self.heads] #For each of the 5 heads in self.heads, run the shared decoder output x through that head (a 1×1 conv, giving [B, 1, H, W]), then apply torch.sigmoid to squash every value into the range 0–1 (interpretable as a probability). We end up with a Python list of 5 tensors, each [B, 1, H, W].
        return torch.cat(outputs, dim=1) #Glue those 5 separate [B, 1, H, W] tensors together along the channel dimension (dim=1), producing one tensor of shape [B, 5, H, W] — this is the final output of the whole model: 5 probability maps, one per attribute.


if __name__ == "__main__": #only run the following code if this file is executed directly (e.g. python unet_multihead.py), not if it's imported into another script." It keeps the test code from running accidentally when you just want to import UNetMultiHead elsewhere.
    # Quick smoke test
    model = UNetMultiHead(pretrained=False)  #Create an instance of the model. pretrained=False here just skips downloading the (large) ImageNet weights, since this is only a shape/wiring test, not real training.
    dummy = torch.randn(2, 3, 256, 256) #Create a fake input tensor: 2 images (batch size 2), 3 channels (RGB), 256×256 pixels, filled with random numbers from a normal distribution. It doesn't need to be a real image — we're only checking that the tensor shapes flow correctly through the network without crashing.
    out = model(dummy) #Run the dummy batch through the model.
    print("Output shape:", out.shape)  # expect [2, 5, 256, 256]
    assert out.shape == (2, 5, 256, 256)
    print("Smoke test passed.")