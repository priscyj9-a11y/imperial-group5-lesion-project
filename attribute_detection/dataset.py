"""
Dataset for Task 2 multi-attribute segmentation.

For a given image ID, it opens the photo and all 5 attribute masks, resizes everything to 512x512 using 
preprocessing/transforms.py and returns a dictionary: {"image": ..., "attributes": ..., "image_id": ...} storing the tensors for 
the image and attribute masks (that will be used as the model input
"""

from pathlib import Path #path handling
import pandas as pd #reads CSV files
import torch #Pytorch for tensor handling
from torch.utils.data import Dataset #base class for datasets
import torchvision.transforms.functional as TF #for normalization
from preprocessing.transforms import load_image, load_mask, DEFAULT_SIZE #pulls in Faisal's preprocessing functions for loading images and masks, and the default size of 512x512


class AttributeDataset(Dataset):
    """
    Dataset returning:
        image            (3,H,W)
        attribute masks  (A,H,W)
    """

    def __init__(self, csv_file, data_root, attributes, size=DEFAULT_SIZE, augment=True): 
        """Initialize the dataset."""
        self.df = pd.read_csv(csv_file) #loads the whole CSV (e.g. train_ids.csv or val_ids.csv - list of image IDs ) into memory as a DataFrame 
        self.data_root = Path(data_root) #where data is stored
        self.attributes = attributes #list of attribute names (imported from mask_io.py)
        self.size = size #512x512
        self.augment = augment #data augmentation
        
       

    def __len__(self): #required by Pytorch's Dataset
        return len(self.df) #tells DataLoader how many samples exist (how many rows in the CSV file)

    def __getitem__(self, idx):
        image_id = str(self.df.iloc[idx]["image_id"]).strip().zfill(6) #Grabs row idx's image_id, forces it to a string, strips stray whitespace, and zero-pads to 6 digits ("1" → "000001")
        image = load_image(self.data_root / "images" / f"{image_id}.jpg", self.size) #builds the real image path by calling preprocessing/transforms.py's load_image() function, which loads the image and resizes it to 512x512
        
        attribute_masks = []
        for attribute in self.attributes: #loops over the 5 attribute names
            mask_path = self.data_root / "task2_gt" / f"{image_id}_attribute_{attribute}.png" #builds each one's real file path (e.g. "data_root/task2_gt/000001_attribute_pigment_network.png")
            if mask_path.exists():
                #load_mask(mask_path, self.size) runs first - function defined in preprocessing/transforms.py that opens the PNG file at mask_path using PIL, converts it into a NumPy array, resizes it to self.size (512×512), and converts pixel values to 0/1. That NumPy array is the thing load_mask returns.
                #torch.from_numpy(...) runs second, wrapping around whatever load_mask just gave back. Takes NumPy array and turns it into a PyTorch tensor
                #.float() runs last, converting that tensor's data type to 32-bit floating point.
                mask = torch.from_numpy(load_mask(mask_path, self.size)).float()
            else: #when an attribute mask file doesn't exist on disk for some image
                mask = torch.zeros((self.size, self.size)).float() #we return a 512x512 mask that's 0 at every pixel (no attribute present) instead of crashing the program

            attribute_masks.append(mask)

        # Convert NumPy arrays from transforms.py into tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() #image was [H, W, 3] (height, width, color channels — standard image layout). .permute(2, 0, 1) reorders to [3, H, W] (channels-first), which is what PyTorch's conv layers expect.
        attribute_masks = torch.stack(attribute_masks).float() #Combines the list of 5 separate mask tensors into one tensor [5, H, W].

        # Shared random augmentation: same flip/rotation applied to the
        # image AND all 5 attribute masks together, so they stay aligned.
        if self.augment:

            #Random horizontal flip with 50% probability
            if torch.rand(1) < 0.5:
                image = TF.hflip(image)
                attribute_masks = TF.hflip(attribute_masks)

            #Random vertical flip with 50% probability
            if torch.rand(1) < 0.5:
                image = TF.vflip(image)
                attribute_masks = TF.vflip(attribute_masks)

            #Random rotation between -30 and +30 degrees with 70% probability (more aggressive)
            if torch.rand(1) < 0.7:
                angle = float(torch.empty(1).uniform_(-30, 30))
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                attribute_masks = TF.rotate(attribute_masks, angle, interpolation=TF.InterpolationMode.NEAREST)

            #Random brightness adjustment between 0.7x and 1.3x with 60% probability (more aggressive)
            if torch.rand(1) < 0.6:
                brightness_factor = float(torch.empty(1).uniform_(0.7, 1.3))
                image = TF.adjust_brightness(image, brightness_factor)

            #Random contrast adjustment between 0.7x and 1.3x with 60% probability (more aggressive)
            if torch.rand(1) < 0.6:
                contrast_factor = float(torch.empty(1).uniform_(0.7, 1.3))
                image = TF.adjust_contrast(image, contrast_factor)

            #Random color jitter: shift hue/saturation with 50% probability
            if torch.rand(1) < 0.5:
                saturation_factor = float(torch.empty(1).uniform_(0.7, 1.3))
                image = TF.adjust_saturation(image, saturation_factor)

            #Random zoom/scale: crop and resize with 50% probability
            if torch.rand(1) < 0.5:
                scale = float(torch.empty(1).uniform_(0.85, 1.0))  # Zoom in by cropping 15%
                h, w = image.shape[1], image.shape[2]
                crop_h, crop_w = int(h * scale), int(w * scale)
                top = torch.randint(0, h - crop_h + 1, (1,)).item()
                left = torch.randint(0, w - crop_w + 1, (1,)).item()
                image = TF.crop(image, top, left, crop_h, crop_w)
                image = TF.resize(image, (h, w), interpolation=TF.InterpolationMode.BILINEAR)
                attribute_masks = TF.crop(attribute_masks, top, left, crop_h, crop_w)
                attribute_masks = TF.resize(attribute_masks, (h, w), interpolation=TF.InterpolationMode.NEAREST)

        # ImageNet normalisation
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Rescales pixel values using ImageNet's mean/std which is required because the model's pretrained VGG16 encoder expects inputs normalized this specific way
        #Note: masks stay as raw 0/1 values, since they're targets, not something being fed through the pretrained encoder.

        return {"image": image, "attributes": attribute_masks, "image_id": image_id,} #returns a dictionary containing the image tensor, the 5-attribute mask tensor, and the image ID string for this sample