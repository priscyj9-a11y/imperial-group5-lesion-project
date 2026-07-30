"""
Loss function + Dice metric for Task 2 multi-attribute segmentation.
the "how wrong were we" scorer used during training to adjust weights in order to minimize it. 
It combines two measurements (BCE + Dice)


Why not plain BCE alone: preprocessing/attribute_stats.py output showed some
attributes are very sparse (streaks present in only 5.3% of images,
negative_network in 7.4%). A model can get deceptively low BCE loss by
just predicting "absent" everywhere for these since it averages over every
pixel, and background pixels vastly outnumber foreground ones.

Dice loss directly measures overlap between prediction and ground truth
foreground, so it doesn't get fooled by an all-background prediction the
same way (a model that predicts nothing gets Dice ~0, not a deceptively
low loss). Combining BCE (stable gradients, pixel-level supervision) with
Dice (robust to imbalance) is standard practice for exactly this kind of
sparse medical segmentation problem, and matches the "Segmentation Loss
(Dice Loss) ... Classification Loss (BCE)" combined-loss slide in the
Project Briefing.
"""

import torch
import torch.nn as nn
from attribute_detection_v3.task2_eval.mask_io import ATTRIBUTE_NAMES


def dice_score(preds, targets, eps=1e-6):
    """
    Computes the Dice coefficient per attribute, averaged over the batch.

    preds:   [B, 5, H, W] probabilities (already sigmoid-ed), in [0, 1]
    targets: [B, 5, H, W] binary ground truth masks, values in {0, 1}

    Returns: tensor of shape [5] — one Dice score per attribute.

    Dice = (2 * |intersection|) / (|preds| + |targets|)
    matches the "Result evaluation" slide's Dice Score formula exactly.
    """
    preds = (preds > 0.5).float()  #checks every single number in the tensor against 0.5, producing True/False everywhere. .float() turns those into 1.0/0.0. This forces the predictions into a hard yes/no — "is this attribute present at this pixel, or not."
    intersection = (preds * targets).sum(dim=(0, 2, 3)) #Since both preds and targets are now made of only 0s and 1s, multiplying them together gives 1 only where both agree there's foreground, and 0 everywhere else. .sum(dim=(0, 2, 3)) adds up all those matching 1s — across the batch (dimension 0), height (dimension 2), and width (dimension 3) — but deliberately leaves dimension 1 (the attribute dimension) alone. So you end up with one single number per attribute, not one number for the whole batch.
    union = preds.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3)) #Counts up all the foreground pixels in the predictions, plus all the foreground pixels in the ground truth, separately, then adds those two counts together — again, one number per attribute.
    return (2.0 * intersection + eps) / (union + eps) #The actual Dice formula: 2 × intersection ÷ union. The tiny eps added to both top and bottom exists purely to avoid a 0 ÷ 0 crash - with eps, it resolves cleanly to 1.0, treating "correctly predicted nothing" as a perfect score.


class DiceBCELoss(nn.Module): #This loss is also built as an nn.Module — even though it has no weights of its own to learn — because that's what lets it move to GPU/CPU alongside everything else in one .to(device) call.
    """
    Combined per-attribute Dice + BCE loss.

    attribute_weights: optional list/tensor of 5 weights, one per attribute,
    to upweight rare attributes (streaks, negative_network) in the loss.
    If None, all attributes are weighted equally.
    """

    def __init__(self, attribute_weights=None, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss(reduction="none")  #Creates PyTorch's ready-made Binary Cross-Entropy loss. reduction="none" means: "don't automatically average this down to one number — give me back the raw, per-pixel values" — because we need to process each attribute's numbers separately before combining, not all mixed together from the start.
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        if attribute_weights is None: #If no custom weighting was given: build a tensor that's just [1.0, 1.0, 1.0, 1.0, 1.0] — all equal, meaning no attribute gets special treatment. If custom weights were passed in, convert whatever was given into a proper PyTorch tensor.
            attribute_weights = torch.ones(len(ATTRIBUTE_NAMES))
        else:
            attribute_weights = torch.as_tensor(attribute_weights, dtype=torch.float32)
        self.register_buffer("attribute_weights", attribute_weights) #tells PyTorch "this tensor belongs to the module and should move with it (e.g. to GPU) whenever .to(device) is called — but it is not a learnable parameter the optimizer should ever adjust." That distinction matters because these weights are a fixed choice you make, not something the network should be allowed to change on its own during training.

    def forward(self, preds, targets):
        """
        preds:   [B, 5, H, W] probabilities (model output — already sigmoid-ed)
        targets: [B, 5, H, W] binary ground truth

        Returns: (total_loss, per_attribute_loss)
            total_loss:        scalar, for .backward()
            per_attribute_loss: tensor [5], for logging/debugging which
                                attribute is struggling
        """
        bce_per_pixel = self.bce(preds, targets)          # Runs BCE. Because we set reduction="none" earlier, this hands back a full tensor the exact same shape as the input — one loss value for every single pixel, nothing averaged yet. [B, 5, H, W]
        bce_per_attr = bce_per_pixel.mean(dim=(0, 2, 3))  # Averages that huge per-pixel tensor down — across batch, height, width — but again, keeps the attribute dimension separate. Result: one BCE number per attribute. [5]

        smooth = 1.0
        intersection = (preds * targets).sum(dim=(0, 2, 3))
        union = preds.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
        dice_coef = (2.0 * intersection + smooth) / (union + smooth) #Same Dice formula as the standalone function above — but here, preds is never thresholded to hard 0/1. It stays as raw probabilities (like 0.73, 0.12). 
        #This distinction really matters: a hard > 0.5 comparison has essentially zero gradient almost everywhere, so it can't be used anywhere near .backward() 
        #Using raw probabilities is exactly what makes this version differentiable and trainable, while the other function (meant only for reporting scores, never training) can safely threshold.
        dice_loss_per_attr = 1.0 - dice_coef #Dice coefficient is "higher is better" (1.0 = perfect). But a loss needs to be "lower is better," for the optimizer to minimize — subtracting from 1 flips that direction.

        per_attribute_loss = self.bce_weight * bce_per_attr + self.dice_weight * dice_loss_per_attr #Blends the two loss types together, per attribute, using the 50/50 (or custom) mix.
        weighted = per_attribute_loss * self.attribute_weights #Multiplies each attribute's combined loss by its corresponding weight — this is the step that actually amplifies rare attributes, if custom weights were supplied.

        total_loss = weighted.sum() / self.attribute_weights.sum() #Adds all 5 (now-weighted) attribute losses into one single number, then divides by the sum of the weights themselves. 
        #This division matters: without it, choosing bigger weights would inflate the overall loss's scale, which could mess with how big a "step" your optimizer takes — dividing keeps the final loss on a consistent scale no matter what weights you pick.

        return total_loss, per_attribute_loss.detach()
        #Hands back two things: the single scalar (for .backward() to use), and the per-attribute breakdown (purely for you to print/log, so .detach() disconnects it from gradient tracking 
        #it's not needed for training, only for humans to look at).

if __name__ == "__main__":
    # Smoke test with fake data — checks shapes and that loss decreases as
    # predictions get closer to the targets.
    torch.manual_seed(0)

    B, C, H, W = 4, 5, 64, 64
    targets = (torch.rand(B, C, H, W) > 0.8).float()

    # Example weighting inversely related to prevalence from attribute_stats.py:
    # pigment_network 61.1%, negative_network 7.4%, streaks 5.3%,
    # milia_like_cyst 21.3%, globules 22.6%
    prevalence = torch.tensor([0.611, 0.074, 0.053, 0.213, 0.226])
    attribute_weights = 1.0 / prevalence
    attribute_weights = attribute_weights / attribute_weights.sum() * 5

    loss_fn = DiceBCELoss(attribute_weights=attribute_weights)

    random_preds = torch.rand(B, C, H, W)
    loss_random, per_attr_random = loss_fn(random_preds, targets)
    print("Random preds  -> total loss:", loss_random.item())
    print("  per-attribute loss:", per_attr_random.tolist())

    perfect_preds = targets.clone().clamp(1e-4, 1 - 1e-4)
    loss_perfect, per_attr_perfect = loss_fn(perfect_preds, targets)
    print("Perfect preds -> total loss:", loss_perfect.item())
    print("  per-attribute loss:", per_attr_perfect.tolist())

    assert loss_perfect.item() < loss_random.item(), "Loss should be lower for near-perfect predictions!"
    assert loss_perfect.item() < 0.05, f"Expected near-zero loss for perfect predictions, got {loss_perfect.item()}"

    scores = dice_score(perfect_preds, targets)
    print("Dice scores (perfect preds):", scores.tolist())
    assert all(s > 0.95 for s in scores.tolist()), "Dice should be near 1.0 for near-perfect predictions"

    print("Smoke test passed.")