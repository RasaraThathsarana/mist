# import torch
# import torch.nn.functional as F
# from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
# from models.spatialtransform import spatial_transformer


# class ViTClassifier(torch.nn.Module):
#     def __init__(self, config, feat_ch):
#         super(ViTClassifier, self).__init__()
#         self.psize = (config.patch_size, config.patch_size)

#         # Load pretrained ViT backbone (or custom if you prefer)
#         self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
#         self.vit.heads = torch.nn.Identity()  # remove classification head

#         # Optional: freeze ViT if using pretrained features
#         # for p in self.vit.parameters():
#         #     p.requires_grad = False

#         self.classifier = torch.nn.Linear(768, config.num_classes)

#     def forward(self, featuremap, bboxes, stride=None):
#         B, N, _ = bboxes.shape
#         _, C, _, _ = featuremap.shape

#         patches = spatial_transformer(featuremap, bboxes, self.psize).view(
#             -1, C, *self.psize
#         )
#         patches = F.interpolate(
#             patches, size=(16, 16), mode="bilinear", align_corners=False
#         )
#         features = self.vit(patches)  # [B*N, 768]
#         logits = self.classifier(features)  # [B*N, num_classes]
#         logits = logits.view(B, N, -1)
#         labels = torch.argmax(logits, dim=2)

#         return labels, logits, {"patches": patches, "features": features}



import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from models.spatialtransform import spatial_transformer

class ViTClassifier(nn.Module):
    def __init__(self, config, feat_ch):
        """
        config: configuration object with attributes:
            - patch_size: used for spatial transformer
            - num_classes: number of output classes
        feat_ch: number of channels in the feature map
        """
        super(ViTClassifier, self).__init__()
        self.psize = (config.patch_size, config.patch_size)
        self.num_classes = config.num_classes

        # Pretrained ViT backbone
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # Remove patch embedding and classification head
        self.vit.patch_embed = nn.Identity()
        self.vit.pos_embed = None
        self.vit.cls_token = None
        self.vit.heads = nn.Identity()

        # Project feature map channels to 3 if needed
        self.channel_proj = nn.Conv2d(feat_ch, 3, kernel_size=1) if feat_ch != 3 else None

        # Flattened patch → ViT embedding
        self.flatten_proj = nn.Linear(3 * self.psize[0] * self.psize[1], 768)

        # Classification head
        self.classifier = nn.Linear(768, self.num_classes)

    def forward(self, featuremap, bboxes, stride=1):
        """
        featuremap: [B, C, H, W]
        bboxes: [B, N, 5] -> [x_center, y_center, w, h, ...]
        stride: scale from original image to featuremap
        """
        B, N, _ = bboxes.shape
        _, C, H, W = featuremap.shape

        # Project channels to 3 if needed
        if self.channel_proj is not None:
            featuremap = self.channel_proj(featuremap)

        # Convert center+size to corner coordinates (x1,y1,x2,y2)
        x_center, y_center, w, h = bboxes[:, :, 0], bboxes[:, :, 1], bboxes[:, :, 2], bboxes[:, :, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        boxes = torch.stack([x1, y1, x2, y2], dim=2)  # [B, N, 4]

        # Extract patches using spatial transformer
        patches = spatial_transformer(featuremap, boxes, self.psize)  # [B, N, C, ph, pw]
        patches = patches.view(B * N, 3, *self.psize)  # flatten batch dimension

        # Flatten each patch to 1D and project to 768-dim embedding
        patch_tokens = patches.flatten(1)  # [B*N, 3*ph*pw]
        patch_tokens = self.flatten_proj(patch_tokens)  # [B*N, 768]

        # Add batch dimension as seq_len=1 for ViT encoder
        patch_tokens = patch_tokens.unsqueeze(1)  # [B*N, 1, 768]
        features = self.vit.encoder(patch_tokens)  # [B*N, 1, 768]
        features = features.squeeze(1)  # [B*N, 768]

        # Classification
        logits = self.classifier(features)  # [B*N, num_classes]
        logits = logits.view(B, N, -1)     # [B, N, num_classes]
        labels = torch.argmax(logits, dim=2)  # [B, N]

        return labels, logits, {"patches": patches, "features": features}
