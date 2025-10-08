import torch
import torch.nn.functional as F
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from models.spatialtransform import spatial_transformer


class ViTClassifier(torch.nn.Module):
    def __init__(self, config, feat_ch):
        super(ViTClassifier, self).__init__()
        self.psize = (config.patch_size, config.patch_size)

        # Load pretrained ViT backbone (or custom if you prefer)
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = torch.nn.Identity()  # remove classification head

        # Optional: freeze ViT if using pretrained features
        # for p in self.vit.parameters():
        #     p.requires_grad = False

        self.classifier = torch.nn.Linear(768, config.num_classes)

    def forward(self, featuremap, bboxes, stride=None):
        B, N, _ = bboxes.shape
        _, C, _, _ = featuremap.shape

        patches = spatial_transformer(featuremap, bboxes, self.psize).view(
            -1, C, *self.psize
        )
        patches = F.interpolate(
            patches, size=(224, 224), mode="bilinear", align_corners=False
        )
        print(patches.shape)
        features = self.vit(patches)  # [B*N, 768]
        logits = self.classifier(features)  # [B*N, num_classes]
        logits = logits.view(B, N, -1)
        labels = torch.argmax(logits, dim=2)

        return labels, logits, {"patches": patches, "features": features}
