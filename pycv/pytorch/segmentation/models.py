from __future__ import annotations
import torch.nn as nn
import ssl
import ssl
import torch.nn as nn
import torch
import torchvision.models.segmentation as segmentation
from ..base.base_model import BaseModel
ssl._create_default_https_context = ssl._create_unverified_context


class SegmentationModel(BaseModel):
    def __init__(self, num_classes: int, name: str):
        super().__init__(name)
        self.num_classes = num_classes

    def forward(self, x: torch.FloatTensor):
        raise Exception("Base class forward(x) not implemented")

    def process_predictions(self, preds: torch.FloatTensor):
        """
        Converts a tensor of shape
        :param preds:
        :return:
        """
        if len(preds.shape) == 3:
            dim = 0
        elif len(preds.shape) == 4:
            dim = 1
        else:
            raise Exception("Invalid shape passed- ", preds.shape)

        n_classes = preds.shape[dim]
        error_str = "Invalid tensor passed - expected shape (batch_size, n_classes, h, w) or (n_classes, h, w)" \
                    " but n_classes does not match the value expected  ({} vs {})".format(n_classes, self.num_classes)
        assert(preds.shape[dim] == self.num_classes), error_str
        if n_classes == 1:
            return torch.sigmoid(preds).round().squeeze(dim=dim)
        else:
            return torch.softmax(preds, dim=dim).argmax(dim=dim)


class DeepLab(SegmentationModel):
    def __init__(self, num_classes, pretrained=True):
        super().__init__(num_classes, "DeepLabV3")
        ssl._create_default_https_context = ssl._create_unverified_context()
        weights = segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
        self.model = segmentation.deeplabv3_resnet50(weights=weights, num_classes=21)
        ssl._create_default_https_context = ssl.create_default_context()

    def forward(self, x):
        x = self.model(x)
        x = x['out'][:, :self.num_classes]
        return x


class UNet(SegmentationModel):
    def __init__(self, num_classes):
        super().__init__(num_classes, "unet")
        self.num_classes = num_classes
        ssl._create_default_https_context = ssl._create_unverified_context()
        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                    in_channels=3, out_channels=num_classes, init_features=32, pretrained=False)
        ssl._create_default_https_context = ssl.create_default_context()

    def forward(self, x):
        return self.model(x)


class FCN(SegmentationModel):
    def __init__(self, num_classes, pretrained=True):
        super().__init__(num_classes, "FCN-ResNet50")
        ssl._create_default_https_context = ssl._create_unverified_context()
        weights = segmentation.FCN_ResNet50_Weights.DEFAULT if pretrained else None
        self.model = segmentation.fcn_resnet50(weights=weights, num_classes=21)
        ssl._create_default_https_context = ssl.create_default_context()

    def forward(self, x):
        x = self.model(x)
        x = x['out'][:, :self.num_classes]
        return x


