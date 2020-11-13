import pytorch_lightning as pl
from torchvision import models


class MobileNetLightingModel(pl.LightningModule):
    """https://pytorch.org/hub/pytorch_vision_mobilenet_v2/"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        print('5')
