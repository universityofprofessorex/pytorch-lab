import torchvision.models as models
import torch.nn as nn
import devices
import argparse
import timm

MODEL_NAMES = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


class ObjLocModel(nn.Module):
    def __init__(self):
        super(ObjLocModel, self).__init__()
        # super().__init__()

        # device = devices.get_optimal_device(args)

        # weights = models.__dict__[args.model_weights].DEFAULT
        # auto_transforms = weights.transforms()
        # model = models.__dict__[args.arch](weights=weights).to(device)
        # model.name = args.arch

        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=4
        )
        # self.backbone = model

    # def init_weights(self, args: argparse.Namespace):
    #     self.backbone = timm.create_model(args.arch, pretrained=True, num_classes=4)

    def forward(self, images, gt_bboxes=None):
        bboxes_logits = self.backbone(images)  ## predicted bounding boxes

        # gt_bboxes = ground truth bounding boxes
        if gt_bboxes != None:
            loss = nn.MSELoss()(bboxes_logits, gt_bboxes)
            return bboxes_logits, loss

        return bboxes_logits
