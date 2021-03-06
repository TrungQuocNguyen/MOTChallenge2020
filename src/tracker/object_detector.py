import torch
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class FRCNN_FPN(FasterRCNN):

    def __init__(self, num_classes, nms_thresh=0.5):
        backbone = resnet_fpn_backbone('resnet50', False) # do not load pretrained backbone on ImageNet
        super(FRCNN_FPN, self).__init__(backbone, num_classes)

        self.roi_heads.nms_thresh = nms_thresh

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]
        return detections['boxes'].detach().cpu(), detections['scores'].detach().cpu()

    #def detect(self, img):
    #    device = list(self.parameters())[0].device
    #    img = img.to(device)

    #    detections = self(img)
    #    list_boxes = [f['boxes'].detach().cpu() for f in detections]
    #    list_scores = [f['scores'].detach().cpu() for f in detections]
    #    return list_boxes, list_scores