from head import *
from yololoss import *

class RecursiveBackbone(nn.Module):
    def __init__(self):
        super(RecursiveBackbone, self).__init__()
        self.seq1 = nn.Sequential(
            CBL(3, 32, 3, 1, 1),
            ResX(1, 32, 64, 3, 2, 1),
            ResX(2, 64, 128, 3, 2, 1),
            ResX(8, 128, 256, 3, 2, 1)
        )
        self.preseq2 = CBL(384,256,1,1,0)
        self.seq2 = ResX(8, 256, 512, 3, 2, 1)
        self.preseq3 = CBL(768,512,1,1,0)
        self.seq3 = ResX(4, 512, 1024, 3, 2, 1)

    def forward(self, x, nl, nm):
        b1 = self.seq1(x)
        o2 = self.preseq2(torch.cat([b1,nl], dim= 1))
        b2 = self.seq2(o2)
        o3 = self.preseq3(torch.cat([b2, nm], dim=1))
        b3 = self.seq3(o3)
        return [b1, b2, b3]


class RecursiveNeck(nn.Module):
    def __init__(self):
        super(RecursiveNeck, self).__init__()
        self.seq1 = CBL5(384, 128)
        self.seq2 = CBL5(768, 256)
        self.seq3 = CBL5(1024, 512)
        self.up1 = nn.Sequential(
            CBL(256, 128, 1, 1, 0),
            nn.Upsample(scale_factor=2)
        )
        self.up2 = nn.Sequential(
            CBL(512, 256, 1, 1, 0),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, backbone_output):
        n3 = self.seq3(backbone_output[2])
        nm = self.up2(n3)
        n2 = self.seq2(torch.cat([backbone_output[1], nm], dim=1))
        nl = self.up1(n2)
        n1 = self.seq1(torch.cat([backbone_output[0], nl], dim=1))
        return [n1, n2, n3], nl, nm

class RecursiveBAndN(nn.Module):
    def __init__(self):
        super(RecursiveBAndN, self).__init__()
        self.first_flag = False
        self.nl = torch.zeros(1,128,80,80)
        self.nm = torch.zeros(1,256,40,40)
        self.backbone = RecursiveBackbone()
        self.neck = RecursiveNeck()

    def forward(self,x):
        if not self.first_flag or x.shape[0] != self.nl.shape[0]:
            self.nl = torch.zeros(x.shape[0], 128, 80, 80).to(x.device)
            self.nm = torch.zeros(x.shape[0], 256, 40, 40).to(x.device)
            self.device_flag = True
        # backbone_output = self.backbone(x, self.nl, self.nm)
        backbone_output = self.backbone(x, self.nl.detach(), self.nm.detach())
        neck_output, self.nl, self.nm = self.neck(backbone_output)
        return neck_output


class Yolox(nn.Module):
    def __init__(self, num_cls, training=False):
        super().__init__()
        self.training = training
        self.backbone_neck = RecursiveBAndN()
        self.head = Head(num_classes=num_cls)
        self.criteria = YOLOLoss(num_classes=num_cls)

    def forward(self, img_tensor, target=None):
        neck_output = self.backbone_neck(img_tensor)
        head_output = self.head(neck_output)
        if self.training:
            return self.criteria(head_output, labels=target)
        else:
            return head_output

