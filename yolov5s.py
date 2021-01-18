from common import *

class My_YOLO_backbone_head(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq0_Focus = Focus(3, 32, 3)
        self.seq1_Conv = Conv(32, 64, 3, 2)
        self.seq2_C3 = C3(64, 64, 1)
        self.seq3_Conv = Conv(64, 128, 3, 2)
        self.seq4_C3 = C3(128, 128, 3)
        self.seq5_Conv = Conv(128, 256, 3, 2)
        self.seq6_C3 = C3(256, 256, 3)
        self.seq7_Conv = Conv(256, 512, 3, 2)
        self.seq8_SPP = SPP(512, 512, [5, 9, 13])
        self.seq9_C3 = C3(512, 512, 1, False)
        self.seq10_Conv = Conv(512, 256, 1, 1)
        self.seq13_C3 = C3(512, 256, 1, False)
        self.seq14_Conv = Conv(256, 128, 1, 1)
        self.seq17_C3 = C3(256, 128, 1, False)
        self.seq18_Conv = Conv(128, 128, 3, 2)
        self.seq20_C3 = C3(256, 256, 1, False)
        self.seq21_Conv = Conv(256, 256, 3, 2)
        self.seq23_C3 = C3(512, 512, 1, False)
    def forward(self, x):
        x = self.seq0_Focus(x)
        x = self.seq1_Conv(x)
        x = self.seq2_C3(x)
        x = self.seq3_Conv(x)
        xRt0 = self.seq4_C3(x)
        x = self.seq5_Conv(xRt0)
        xRt1 = self.seq6_C3(x)
        x = self.seq7_Conv(xRt1)
        x = self.seq8_SPP(x)
        x = self.seq9_C3(x)
        xRt2 = self.seq10_Conv(x)
        route = F.interpolate(xRt2, size=(int(xRt2.shape[2] * 2), int(xRt2.shape[3] * 2)), mode='nearest')
        x = torch.cat([route, xRt1], dim=1)
        x = self.seq13_C3(x)
        xRt3 = self.seq14_Conv(x)
        route = F.interpolate(xRt3, size=(int(xRt3.shape[2] * 2), int(xRt3.shape[3] * 2)), mode='nearest')
        x = torch.cat([route, xRt0], dim=1)
        out0 = self.seq17_C3(x)
        x = self.seq18_Conv(out0)
        x = torch.cat([x, xRt3], dim=1)
        out1 = self.seq20_C3(x)
        x = self.seq21_Conv(out1)
        x = torch.cat([x, xRt2], dim=1)
        out2 = self.seq23_C3(x)
        return out0, out1, out2

class My_YOLO(nn.Module):
    def __init__(self, num_classes, anchors=(), training=False):
        super().__init__()
        self.backbone_head = My_YOLO_backbone_head()
        self.yolo_layers = Yolo_Layers(nc=num_classes, anchors=anchors, ch=(128,256,512),training=training)
    def forward(self, x):
        out0, out1, out2 = self.backbone_head(x)
        output = self.yolo_layers([out0, out1, out2])
        return output
