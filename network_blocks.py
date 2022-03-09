from torch import nn


class CBL(nn.Module):
    def __init__(self, *args):
        """args are for convolution step"""
        super(CBL, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(*args),
            nn.BatchNorm2d(args[1]),
            nn.LeakyReLU(negative_slope=1e-1)
        )

    def forward(self, x):
        return self.seq(x)


class ResUnit(nn.Module):
    def __init__(self, *args):
        """args[0] is num of in channels"""
        super(ResUnit, self).__init__()
        self.seq = nn.Sequential(
            CBL(args[0], args[0]//2, 1, 1, 0),
            CBL(args[0]//2, args[0], 3, 1, 1)
        )

    def forward(self, x):
        return x+self.seq(x)


class ResX(nn.Module):
    def __init__(self, n, *args):
        """n is for the number of res unit
           args and kwargs are for the down sampling cbl"""
        super(ResX, self).__init__()
        self.seq = nn.Sequential(
            CBL(*args),
        )
        for i in range(n):
            self.seq.add_module("res unit", ResUnit(args[1]))

    def forward(self, x):
        return self.seq(x)


class CBL5(nn.Module):
    def __init__(self, *args):
        """
        args[0] is num of input channels,
        args[1] is of output channels
        """
        super(CBL5, self).__init__()
        out2 = int(2*args[1])
        self.seq = nn.Sequential(
            CBL(args[0], args[1], 1, 1, 0),
            CBL(args[1], out2, 3, 1, 1),
            CBL(out2, args[1], 1, 1, 0),
            CBL(args[1], out2, 3, 1, 1),
            CBL(out2, args[1], 1, 1, 0)
        )

    def forward(self, x):
        return self.seq(x)


class DHFirst(nn.Module):
    def __init__(self, *args):
        """
        args[0] is the input channels
        args[1] is the number of classes (abandoned)
        output all without sigmoid (will be added in train or inference stage)
        """
        super(DHFirst, self).__init__()
        self.stem = CBL(args[0], 256, 1, 1, 0)
        self.box_conf_stem = nn.Sequential(
            CBL(256, 256, 3, 1, 1),
            CBL(256, 256, 3, 1, 1)
        )
        self.cls_conv = nn.Sequential(
            CBL(256, 256, 3, 1, 1),
            CBL(256, 256, 3, 1, 1),
        )


    def forward(self, neck_output):
        """neck_output is only an element of tensor list neck_outputs
        (batch, c, h, w)
        [(1,128,80,80), (1,256,40,40), (1,512,20,20)]
        """
        neck_output = self.stem(neck_output)
        cls_raw = self.cls_conv(neck_output)
        box_conf_raw = self.box_conf_stem(neck_output)
        return box_conf_raw, cls_raw