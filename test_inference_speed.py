import time

from torchstat import stat

from utils_bbox import *
from yolox import *


n = 5
k = 12
device = 'cpu'
num_cls = 1
input_shape = [640, 640]
nms_conf_threshold = 0.7
nms_iou_threshold =0.5

model = Yolox(num_cls=num_cls, training=False).to(device)
stat(model,(3,640,640))
exit(0)

model.eval()
with torch.no_grad():
    start = time.time()
    for i in range(k*n):
        x = torch.randn(1,3,640,640).to(device)
        y = model(x)
        y = decode_outputs(y, input_shape)
        y = non_max_suppression(y,
                                num_classes=num_cls,
                                input_shape=input_shape,
                                image_shape=input_shape,
                                letterbox_image=False,
                                conf_thres=nms_conf_threshold,
                                nms_thres=nms_iou_threshold)
        print(f"progress {round(i/(k*n),3)}")
    end = time.time()
print(f"device : {device}")
print(f"time {round(end-start,5)}s fps {round(k*n/(end-start),5)}")
print(f"res : {round((end-start)/n,5)}s/12frames   std : <=1s/12frames")