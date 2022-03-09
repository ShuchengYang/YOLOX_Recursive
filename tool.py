import json
import os
import random

import cv2
import numpy as np
import torch
from torchvision import transforms

from utils_bbox import *
from yolox import *

#help func, dont call
def parse_ann(ann_json, xyxy=True):
    x1 = ann_json['labels'][0]['x1']
    x2 = ann_json['labels'][0]['x2']
    y1 = ann_json['labels'][0]['y1']
    y2 = ann_json['labels'][0]['y2']

    xlim = ann_json['labels'][0]['size']['width']
    ylim = ann_json['labels'][0]['size']['height']

    xmin = max(0, x1)
    xmax = min(x2, xlim)
    ymin = max(0, y1)
    ymax = min(y2, ylim)

    xmin = 640. * xmin / xlim
    xmax = 640. * xmax / xlim
    ymin = 640. * ymin / ylim
    ymax = 640. * ymax / ylim

    if xyxy:
        return (xmin, ymin, xmax, ymax)

    xc = (xmin + xmax) / 2.
    yc = (ymin + ymax) / 2.
    bw = xmax - xmin
    bh = ymax - ymin

    return (xc, yc, bw, bh)

def read_ann(ann_path):
    with open(ann_path,'r') as f:
        ann_json = json.load(f)
        return parse_ann(ann_json)


#todo
def expand_train_dataset(config):
    current_train_size = config['train_size']
    total_img_size = config['total_size']
    img_shape = config['img_shape']

    model = Yolox(num_cls=config['num_cls'], training=False).to(config['device'])
    params = torch.load(config['params_path'])
    model.load_state_dict(params['model'])
    model.eval()
    with torch.no_grad():
        for i in range(total_img_size):
            path = os.path.join(config['untag_img_path'],str(i)+'.png')
            if os.path.exists(path):
                img_tensor = read_img(path=path,del_after=True,device=config['device'])
                res = model(img_tensor)
                res = decode_outputs(res, [img_shape, img_shape])
                results = non_max_suppression(res,
                                              num_classes=config['num_cls'],
                                              input_shape=[img_shape, img_shape],
                                              image_shape=[img_shape, img_shape],
                                              letterbox_image=False,
                                              conf_thres=config['conf'],
                                              nms_thres=config['nms']
                                              )
            else:
                print("todo in tool.py row 39")#todo


def tojson(top, left, bottom, right, save_path='test.json'):
    size = {'height':640,'width':640}
    label = {'name':'needle',
             'x1':left,
             'y1':top,
             'x2':right,
             'y2':bottom,
             'size':size}
    dict = {'labels':[label]}
    with open(save_path,'w+') as f:
        json.dump(dict, f)


def read_img(
        file=None,
        location=None,
        path=None,
        resolution=640,
        del_after=False,
        tensor=True,
        device='cuda'):
    if path is not None:
        p = path
    else:
        if file is not None:
            if location is not None:
                p = os.path.join(location, file)
            else:
                p = file
    if os.path.exists(p):
        img = cv2.imread(p)
        if del_after:
            os.remove(p)
        std_img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        return transforms.ToTensor()(std_img).unsqueeze(0).to(device) if tensor else std_img
    else:
        print(f"file in path {path} doesnt exist")
    return None

def write_img(img, file=None, location=None):
    if file is not None:
        if location is not None:
            if not os.path.exists(location):
                os.makedirs(location)
            path = os.path.join(location, file)
        else:
            path = file
        cv2.imwrite(path, img)


# help func, dont call
def iou(rec1, rec2):
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def expct_iou(results, gt_rec, input_shape=640):
    exp_iou = 0.
    if results[0] is not None:
        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = results[0][:, 4] * results[0][:, 5]
        top_boxes = results[0][:, :4]

        # plotting
        for k, c in list(enumerate(top_label)):
            top, left, bottom, right = top_boxes[k]
            score = top_conf[k]

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(input_shape, np.floor(bottom).astype('int32'))
            right = min(input_shape, np.floor(right).astype('int32'))
            iou_val = iou((left, top, right, bottom), gt_rec)
            exp_iou += score*iou_val*1.
    return exp_iou



def plot_img(img, results, input_shape=640, version_info=None, color=(0,0,0)):
    scr = 0.
    t, l, b, r = 0, 0, 0, 0

    if results[0] is not None:
        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = results[0][:, 4] * results[0][:, 5]
        top_boxes = results[0][:, :4]

        # plotting
        for k, c in list(enumerate(top_label)):
            if version_info is None:
                predicted_class = 'needle'
            else:
                predicted_class = version_info + "needle"
            top, left, bottom, right = top_boxes[k]
            score = top_conf[k]
            scr = score

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(input_shape, np.floor(bottom).astype('int32'))
            right = min(input_shape, np.floor(right).astype('int32'))
            t, l, b, r = top, left, bottom, right

            label = '{} {:.2f}'.format(predicted_class, score)
            label = label.encode('utf-8')

            text_size = cv2.getTextSize(str(label, 'UTF-8'), cv2.FONT_ITALIC, 0.5, 2)
            t_width = text_size[0][0]
            t_height = text_size[0][1]

            # color in GBR format not RGB
            # color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            cv2.rectangle(img, (left + k, top + k), (right - k, bottom - k), color, thickness=2)
            cv2.rectangle(img, (left, bottom), (left + t_width, bottom + 2 + t_height), color,
                          thickness=-1)
            cv2.putText(img=img,
                        text=str(label, 'UTF-8'),
                        org=(left + k, bottom - k + t_height),
                        fontFace=cv2.FONT_ITALIC,
                        fontScale=0.5,
                        color=(255, 255, 255),
                        thickness=1)
    return img, (scr, t, l, b, r)





if __name__ == '__main__':
    print(read_img(file='0.png', del_after=True).shape)