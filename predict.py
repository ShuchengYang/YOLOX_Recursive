import os
import time
from random import randint

import cv2
from torchvision import transforms
from yolox import *
from utils_bbox import *

#TODO UPDATE:
version_info = '[v3.1]'
pre_path = 'E:/proj/'

infer_src_img_path = pre_path + 'needleV2/imgs'
plot_src_img_path = pre_path + 'needleV2/imgs'

res_img_saving_path = pre_path + 'needleV2/' + version_info + 'track'
#TODO ATTENTION:
model_param_path = pre_path + 'cp/[V3.0-Recursive-test]Ep-47-NewMin.tar'

input_shape = 640
device = 'cuda'
num_cls = 1
nms_conf_threshold = 0.7
nms_iou_threshold = 0.5
color = (randint(0,255), randint(0,255), randint(0,255))

#preperation for model param
checkpoint = torch.load(model_param_path)
#count img ttl num
imgs_folder = os.listdir(infer_src_img_path)
num_imgs = len(imgs_folder)


# build up model
model = Yolox(num_cls=num_cls, training=False)
#load model param
model.load_state_dict(checkpoint['model'])
# move model to device
model.to(device)
#switch to evaluation mode
model.eval()
with torch.no_grad():
    start = time.time()
    for i_img in range(num_imgs):
        #cvimg is img used to predict
        cvimg = cv2.imread(infer_src_img_path + '/' + str(i_img) + '.png')
        #pimg is img used to visualize result
        pimg = cv2.imread(plot_src_img_path + '/' + str(i_img) + '.png')

        std_cvimg = cv2.resize(cvimg, (input_shape, input_shape), interpolation=cv2.INTER_CUBIC)
        std_pimg = cv2.resize(pimg, (input_shape, input_shape), interpolation=cv2.INTER_CUBIC)

        cvimg_tensor = transforms.ToTensor()(std_cvimg).unsqueeze(0).to(device)
        cvpreds = model(cvimg_tensor)
        cvpreds = decode_outputs(cvpreds, [input_shape, input_shape])
        cvresults = non_max_suppression(cvpreds,
                                        num_classes=num_cls,
                                        input_shape=[input_shape, input_shape],
                                        image_shape=[input_shape, input_shape],
                                        letterbox_image=False,
                                        conf_thres=nms_conf_threshold,
                                        nms_thres=nms_iou_threshold)
        if cvresults[0] is not None:
            top_label = np.array(cvresults[0][:, 6], dtype='int32')
            top_conf = cvresults[0][:, 4] * cvresults[0][:, 5]
            top_boxes = cvresults[0][:, :4]

            #plotting
            for k, c in list(enumerate(top_label)):
                predicted_class = version_info+"needle"
                top, left, bottom, right = top_boxes[k]
                score = top_conf[k]

                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(input_shape, np.floor(bottom).astype('int32'))
                right = min(input_shape, np.floor(right).astype('int32'))

                label = '{} {:.2f}'.format(predicted_class, score)
                label = label.encode('utf-8')
                print(round(i_img / num_imgs, 3), str(label), top, left, bottom, right)
                text_size = cv2.getTextSize(str(label, 'UTF-8'), cv2.FONT_ITALIC, 0.5, 2)
                t_width = text_size[0][0]
                t_height = text_size[0][1]

                # color in GBR format not RGB
                cv2.rectangle(std_pimg, (left + k, top + k), (right - k, bottom - k), color=color, thickness=2)
                cv2.rectangle(std_pimg, (left, bottom), (left + t_width, bottom + 2 + t_height), color=color,
                              thickness=-1)
                cv2.putText(img=std_pimg,
                            text=str(label, 'UTF-8'),
                            org=(left + k, bottom - k + t_height),
                            fontFace=cv2.FONT_ITALIC,
                            fontScale=0.5,
                            color=(255, 255, 255),
                            thickness=1)
            if not os.path.exists(res_img_saving_path):
                os.makedirs(res_img_saving_path)
            cv2.imwrite(res_img_saving_path + '/' + str(i_img) + '.png', std_pimg)
            # cv2.imshow('needle tracking',std_cvimg)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            if i_img == 10:
                exit(233)
        else:
            print("nothing found")

    end = time.time()
    print("predicting {} pics takes {}s, fps: {}".format(num_imgs, round(end - start, 2),
                                                         round(num_imgs / (end - start), 2)))
