# coding = utf8
# labelsvehicle-13998a6ad038-360180A00068975-CronImage-1669110350124-651292-w-1280-h-720
# python coco2yolo5.py --annotation_path instances_aopengkeji.json --save_base_path 360_vehicle/labels/train


import argparse
import os

import cv2
import numpy as np
import tqdm
from pycocotools.coco import COCO

image_dir = "d:\\save"
image_save_dir = "yolov5_360_vehicle\\images\\train"

default_annotation_path = "instances_aopengkeji5.json"
default_save_base_path = "yolov5_360_vehicle\\labels\\train"

def arg_parser():
    parser = argparse.ArgumentParser('code by rbj')
    parser.add_argument('--annotation_path', type=str,
                        default=default_annotation_path)
    #生成的txt文件保存的目录
    parser.add_argument('--save_base_path', type=str, default=default_save_base_path)
    args = parser.parse_args()
    #原网页中是args = parser.parse_args()会报错，改成这个以后解决了
    return args
if __name__ == '__main__':
    args = arg_parser()
    annotation_path = args.annotation_path
    save_base_path = args.save_base_path

    data_source = COCO(annotation_file=annotation_path)
    catIds = data_source.getCatIds()
    categories = data_source.loadCats(catIds)
    categories.sort(key=lambda x: x['id'])
    classes = {}
    coco_labels = {}
    coco_labels_inverse = {}
    for c in categories:
        coco_labels[len(classes)] = c['id']
        coco_labels_inverse[c['id']] = len(classes)
        classes[c['name']] = len(classes)

    img_ids = data_source.getImgIds()
    for index, img_id in tqdm.tqdm(enumerate(img_ids), desc='change .json file to .txt file'):
        img_info = data_source.loadImgs(img_id)[0]
        file_name = img_info['file_name'].split('.')[0]
        height = img_info['height']
        width = img_info['width']

        image_path = os.path.join(image_dir, img_info['file_name']) 
        image = cv2.imread(image_path)
        
        save_path = os.path.join(save_base_path, file_name + '.txt')
        with open(save_path, mode='w') as fp:
            annotation_id = data_source.getAnnIds(img_id)
            boxes = np.zeros((0, 5))
            if len(annotation_id) == 0:
                continue
            annotations = data_source.loadAnns(annotation_id)
            lines = ''
            for annotation in annotations:
                box = annotation['bbox']
                pts = annotation['segmentation']
                if len(pts) != 0:
                    pts = np.array(pts, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(image, [pts], color=(114, 114, 114))
                    continue
                # some annotations have basically no width / height, skip them
                if box[2] < 1 or box[3] < 1:
                    continue
                #top_x,top_y,width,height---->cen_x,cen_y,width,height
                box[0] = round((box[0] + box[2] / 2) / width, 6)
                box[1] = round((box[1] + box[3] / 2) / height, 6)
                box[2] = round(box[2] / width, 6)
                box[3] = round(box[3] / height, 6)
                label = coco_labels_inverse[annotation['category_id']]
                # lines = lines + str(label)
                lines = lines + '0'
                for i in box:
                    lines += ' ' + str(i)
                lines += '\n'
            
            save_image_path = os.path.join(image_save_dir, file_name + '.jpg')
            if lines != '':
                cv2.imwrite(save_image_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                fp.writelines(lines)
    print('finish')
