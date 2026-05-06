#
# Add labels to all unlabeld images
#
import os, shutil, glob
import sys
from subprocess import call

import torch
import tkinter as tk
import re
import random

import PIL
from PIL import ImageTk, Image
import train

'''
img_file = '/Users/wiggel/NotenCalc/187857_shift.png'
model_path = '/workspace/HOME/Python/py_yolo/yolov5/runs/train/exp17/weights/best.pt'
# yolo_path see: https://github.com/ultralytics/yolov5/discussions/5872
yolo_path = '/workspace/HOME/Python/py_yolo/yolov5'
'''

img_file = '/Users/wiggel/Python/eKlausurData/Temp/1_page_0.png'
# yolo_all:
# model_path = '/Users/wiggel/Python/py_yolo/yolov5/runs/train/good_working_exp17/weights/best.pt'
# yolo_single:
model_path = '/Users/wiggel/Python/py_yolo/yolov5/runs/train/good_working_exp17/weights/best.pt'
# yolo_single: model_path = '/Users/wiggel/Python/py_yolo/yolov5/runs/train/single_class_paper/weights/best.pt'
# yolo_path see: https://github.com/ultralytics/yolov5/discussions/5872
yolo_path = '/Users/wiggel/Python/py_yolo/yolov5'
pred_img_size = 640
model = None
agnostic = True   #No overlaying boxes -> only one box
silent = True


def load_model(model_path):
    model = torch.hub.load(yolo_path, 'custom', path=model_path, source='local')
    model.conf = 0.2
    return model


def get_cropped_image(img, x1, y1, x2, y2):
    crop_img0 = Image.open(img)
    crop_img1 = crop_img0.crop((x1, y1, x2, y2))
    crop_img2 = resize(crop_img1, 28, 28)
    return crop_img2


def save_boxes(img, results):
    box_file = os.path.splitext(img)[0];
    idx = 0
    for obj in results.pred[0]:
        ycat, x1, y1, x2, y2, conf = get_prediction_at_row(results, idx, False)
        crop_img2 = get_cropped_image(img, x1, y1, x2, y2)
        box_file0 = box_file + '_box_' + str(idx) + '.png'
        crop_img2.save(box_file0)
        idx = idx + 1


def resize(image_pil, width, height):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    ratio_w = width / image_pil.width
    ratio_h = height / image_pil.height
    if ratio_w < ratio_h:
        # It must be fixed by width
        resize_width = width
        resize_height = round(ratio_w * image_pil.height)
    else:
        # Fixed by height
        resize_width = round(ratio_h * image_pil.width)
        resize_height = height
    image_resize = image_pil.resize((resize_width, resize_height), PIL.Image.LANCZOS)
    background = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
    background.paste(image_resize, offset)
    return background.convert('RGB')


def predict(model, img_file, agnostic):
    img = img_file
    if not silent: print("Finding boxes in: ", img)
    global image_size
    img_temp = Image.open(img)
    image_size = Image.open(img).size
    results = None
    if (model != None):
        model.agnostic = agnostic
        results = model(img, size=pred_img_size)
        print(get_prediction_at_row(results, 0))
    else:
        print("No model load.")

    return results


def calc_xywhn(x1, y1, x2, y2, cls):
    global image_size
    image_width, image_height = image_size
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    x_center_image = x1 / image_width + (width / 2)
    y_center_image = y1 / image_height + (height / 2)
    return cls, x_center_image, y_center_image, width, height


def get_prediction_at_row(results, idx, yolo_format=True):
    # get image and display
    # print(results.pandas().xyxy[0])
    # From: https://donghao.org/2022/10/07/how-to-get-results-of-yolov5/
    x = results.pred[0][idx]
    x1, y1, x2, y2, conf, cat = x.numpy()
    x1, y1, x2, y2, cat = int(x1), int(y1), int(x2), int(y2), int(cat)
    if yolo_format:
        # it is ycat, xc, yc, xw, xh
        cat, x1, y1, x2, y2 = calc_xywhn(x1, y1, x2, y2, cat)
    else:
        # Nothing to do
        pass
    # if yolo-format ycat, xc, yc, xw, xh
    return cat, x1, y1, x2, y2, conf


def save_label(img, results):
    label_file = os.path.splitext(img)[0] + ".txt"
    f = open(label_file, "w")
    idx = 0
    for obj in results.pred[0]:
        ycat, ycx, ycy, yw, yh, conf = get_prediction_at_row(results, idx)
        f.write(f"{ycat} {ycx} {ycy} {yw} {yh}\n")
        print(f"{ycat} {ycx} {ycy} {yw} {yh}\n")
        idx = idx + 1
    f.close()


def get_boxes(img_file, agnostic):
    global model
    if model is None:
        model = load_model(model_path)
    results = predict(model, img_file, agnostic)
    save_boxes(img_file, results)
    return results


def main():
    results = get_boxes(img_file, agnostic)
    save_label(img_file, results)
    save_boxes(img_file, results)


if __name__ == "__main__":
    main()
