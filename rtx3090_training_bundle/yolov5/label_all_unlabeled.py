#
# Add labels to all unlabeld images
#
import os, shutil, glob
import sys
from subprocess import call
from typing import List
from natsort import os_sorted

import torch
import tkinter as tk
import re
import random

from PIL import ImageTk, Image
import train


yolo_path = '/workspace/HOME/Python/py_yolo/yolov5'

#label_img_path  = '/workspace/SSD/eKlausurData/YoloMultiClassGenerated'
#label_img_path  = '/Users/wiggel/Python/eKlausurData/YoloMultiClassGenerated'
# label_img_path  = '/Users/wiggel/Python/eKlausurData/Temp'
# label_img_path  = '/Users/wiggel/Python/eKlausurData/YoloMultiClassGenerated'
label_img_path  = '/Users/wiggel/Python/eKlausurData/TestSmallMovements'
#label_img_path  = '/workspace/SSD/eKlausurData/YoloSingleClassByHand'
#label_img_path  = '/Users/wiggel/Python/eKlausurData/YoloDigits'
yolo_path = '/Users/wiggel/Python/py_yolo/yolov5'

learn_data_path = label_img_path
extensions_for_testing = ['.png', '.jpg']
extension_for_learning = '.png'
pred_img_size = 640
model = None
pointer = 0

#
# set deterministic to warn in general.py
# torch.use_deterministic_algorithms(True, warn_only=True)
# see https://github.com/ultralytics/yolov5/pull/8213
#
def train_model():
    copy_training_data()
    # train.run(data='dataset_hg_digits.yaml', imgsz=128, weights='yolov5s.pt', hyp='hyp_hg.yaml', epochs=200, batch=-1)
    train.run(data='dataset_hg_singleclass.yaml', imgsz=640, weights='yolov5s.pt', hyp='hyp_hg_table.yaml', epochs=20, batch=-1)


def load_latest_model(search_dir='./runs'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/best*.pt', recursive=True)
    model_path = max(last_list, key=os.path.getctime) if last_list else ''
    model_path = '/Users/wiggel/Python/py_yolo/yolov5/runs/train/exp10/weights/best.pt'
    print(model_path)
    if (model_path == ''):
        print('No model found ...')
        global model
        model = None
    else:
        load_model(model_path)


def load_model(model_path):
    global model
    model = torch.hub.load('.', 'custom', path=model_path, source='local')
    model.conf = 0.2
    model.agnostic = True
    # more parameters, see
    # https://github.com/ultralytics/yolov5/issues/36


def save_boxes():
    img = label_img_path + "/" + img_files[pointer]
    box_file = os.path.splitext(img)[0];
    idx = 0
    for obj in results.pred[0]:
        ycat, x1, y1, x2, y2 = get_prediction_at_row(results, idx, False)
        crop_img0 = Image.open(img)

        crop_img1 = crop_img0.crop( (x1,y1,x2,y2) )
        box_file0 = box_file + '_box_' + str(idx) + '.png'
        crop_img1.save(box_file0)
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
    image_resize = image_pil.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
    background.paste(image_resize, offset)
    return background.convert('RGB')

def load_model(model_path):
    global model
    model = torch.hub.load('.', 'custom', path=model_path, source='local')
    model.conf = 0.2
    model.agnostic = False #No overlaying boxes -> only one box
    # more parameters, see
    # https://github.com/ultralytics/yolov5/issues/36

def init_window():
    global root
    root = tk.Tk()
    frame = tk.Frame(root)
    frame.pack()
    global image_label
    image_label = tk.Label(frame)
    image_label.pack(side=tk.LEFT)
    button = tk.Button(frame,
                       text="QUIT",
                       fg="red",
                       command=quit)
    button.pack(side=tk.BOTTOM)
    next10 = tk.Button(frame,
                     text="Next10",
                     command= lambda: next_image(13) )
    next10.pack(side=tk.BOTTOM)
    prev10 = tk.Button(frame,
                     text="Prev10",
                     command= lambda: prev_image(13))
    prev10.pack(side=tk.BOTTOM)
    next = tk.Button(frame,
                     text="Next",
                     command=next_image)
    next.pack(side=tk.BOTTOM)
    prev = tk.Button(frame,
                     text="Prev",
                     command=prev_image)
    prev.pack(side=tk.BOTTOM)
    take = tk.Button(frame,
                     text="Take",
                     command=lambda: save_label())
    take.pack(side=tk.BOTTOM)
    take = tk.Button(frame,
                     text="Train",
                     command=train_model)
    take.pack(side=tk.BOTTOM)
    take = tk.Button(frame,
                     text="Re-Load",
                     command=load_latest_model)
    take.pack(side=tk.BOTTOM)
    take = tk.Button(frame,
                     text="SaveBox",
                     command=save_boxes)
    take.pack(side=tk.BOTTOM)


def next_image(step=1):
    global pointer
    pointer = pointer + step
    show_image()


def prev_image(step=1):
    global pointer
    pointer = pointer - step
    show_image()


def show_image():
    global pointer
    img = label_img_path + "/" + img_files[pointer]
    root.title('Image: ' + str(pointer) + '  ' + img)
    # print("Image: ", img)
    global image_size
    img_temp = Image.open(img)
    image_size = Image.open(img).size
    global results

    if (model != None):
        results = model(img, size=pred_img_size)
        print(get_prediction_at_row(results, 0))
        results.render()
        image = ImageTk.PhotoImage(image=Image.fromarray(results.ims[0]))
    else:
        image = ImageTk.PhotoImage(image=img_temp)

    image_label.configure(image=image)
    image_label.image = image





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
    print(results.pandas().xyxy[0])
    # From: https://donghao.org/2022/10/07/how-to-get-results-of-yolov5/
    x = results.pred[0]
    print (x.shape[0])
    if x.shape[0] == 0: #No results
    	return 0,0,0,0,0
    x = results.pred[0][idx]
    x1, y1, x2, y2, conf, cat = x.cpu().numpy()
    x1, y1, x2, y2, cat = int(x1), int(y1), int(x2), int(y2), int(cat)
    if yolo_format:
        #it is ycat, xc, yc, xw, xh
        cat, x1, y1, x2, y2 = calc_xywhn(x1, y1, x2, y2, cat)
    else:
        # Nothing to do
        pass
    # if yolo-format ycat, xc, yc, xw, xh
    return cat, x1, y1, x2, y2


def save_label():
    img = label_img_path + "/" + img_files[pointer]
    label_file = os.path.splitext(img)[0] + ".txt"
    f = open(label_file, "w")
    idx = 0
    for obj in results.pred[0]:
        ycat, ycx, ycy, yw, yh = get_prediction_at_row(results, idx)
        print(label_file)
        f.write(f"{ycat} {ycx} {ycy} {yw} {yh}\n")
        idx = idx + 1
    f.close()
    next_image()


def getint(name):
    num, ext = name.split('.')
    try:
        return int(num)
    except:
        return num


def gen_list_of_files():
    files = []
    for ext in extensions_for_testing:
        list = glob.glob1(label_img_path, '*' + ext)
        files = files + list
    sorted_list = (os_sorted(files))
    # img_files = list(filter(lambda f: f.endswith(extension_image), sorted_list))
    img_files = sorted_list
    print(img_files)
    return img_files


def copy_training_data():
    # copy data (image and label) from testdata
    # to testdata/images and testdata/labels
    # generate train and validation data inside
    # copy only, if ".txt" is available
    extension_label = ".txt"
    split_percentage = 90

    if not os.path.exists(learn_data_path):
        os.mkdir(learn_data_path)

    images_path = learn_data_path + '/images'
    if os.path.exists(images_path):
        shutil.rmtree(images_path)
    os.mkdir(images_path)

    labels_path = learn_data_path + '/labels'
    if os.path.exists(labels_path):
        shutil.rmtree(labels_path)
    os.mkdir(labels_path)

    training_images_path = images_path + '/training'
    validation_images_path = images_path + '/validation'
    training_labels_path = labels_path + '/training'
    validation_labels_path = labels_path + '/validation'

    os.mkdir(training_images_path)
    os.mkdir(validation_images_path)
    os.mkdir(training_labels_path)
    os.mkdir(validation_labels_path)

    files = []
    ext_len = len(extension_label)
    for r, d, f in os.walk(learn_data_path):
        for file in f:
            if file.endswith(extension_label):
                strip = file[0:len(file) - ext_len]
                if os.path.exists(learn_data_path + "/" + strip + extension_for_learning):
                    files.append(strip)
    random.shuffle(files)
    size = len(files)
    split = int(split_percentage * size / 100)

    print("copying training_data")
    for i in range(split):
        strip = files[i]

        image_file = strip + extension_for_learning
        src_image = learn_data_path + "/" + image_file
        shutil.copy(src_image, training_images_path)

        annotation_file = strip + '.txt'
        src_label = learn_data_path + "/" + annotation_file
        shutil.copy(src_label, training_labels_path)

    print("copying validation_data")
    for i in range(split, size):
        strip = files[i]

        image_file = strip + extension_for_learning
        src_image = learn_data_path + "/" + image_file
        shutil.copy(src_image, validation_images_path)

        annotation_file = strip + '.txt'
        src_label = learn_data_path + "/" + annotation_file
        shutil.copy(src_label, validation_labels_path)

    print("finished")


def init_classes():
    classes_file = label_img_path + "/classes.txt"
    if not os.path.exists(classes_file):
        with open(classes_file, "w") as f:
            f.write("SC\n")
        print("Created classes.txt file.")
    else:
        print("classes.txt file already exists.")

def main():
    init_window()
    global img_files
    global pointer
    pointer = 0
    init_classes()
    img_files = gen_list_of_files()
    load_latest_model()
    show_image()
    root.mainloop()


if __name__ == "__main__":
    main()
