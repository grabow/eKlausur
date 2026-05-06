import io

import imageio
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import matplotlib.pyplot as plt
from skimage.util import random_noise
from PIL import Image
from random import random, randint, uniform

base_dir = '/workspace/SSD'
base_dir = '/Users/wiggel/Python'
class_mapping = 'ABCDEFGHIJKLMNOPQRSTUVWXYZf!'
# Emnist data must be transposed in line 37
# class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
dir_pages = base_dir + "/eKlausurData/EmptyTables"
digit_dir = base_dir + '/eKlausurData/hg_letters'
out_dir = base_dir + '/eKlausurData/YoloMultiClassGenerated'
data_filepath = base_dir + '/eKlausurData/input/emnist_hg.csv'
mbase_y = 240


def load_emnist_data():
    training_letter = pd.read_csv(data_filepath, header=None)
    # training_letter = pd.read_csv('/workspace/SSD/eKlausurData/input/emnist-balanced-test.csv', header=None)
    print(training_letter.shape)
    y1 = np.array(training_letter.iloc[:, 0].values)
    x1 = np.array(training_letter.iloc[:, 1:].values)
    print(y1.shape)
    print(x1.shape)
    # Transform all data from (x1,y1) to transposed numpy (28,28,1) 0..255
    data_rows = len(y1)
    num_of_classes = len(np.unique(y1))
    imgs_ori = x1.reshape(data_rows, 28, 28, 1)
    # X = np.transpose(imgs_ori, axes=(0, 2, 1, 3))
    X = imgs_ori
    return X, y1, data_rows


def gen_streched_rot_erode_image(pic):
    # select image
    # print(class_mapping[label])

    # convert numpy to image
    img_flat = pic.reshape(28, 28)
    gray_image = Image.fromarray(img_flat.astype('uint8'), mode='L')

    # generate stretched images for arbitrary bounding boxes
    '''
    x_max = 1.5
    x_min = 0.9
    y_max = 2
    y_min = 0.9
    '''
    x_max = 1.0
    x_min = 1.0
    y_max = 1.0
    y_min = 1.0
    base_size = 38
    value = random()
    value_x = x_min + (value * (x_max - x_min))
    value_x_int = int(base_size * value_x)
    value = random()
    value_y = y_min + (value * (y_max - y_min))
    value_y_int = int(base_size * value_y)
    gray_image2 = gray_image.resize((value_x_int, value_y_int), Image.BILINEAR)
    gray_image3 = np.array(gray_image2).reshape(value_y_int, value_x_int)

    # erode image
    erode_factor = randint(1, 2)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(gray_image3, kernel, iterations=erode_factor)
    np_img = np.array(img)
    sum = np_img.sum()
    if sum < 4000:
        img = gray_image3

    # rotate image
    angle = randint(-10, 10)
    rot_img = np.array(rotate_bound(img, angle, 0))

    return rot_img


def open_rotate_resize_background():
    p_nr = randint(0, 3)
    filename = "page_" + str(p_nr) + ".jpg"
    filepath = os.path.join(dir_pages, filename)
    print(filepath)
    page = Image.open(filepath).convert('L')
    degree = randint(-2, 0)
    page = page.rotate(degree)
    factor = 640 / page.size[0]
    page = page.resize((640, int(page.size[1] * factor)), Image.BILINEAR)
    # page = page.resize((640, 420), Image.BILINEAR)
    page_np = 255 - np.array(page)
    return page_np


def gen_image_to_add(img, page, off_base_x, off_base_y):
    v_up = randint(-8, 8)
    v_lr = randint(-8, 8)
    offsetx = off_base_x + v_lr
    offsety = off_base_y + v_up
    black_img = np.zeros((page.shape[0], page.shape[1]), dtype=np.uint8)
    x1 = offsetx
    x2 = offsetx + img.shape[1]
    y1 = offsety
    y2 = offsety + img.shape[0]

    if y2 > black_img.shape[0]:
        diffy = y2 - black_img.shape[0]
        y1 = y1 - diffy
        y2 = y2 - diffy

    if x2 > black_img.shape[1]:
        diffx = x2 - black_img.shape[1]
        x1 = x1 - diffx
        x2 = x2 - diffx

    black_img[y1:y2, x1:x2] = img
    return black_img, x1, x2, y1, y2


def add_image(img, page):
    Z = np.stack((img, page))
    Zmax = Z.max(0)
    return Zmax


def write_label_info(black_img, label, x1, x2, y1, y2):
    # Write boxes for yolov5
    x_max = black_img.shape[1]
    y_max = black_img.shape[0]
    class_num = label
    x_center = (x1 + (x2 - x1) / 2) / x_max
    y_center = (y1 + (y2 - y1) / 2) / y_max
    width = (x2 - x1) / x_max  # add 0 pixels as boundary
    height = (y2 - y1) / y_max  # add 0 pixels as boundary
    erg = str(class_num) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height)
    # print(erg)
    return erg


def rotate_bound(image, angle, fill=255):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=(fill))


def add_gaussian_noise(image, mean=0, sigma=20):
    """Add Gaussian noise to an image of type np.uint8."""
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    gaussian_noise = gaussian_noise.reshape(image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    # show_numpy(noisy_image)
    return noisy_image


def add_s_u_p(image, amount=0.1):
    # Add salt-and-pepper noise to the image.
    noise_img = random_noise(image, mode='s&p', amount=amount)
    noise_img = np.array(255 * noise_img, dtype='uint8')
    return noise_img


# Image is 2D numpy array, q is quality 0-100
def add_jpgblurr(im, q):
    buf = io.BytesIO()
    imageio.imwrite(buf, im, format='jpg', quality=q)
    s = buf.getbuffer()
    return imageio.imread(s, format='jpg')


def load_digit_rot_and_resize():
    #size_x1 = randint(11, 15)  # x=13
    #size_y1 = randint(16, 20)  # y=18
    #blurr = randint(5, 10)
    factor1 = uniform(0.4, 0.6) # about 05
    factor2 = uniform(0.4, 0.6)
    angle = randint(-5, 5)
    # nr = randint(0, 89)
    nr = randint(0, 8)
    filename = 'digit_' + str(nr+1) + ".png"
    filepath = os.path.join(digit_dir, filename)
    digit = Image.open(filepath).convert('L')
    digit0 = np.array(digit)
    digit1 = rotate_bound(digit0, angle)
    digit2 = Image.fromarray(digit1)
    w = digit2.size[0]
    h = digit2.size[1]
    size_x = int(w * factor1)
    size_y = int(h * factor2)
    digti3 = digit2.resize((size_x, size_y), Image.BILINEAR)
    # digti3 = digit2.resize((size_x1, size_y1), Image.BILINEAR)
    digit4 = np.array(digti3)
    # digit5 = 255 - add_gaussian_noise(digit4)
    # digit5 = 255 - add_jpgblurr(digit4, blurr)
    digit5 = 255 - digit4
    # class_nr = int(nr / 10)
    class_nr = int(nr)
    label = class_nr + len(class_mapping)
    # show_numpy(digit5)
    return digit5, label


def show_numpy(p):
    plt.imshow(p, cmap='Greys_r')
    plt.colorbar()
    plt.show()


def add_emnist_to_background_at_pos(back_page, img_nr, X, y, pos_x, pos_y):
    pic = X[img_nr]
    label = y[img_nr]
    img = gen_streched_rot_erode_image(pic)
    img2, x1, x2, y1, y2 = gen_image_to_add(img, back_page, pos_x, pos_y)
    img3 = add_image(back_page, img2)
    erg = write_label_info(back_page, label, x1, x2, y1, y2)
    return img3, erg


def add_nine_emnist_from(back_page, X, y, start):
    label = ""
    base_x = 40
    base_y = mbase_y + 70
    step_x = 60
    base_y_below = base_y + 75
    base_y_above = base_y - 100
    for i in range(9):
        pos_x = base_x + step_x * i
        pos_y = base_y
        back_page, erg = add_emnist_to_background_at_pos(back_page, i + start, X, y, pos_x, pos_y)
        label = label + erg + "\n"

    # Add char below table
    for i in range(0, 9, 2):
        pos_x = base_x + step_x * i
        pos_y = base_y_below
        back_page, erg = add_emnist_to_background_at_pos(back_page, 9 + start - i, X, y, pos_x, pos_y)
        label = label + erg + "\n"

    # Add char above table
    for i in range(1, 9, 2):
        pos_x = base_x + step_x * i
        pos_y = base_y_above
        back_page, erg = add_emnist_to_background_at_pos(back_page, 9 + start - i, X, y, pos_x, pos_y)
        label = label + erg + "\n"

    # show_numpy(back_page)
    return back_page, label


def add_digit_to_background_at_pos(back_page, pos_x, pos_y):
    img, label = load_digit_rot_and_resize()
    img2, x1, x2, y1, y2 = gen_image_to_add(img, back_page, pos_x, pos_y)
    img3 = add_image(back_page, img2)
    erg = write_label_info(back_page, label, x1, x2, y1, y2)
    return img3, erg


def add_digit_to_background(back_page):
    base_x = 40
    base_y = mbase_y + 40
    step_x = 60
    steps = randint(0, 8)
    pos_x = base_x + steps * step_x
    img, erg = add_digit_to_background_at_pos(back_page, pos_x, base_y)
    return img, erg


def gen_samples(prefix, X, y, data_rows):
    end_range = data_rows - 9
    # end_range = 72
    for nr in range(0, end_range, 9):
        print(nr)
        back_page = open_rotate_resize_background()
        page, l1 = add_nine_emnist_from(back_page, X, y, nr)
        page, l2 = add_digit_to_background(page)
        erg = l1 + l2
        filename1 = prefix + '_page_' + str(nr) + ".png"
        filename2 = prefix + '_page_' + str(nr) + ".txt"
        filepath1 = os.path.join(out_dir, filename1)
        filepath2 = os.path.join(out_dir, filename2)
        im = Image.fromarray(page)
        im.save(filepath1)
        with open(filepath2, 'w') as f:
            f.write(erg)


def main():
    print("Hallo")
    X, y, data_rows = load_emnist_data()
    gen_samples("1", X, y, data_rows)
    # gen_samples("2", X, y, data_rows)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
