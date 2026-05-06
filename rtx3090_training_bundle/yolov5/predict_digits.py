from PIL import Image
import numpy as np;
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow import keras

img_file = '/Users/wiggel/Python/eKlausurData/Temp/186440_page_4_box_7.png'
model_path = '/Users/wiggel/Python/Emnist_Model01/digit_class_hg.h5'
test_out_path='/Users/wiggel/Python/eKlausurData/Temp'
img_size = 28
model = None

def class_mapper(i):
    class_mapping = '123456789?'
    #class_mapping = '0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}'
    return class_mapping[i]


def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model


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


def run_prediction_for_image(model, img, show=False):
    img_flip = np.transpose(img, axes=[0,1,2,3])
    res_vec = model.predict(img_flip)
    result = np.argmax(res_vec)
    prop = res_vec[0,result]
    if show:
        print('Prediction: ', result, ', Char: ', class_mapper(result), ' , Prop: ', prop)
        plt.figure()
        plt.imshow(img_flip[0], cmap='gray')
        plt.colorbar()
        fname = datetime.utcnow().strftime("%Y%m%d-%H%M%S.%f")[:-3]
        plt.savefig(test_out_path + "/" + fname + '.png')
        plt.show()  #not working if called from other file
        top_values_index = sorted(range(len(res_vec[0])), key=lambda i: res_vec[0,i])[-3:]
        for i in top_values_index:
            print(class_mapper(i), ':', res_vec[0,i])
    return result, prop


def rescale(image):
    img3 = ((255-np.array(image))/255.0)[:,:,0];
    return img3;


def predict_digit(img1, show=False):
    global model
    if model is None:
        model = load_model(model_path)
    img2 = resize(img1, img_size, img_size)
    img3 = rescale(img2)
    img4 = img3.reshape(1, 28, 28, 1)
    label, prop = run_prediction_for_image(model, img4, show)
    return label, prop


def main():
    img1 = Image.open(img_file, 'r')
    predict_digit(img1, True)


if __name__ == "__main__":
    main()
