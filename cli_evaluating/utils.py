import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import random
import cv2
import os
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import *
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import tensorflow_hub as hub
from sklearn import metrics
from PIL import Image
from glob import glob
import pickle as pk
from numpy.random import rand

random.seed(5)

import cv2

def resize_image(image, new_size):
    dimensions = None
    target_height, target_width = new_size
    image_height, image_width, _ = image.shape
    
    # If neither width nor height is provided, return the original image
    if target_width is None and target_height is None:
        return image
    
    # If both height and width are provided, resize without maintaining aspect ratio
    if target_width is not None and target_height is not None:
        resized = cv2.resize(image, (target_width, target_height))
        return resized
    
    if target_width is None:
        ratio = target_height / float(image_height)
        dimensions = (int(image_width * ratio), target_height)
    else:
        ratio = target_width / float(image_width)
        dimensions = (target_width, int(image_height * ratio))
    
    resized = cv2.resize(image, dimensions)
    return resized


def resize_with_max(image, max_size, apply_to_all=False):
    image_height, image_width, _ = image.shape
    
    if apply_to_all:
        if image_height > image_width:
            image = resize_image(image, new_size=(max_size[0], None))
        elif image_width > image_height:
            image = resize_image(image, new_size=(None, max_size[1]))
        else:
            image = cv2.resize(image, max_size)
    elif max((image_height, image_width)) > max_size[0]:
        if image_height > image_width:
            image = resize_image(image, new_size=(max_size[0], None))
        elif image_width > image_height:
            image = resize_image(image, new_size=(None, max_size[1]))
        else:
            image = cv2.resize(image, max_size)
    
    return image

def add_border_resize(image, size):
    height, width, _ = image.shape
    
    if height > width:
        resized_image = resize_image(image, size=(size[0], None))
        border_size_left = int((size[1] - resized_image.shape[1]) / 2)
        border_size_right = size[1] - (border_size_left + resized_image.shape[1])
        bordered_image = cv2.copyMakeBorder(resized_image, top=0, bottom=0,
                                            left=border_size_left, right=border_size_right,
                                            borderType=cv2.BORDER_CONSTANT, value=0)
    else:
        resized_image = resize_image(image, size=(None, size[1]))
        border_size_top = int((size[0] - resized_image.shape[0]) / 2)
        border_size_bottom = size[0] - (border_size_top + resized_image.shape[0])
        bordered_image = cv2.copyMakeBorder(resized_image, top=border_size_top, bottom=border_size_bottom,
                                            left=0, right=0,
                                            borderType=cv2.BORDER_CONSTANT, value=0)
    
    return bordered_image


def build_multi_gap_inception_resnet(input_shape=(None, None, 3),
                                    return_sizes=False, model_path='models/quality-mlsp-mtl-mse-loss.hdf5'):
    """
    Build a multi-GAP model based on InceptionResNetV2 to extract narrow MLSP features.
    :param input_shape: shape of the input images
    :param return_sizes: return the sizes of each layer: (model, gap_sizes)
    :return: model or (model, gap_sizes)
    """
    base_model = InceptionResNetV2(weights='imagenet',
                                   include_top=False,
                                   input_shape=input_shape)
    base_model.load_weights(model_path)
    feature_layers = [layer for layer in base_model.layers if 'mixed' in layer.name]
    gaps = [GlobalAveragePooling2D(name="gap%d" % i)(layer.output)
            for i, layer in enumerate(feature_layers)]
    concatenated_gaps = Concatenate(name='concatenated_gaps')(gaps)
    model = Model(inputs=base_model.input, outputs=concatenated_gaps)
    
    if return_sizes:
        gap_sizes = [np.int32(gap.get_shape()[1]) for gap in gaps]
        return (model, gap_sizes)
    else:
        return model


def fc_model_softmax(input_num=16928):
    input_ = Input(shape=(input_num,))
    x = Dense(2048, kernel_initializer='he_normal', activation='relu')(input_)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(1024, kernel_initializer='he_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(256, kernel_initializer='he_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    pred = Dense(2, activation='softmax')(x)

    model = Model(input_,pred)
    return model


def predict(x, y=None, model_gap=None, model=None, model_cnn=None, is_norm=False, pca_mg=None, pca_cnn = None):
    '''
    Does prediction on given numpy image using
    model_gap and model
    '''
    try:
        feat_mg = model_gap.predict(x, verbose=0)
    except:
        x = x[None] #changed 02.08 for evaluator visualizing predictions
        feat_mg = model_gap.predict(x, verbose=0)
        
    if pca_mg:
        feat_mg = pca_mg.transform(feat_mg)
    if model_cnn:
        feat_cnn = model_cnn.predict(y, verbose=0)
        if is_norm:
            feat_cnn = normalize_feat_vector(feat_cnn)
        if pca_cnn:
            feat_cnn = pca_cnn.transform(feat_cnn)
        feat = np.concatenate((np.squeeze(feat_mg), np.squeeze(feat_cnn)))
        feat = feat[None]
    else:
        feat = feat_mg
    pred_score = model.predict(feat, verbose=0)

    return pred_score
    
def predict_from_path(model_gap, model, paths, resize_func=None, size=None, for_all=False, save_results=None, 
                      save_to=None, model_cnn=None, is_norm=False, pca_mg = None, pca_cnn = None):
    #always requires list of paths
    predicted = []
    
    for i, path in enumerate(paths):
        img_mg = read_img(path=path, resize_func=resize_func, size=size, for_all=True)
        img_cnn = None
        if model_cnn:
            img_cnn = read_img(path=path, resize_func=add_border_resize, size=(600, 600))
        pred_score = predict(img_mg, img_cnn, model_gap, model, model_cnn, is_norm, pca_mg, pca_cnn)
        predicted.append(pred_score)
    
    predicted = np.array(predicted)
    predicted = np.squeeze(predicted)
    
    if save_results:
        np.save(save_to, np.argmax(predicted, axis=-1))
        
    return predicted

def plot_pred_orig(model_gap, model, imgs_bench, label=None, row_count=2, column_count=10, resize_func=None, size=None, for_all=False, model_cnn=None, is_norm=False, pca_mg=None, pca_cnn=None):
    f, axarr = plt.subplots(row_count, column_count,  figsize=(20,5))

    for i, path in enumerate(imgs_bench):
        x = i // column_count
        y = i % column_count

        img_mg = read_img(path, resize_func=resize_func, size=size, for_all=for_all)
        
        img_cnn = None
        if model_cnn:
            img_cnn = read_img(path=path, resize_func=add_border_resize, size=(600, 600))
            
        pred_score = predict(img_mg, img_cnn, model_gap, model, model_cnn, is_norm, pca_mg, pca_cnn)

        im = cv2.imread(path)
        im = cv2.resize(im, (400, 400))
        
        if row_count == 1:
            axarr[i].imshow(im[..., ::-1]) 
            axarr[i].set_title(f'{str(np.argmax(pred_score, axis=-1)[0])}\n{str(np.round(np.max(pred_score, axis=-1),3)[0])}', fontsize=12)
        else:
            axarr[x, y].imshow(im[..., ::-1]) 
            axarr[x, y].set_title(f'{str(np.argmax(pred_score, axis=-1)[0])}\n{str(np.round(np.max(pred_score, axis=-1),3)[0])}', fontsize=12)

    if label:
        f.suptitle('DeepFL test on ' + label, fontsize=17)
    else: 
        f.suptitle('DeepFL Predictions', fontsize=17)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()




def read_img(path, preprocess=True, resize_func=None, size=None, for_all=False):
    im = Image.open(path).convert('RGB')
    x = img_to_array(im)    
    im.close()
    
    if preprocess:
        if resize_func and for_all:
            x = resize_func(x, size, for_all)
        elif resize_func:
            x = resize_func(x, size)
        x = np.expand_dims(x, axis=0)
        x = x / 255
    
    return x

def calc_metrics(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    precision = np.round(precision * 100, 2)
    recall = np.round(recall * 100, 2)
    
    print(f'Precision: {precision} %')
    print(f'Recall: {recall} %')


def pca_transform(vector, path):
    pca = pk.load(open(path,'rb'))
    return pca.transform(vector)

def preproccess_img(x):
    x = np.expand_dims(x, axis=0)
    x = x / 255
    
    return x


def generate_root_path():
    if glob('Data/AesthAI/alm/splitted/alm_train/images/good/good1/*'): #or if os.path.exists('D:Data/AesthAI')
        return 'D:'
    else:
        return ''