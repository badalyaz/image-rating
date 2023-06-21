import random
import cv2
import os

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import *
from random import shuffle
from tensorflow.keras.models import Model
from tensorflow.keras.applications import *
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

from sklearn import metrics
from PIL import Image
from glob import glob
import pickle as pk
from numpy.random import rand

import numpy as np
import matplotlib.pyplot as plt

random.seed(5)

def resize_add_border(img_arr, size):
    h, w, _ = img_arr.shape
    if h > w:
        img_arr = resize_main(img_arr, size=(size[0],None))
        b_size_left = int((size[1] - img_arr.shape[1])/2)
        b_size_right = size[1] - (b_size_left + img_arr.shape[1])
        img_arr = cv2.copyMakeBorder(img_arr, top=0, bottom=0,
                                         left=b_size_left, right=b_size_right,
                                         borderType=cv2.BORDER_CONSTANT, value=0)
    else:
        img_arr = resize_main(img_arr, size=(None, size[1]))
        b_size_top = int((size[0] - img_arr.shape[0])/2)
        b_size_bottom = size[0] - (b_size_top + img_arr.shape[0])
        img_arr = cv2.copyMakeBorder(img_arr, top=b_size_top, bottom=b_size_bottom,
                                         left=0, right=0,
                                         borderType=cv2.BORDER_CONSTANT, value=0)
    return img_arr

def resize_main(img_arr, size):
    dim = None
    height, width = size
    h, w, _ = img_arr.shape
    #if not given width or height return the original image
    if width is None and height is None:
        return img_arr
    #give both height and width to do regular resize without keeping aspect ratio
    if width is not None and height is not None:
        resized = cv2.resize(img_arr, (height, width))
        return resized
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(img_arr, dim)
    return resized


def resize_max(img_arr, size, for_all=False):
    h, w, _ = img_arr.shape
    
    if for_all:
        if h > w:
            img_arr = resize_main(img_arr, size = (size[0], None))
        elif w > h:
            img_arr = resize_main(img_arr, size = (None, size[1]))
        else:
            img_arr = cv2.resize(img_arr, (size))
    elif max((h, w)) > size[0]:
        if h > w:
            img_arr = resize_main(img_arr, size = (size[0], None))
        elif w > h:
            img_arr = resize_main(img_arr, size = (None, size[1]))
        else:
            img_arr = cv2.resize(img_arr, (size))
    return img_arr


def model_inceptionresnet_multigap(input_shape=(None, None, 3), 
                                   return_sizes=False, model_path='models/quality-mlsp-mtl-mse-loss.hdf5'):
    """
    Build InceptionResNetV2 multi-GAP model, that extracts narrow MLSP features.

    :param input_shape: shape of the input images
    :param return_sizes: return the sizes of each layer: (model, gap_sizes)
    :return: model or (model, gap_sizes)
    """
    model_base = InceptionResNetV2(weights='imagenet',
                                  include_top=False,
                                  input_shape=input_shape)

    model_base.load_weights(model_path)

    feature_layers = [l for l in model_base.layers if 'mixed' in l.name]
    gaps = [GlobalAveragePooling2D(name="gap%d" % i)(l.output)
           for i, l in enumerate(feature_layers)]
    concat_gaps = Concatenate(name='concatenated_gaps')(gaps)

    model = Model(inputs=model_base.input, outputs=concat_gaps)

    if return_sizes:
        gap_sizes = [np.int32(g.get_shape()[1]) for g in gaps]
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
            img_cnn = read_img(path=path, resize_func=resize_add_border, size=(600, 600))
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
            img_cnn = read_img(path=path, resize_func=resize_add_border, size=(600, 600))
            
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