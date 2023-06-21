import sklearn
import imutils
import random
import scipy
import time
import json
import cv2
import os

import torch
from torchvision import transforms

import tensorflow as tf
from keras.layers import *
from random import shuffle
from keras.models import Model
from keras.applications import *
from keras.optimizers import SGD, Adam
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.model_selection import train_test_split

from sklearn import metrics
from PIL import Image
from glob import glob
from numpy.random import rand

import keras
import tensorflow.keras
import tensorflow.keras.backend as K

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = 'cpu'
if torch.cuda.is_available:
    device = 'cuda'


random.seed(5)


def build_pytorch_feature_extractor_model(weights_path):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', num_classes=16928).to(device)
    
    if weights_path:
        model.load_state_dict(torch.load(weights_path));
    
    print(weights_path)
    model.eval()
    
    return model


def calc_metrics(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    print(f'TN - {tn}, FP - {fp}, FN - {fn}, TP - {tp}\n')
    
    precision = tp / (tp + fp)
    precision_negative = tn / (tn + fn)
    recall = tp / (tp + fn)
    recall_negative = tn / (tn + fp)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (tn + tp) / (tn + tp + fn + fp)
    
    print(f'Precision \t\t\t {precision}')
    print(f'Precision (for negative class)\t {precision_negative}')
    print(f'Recall \t\t\t\t {recall}')
    print(f'Recall (for negative class)\t {recall_negative}')
    print(f'Accuracy \t\t\t {acc}\n')
    
    
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
        b_size_botomm = size[0] - (b_size_top + img_arr.shape[0])
        img_arr = cv2.copyMakeBorder(img_arr, top=b_size_top, bottom=_size_bottom,
                                         left=0, right=0,
                                         borderType=cv2.BORDER_CONSTANT, value=0)
    return img_arr


def resize_max(img_arr, size):
    h, w, _ = img_arr.shape
    if h > w:
        img_arr = resize_main(img_arr, size = (size[0], None))
    elif w > h:
        img_arr = resize_main(img_arr, size = (None, size[1]))
    else:
        img_arr = cv2.resize(img_arr, (size))
    return img_arr



def random_crop(img, scale_range=(0.4, 0.6)):
    scale = np.random.uniform(scale_range[0], scale_range[1], size=1)

    height, width = int(img.shape[0]*np.sqrt(scale)), int(img.shape[1]*np.sqrt(scale))
    x = random.randint(0, img.shape[1] - int(width))
    y = random.randint(0, img.shape[0] - int(height))
    cropped = img[y:y+height, x:x+width]
    
    return cropped


def random_crop_without_ar(img, scale_range=(0.4, 0.6)):
    scale = np.random.uniform(scale_range[0], scale_range[1], size=1)

    scale_h = np.sqrt(scale)
    scale_h = np.random.uniform(scale_h, 0.9, size=1)
    height = int(scale_h * img.shape[0])
    width = int(scale*img.shape[0]*img.shape[1]/height)
    x = random.randint(0, img.shape[1] - int(width))
    y = random.randint(0, img.shape[0] - int(height))
    cropped = img[y:y+height, x:x+width]

    return cropped


def read_img(path, preprocess=True, resize_func=None, size=None):
    im = Image.open(path).convert('RGB')
    x = img_to_array(im)    
    im.close()
    
    if preprocess:
        if resize_func:
            x = resize_func(x, size)

        x = np.expand_dims(x, axis=0)
        x = x / 255
    
    return x

def preprocess_img(x):
    x = np.expand_dims(x, axis=0)
    x = x / 255
    
    return x

def read_img_pytorch(path, resize_func=None, size=None):
    img = Image.open(path).convert('RGB')

    transform = transforms.Compose([
            transforms.ToTensor(),
            ])
    
    if resize_func:
        transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        ])
        
    img_tensor = transform(img).to(device)
    
    return img_tensor


def get_train_pairs(bad, good,train_size=0.9,shuffle=True):
    # Get pairs
    train_data_bad = bad
    print('train_data_bad shape', train_data_bad.shape)
    train_data_good = good
    print('train_data_good shape', train_data_good.shape)

    input_data = np.concatenate( (train_data_bad,train_data_good),axis=0)
    target_data = np.concatenate( (np.zeros(train_data_bad.shape[0]),np.ones(train_data_good.shape[0])) , axis=0 )
    
    X_train, X_test, y_train, y_test = train_test_split(input_data,target_data, train_size=train_size, shuffle=shuffle)

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def plot_pred_orig(model_gap, model, imgs_bench, label, row_count=2, column_count=10, resize_func=None, size=None):
    f, axarr = plt.subplots(row_count, column_count,  figsize=(20,5))

    for i, path in enumerate(imgs_bench):
        x = i // column_count
        y = i % column_count

        img = read_img(path, resize_func, size)
            
        pred_score = predict(img, model_gap, model)

        im = cv2.imread(path)
        im = cv2.resize(im, (400, 400))
        
        if row_count == 1:
            axarr[i].imshow(im[..., ::-1]) 
            axarr[i].set_title(f'{str(np.argmax(pred_score, axis=-1)[0])}', fontsize=17)
        else:
            axarr[x, y].imshow(im[..., ::-1]) 
            axarr[x, y].set_title(f'{str(np.argmax(pred_score, axis=-1)[0])}', fontsize=17)

    f.suptitle('DeepFL test on ' + label, fontsize=20)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()
    

def pytorch_predict(x, feature_extractor_model, model):
    feat_torch = feature_extractor_model(x)
    np_tensor = feat_torch.detach().cpu().numpy()
    feat = tf.convert_to_tensor(np_tensor)
    pred_score = model.predict(feat, verbose=0)

    return pred_score

    
def pytorch_predict_from_path(model_gap, model, paths, resize_func=None, size=None, save_results=None, save_to=None):
    predicted = []
    
    for i, path in enumerate(paths):
        img = read_img_pytorch(path, resize_func, size)
            
        pred_score = pytorch_predict(img[None], model_gap, model)
        predicted.append(pred_score)
    
    predicted = np.array(predicted)
    predicted = np.squeeze(predicted)
    
    if save_results:
        np.save(save_to, np.argmax(predicted, axis=-1))
        
    return predicted

    
def predict(x, model_gap, model):
    '''
    Does prediction on given numpy image using
    model_gap and model
    '''
    feat = model_gap.predict(x, verbose=0)
    pred_score = model.predict(feat, verbose=0)

    return pred_score
    

def predict_from_path(model_gap, model, paths, resize_func=None, size=None, save_results=None, save_to=None):
    #always requires list of paths
    predicted = []
    
    for i, path in enumerate(paths):
        img = read_img(path, resize_func, size)
            
        pred_score = predict(img, model_gap, model)
        predicted.append(pred_score)
    
    predicted = np.array(predicted)
    predicted = np.squeeze(predicted)
    
    if save_results:
        np.save(save_to, np.argmax(predicted, axis=-1))
        
    return predicted


def calc_acc(model, weights_path, X_test, y_test):
    '''
    Compares Max classes with targets, getting mean class precision
    '''
    X_test = data_loader(X_test)
    model.load_weights(weights_path)
    y_pred = model.predict(X_test)
    acc = (np.argmax(y_pred,axis=-1) == y_test).mean()
    
    return acc


def calc_acc_from_path(model_gap, model, predict, paths, labels, resize_func=None, size=None):
    predicted = predict(model_gap, model, paths, resize_func, size)
    acc = (np.argmax(predicted, axis=-1) == labels).mean()

    return acc    
    
    
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


def extract_static_val_data(data, perc = 0.1):
    np.random.seed(0)
    np.random.shuffle(data)
    lensplit = int( len(data) * perc )
    data_val = data[:lensplit]
    data = data[lensplit:]
    return data, data_val


def extract_mlsp_feats_from_rand(model, path):
    feats = []
    i = 1
    
    im = np.load(path)
    
    for img in im:
        img = img / 255

        feat = model.predict(img, verbose=0)
        feats.append(feat)

        if i % 100 == 0:
            print('%d images' % (i))   
        i += 1  

    print('Done...')    
    return np.squeeze(np.array(feats), axis=1)


def extract_features_from_path_automated(source_file, target_file, model, batch_size , crop_func=None, resize_func=None, size=None):
    '''
    Takes file split with batch size puts in the deepfl and saves the features in a separate folder
    '''
    res = {}
    
    source_file_1 =f'{source_file}'
    target_file_1 = f'{target_file}'
    print('Source = ', source_file_1)
    print('Target = ', target_file_1)

    paths = glob(f'{source_file}/*')
    count = int((len(paths)/batch_size))
    name = source_file.split('\\')[-1]
    
    
    for step in range(batch_size):
        path_1 = random.sample(paths, count)
        
        paths = set(paths) - set(path_1)
        paths = list(paths)
        path_1 = list(path_1)
        
        if len(paths) < count and paths != []:
            path_1.append(paths)
            
        i = 1
        feats = []
        skiped_list = []
        
        for path in path_1:
            try:               
                x = read_img(path, preprocess=None)
                
                if resize_func:
                    x = resize_func(x, size)
                    
                if crop_func:
                    x = crop_func(x)
                    
                x = preproccess_img(x)
                
                feat = model.predict(x, verbose=0)
                feats.append(feat)
                     
                if i % 100 == 0:
                    print('%d images' % (i))

                i += 1
              
            except:
                print('Skip')
                skiped_list.append(path)
        
        
        paths_list = set(path_1) - set(skiped_list)
        with open(f'{target_file}/{name}_{step}_paths.txt', 'w') as f:
            for path_txt in paths_list:
                f.write(path_txt + '\n')

        try:
            np.save(f'{target_file}{name}_{step}', np.squeeze(np.array(feats), axis=1))
            print('Done...')
        
        except:
            res[f'{name}_{step}'].append((np.squeeze(np.array(feats), axis=1)))
                                         
    
    return res  


def scheduler(epoch, lr):
    if epoch % 5 != 0:
        return lr 
    else:
        return lr * 0.1


def extract_mlsp_feats(ids, model, data_dir, resize_func=None, size=None):
    feats = []
    i = 1

    for index, row in ids.iterrows():
        path = data_dir + str(row[0])
        
        img = read_img(path, resize_func, size)

        feat = model.predict(img, verbose=0)
        feats.append(feat)
        
        if i % 100 == 0:
            print('%d images' % (i))
            
        i += 1

    print('Done...')   
    return np.squeeze(np.array(feats), axis=1)