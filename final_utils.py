import tensorflow as tf
import tensorflow_hub as hub
import keras
import tensorflow.keras
import tensorflow.keras.backend as K
import sklearn
import imutils
import random
import socket
import scipy
import time
import math
import json
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import *
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
import pickle as pk
from sklearn.decomposition import PCA
from random import shuffle
from pathlib import Path

random.seed(5)

def resize_main(img_arr, size, interpolation = cv2.INTER_LINEAR):
    dim = None
    height, width = size
    h, w, _ = img_arr.shape
    #if not given width or height return the original image
    if width is None and height is None:
        return img_arr
    #give both height and width to do regular resize without keeping aspect ratio
    if width is not None and height is not None:
        resized = cv2.resize(img_arr, (height, width), interpolation = interpolation)
        return resized
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(img_arr, dim, interpolation = interpolation)
    return resized


def resize_add_border(img_arr, size, interpolation = cv2.INTER_LINEAR):
    h, w, _ = img_arr.shape
    if h > w:
        img_arr = resize_main(img_arr, size=(size[0],None), interpolation = interpolation)
        b_size_left = int((size[1] - img_arr.shape[1])/2)
        b_size_right = size[1] - (b_size_left + img_arr.shape[1])
        img_arr = cv2.copyMakeBorder(img_arr, top=0, bottom=0,
                                         left=b_size_left, right=b_size_right,
                                         borderType=cv2.BORDER_CONSTANT, value=0)
    else:
        img_arr = resize_main(img_arr, size=(None, size[1]),interpolation = interpolation)
        b_size_top = int((size[0] - img_arr.shape[0])/2)
        b_size_bottom = size[0] - (b_size_top + img_arr.shape[0])
        img_arr = cv2.copyMakeBorder(img_arr, top=b_size_top, bottom=b_size_bottom,
                                         left=0, right=0,
                                         borderType=cv2.BORDER_CONSTANT, value=0)
    return img_arr

def resize_area(img_arr, size=(500, 500),interpolation = cv2.INTER_LINEAR):
    area = size[0] * size[1]
    h, w, _ = img_arr.shape
    scale = np.sqrt(float(area) / (w * h))
    resized = cv2.resize(img_arr, (int(w * scale), int(h * scale)),interpolation = interpolation)
    return resized


def resize_max(img_arr, size, for_all=False,interpolation = cv2.INTER_LINEAR):
    h, w, _ = img_arr.shape
    
    if for_all:
        if h > w:
            img_arr = resize_main(img_arr, size = (size[0], None),interpolation = interpolation)
        elif w > h:
            img_arr = resize_main(img_arr, size = (None, size[1]),interpolation = interpolation )
        else:
            img_arr = cv2.resize(img_arr, (size),interpolation = interpolation)
    elif max((h, w)) > size[0]:
        if h > w:
            img_arr = resize_main(img_arr, size = (size[0], None),interpolation = interpolation)
        elif w > h:
            img_arr = resize_main(img_arr, size = (None, size[1]),interpolation = interpolation)
        else:
            img_arr = cv2.resize(img_arr, (size),interpolation = interpolation)
    return img_arr

def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


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


def largest_rotated_rect(w, h, angle):
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def rotate_crop(image, angle):
    """
    Demos the largest_rotated_rect function
    """
    image_height, image_width = image.shape[0:2]

    image_rotated = rotate_image(image, angle)
    image_rotated_cropped = crop_around_center(
            image_rotated,
            *largest_rotated_rect(
                image_width,
                image_height,
                math.radians(angle)))

    return image_rotated_cropped
#Pytorch Model

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


def build_pytorch_feature_extractor_model(weights_path):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', num_classes=16928).to(device)
    
    if weights_path:
        model.load_state_dict(torch.load(weights_path));
    
    print(weights_path)
    model.eval()
    
    return model


def pytorch_extract_features_from_path_automated(source_file, target_file, model, batch_size , resize_func=None, size=None):
    '''
    Takes file split with batch size puts in the deepfl and saves the features in a separate folder
    '''
    random.seed(5)
    
    res = {}
    
    source_file_1 =f'{source_file}'
    target_file_1 = f'{target_file}'
    print('Source = ', source_file_1)
    print('Target = ', target_file_1)

    paths = glob(f'{source_file}/*')
    
    count = int((len(paths)/batch_size))
    name = os.path.basename(source_file)
    
    for step in range(batch_size):        
        path_1 = random.sample(paths, count)
        paths = set(paths) - set(path_1)
        
        if len(paths) < count and paths != []:
            path1 = path_1 + list(paths)

        i = 1
        feats = []
        skiped_list = []
        
        for path in path_1:
            try:           
                img = read_img_pytorch(path, resize_func, size)

                feat = model(img[None])
                feats.append(feat)

                if i % 100 == 0:
                    print('%d images' % (i))
                i += 1   
                
            except:
                print('Skip')
                skiped_list.append(path)
                
        if paths_list != []:
            paths_list = set(path_1) - set(skiped_list)
            
        else:
            paths_list = path_1
        
        with open(f'{target_file_1}/{name}_paths.txt', 'w') as f:
            for path_txt in paths_list:
                f.write(path_txt + '\n')
        
        try:
            np.save(f'{target_file}{name}_{step}', np.squeeze(np.array(feats), axis=1))
            print('Done...')
        
        except:
            res[f'{name}_{step}'].append((np.squeeze(np.array(feats), axis=1)))
            
    return res


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


#Tensorflow Model

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


def extract_features_from_path_automated(source_file, target_file, model, batch_size , crop_func=None, resize_func=True, size=None):
    '''
    Takes file split with batch size puts in the deepfl and saves the features in a separate folder
    '''
    res = {}
    source_file_1 = f'{source_file}'
    target_file_1 = f'{target_file}'
    print('Source = ', source_file_1)
    print('Target = ', target_file_1)

    paths = glob(f'{source_file}/*')
    count = int((len(paths)/batch_size))
    name = os.path.basename(source_file)

    for step in range(batch_size):
        path_1 = random.sample(paths, count)
        paths = set(paths) - set(path_1)

        if len(paths) < count and paths != []:
            path1 = path_1 + list(paths)

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

        if skiped_list != []:
            paths_list = set(path_1) - set(skiped_list)
        else:
            paths_list = path_1

        with open(f'{target_file}/{name}_{step}_paths.txt', 'w') as f:
            for path_txt in paths_list:
                f.write(path_txt + '\n')

        try:
            np.save(f'{target_file}{name}_{step}', np.squeeze(np.array(feats), axis=1))
            print('Done...')
        except:
            res[f'{name}_{step}'].append((np.squeeze(np.array(feats), axis=1)))                                     

    return res  


def extract_features_from_path_automated_json(source_file, target_file, model, label, splitted='',
                                             crop_func=None, resize_func=True, size=None, for_all=False, save_json=False):
    '''
    Takes file split with batch size puts in the deepfl and saves the features in a separate folder
    '''
    print('Source = ', source_file)
    print('Target = ', target_file)
    
    paths = glob(f'{source_file}/*')
    i = 1
    results = []
    skiped_list = []

    for path in paths:
        im_stem = Path(path).stem
        im_name = Path(path).name
        im_npy = im_stem + '.npy'
        try:
            x = read_img(path, preprocess=None)
    
        except:
            print('Skip')
            continue

        if resize_func and for_all:
            x = resize_func(x, size, for_all=for_all)
        elif resize_func:
            x = resize_func(x, size)
            
        if crop_func:
            x = crop_func(x)

        x = preproccess_img(x) #no need read_img do all preproc, changed 26.09
        feat = model.predict(x, batch_size=1, verbose=0) #changed and added batch_size=1 for shufflenet_tf.predict

        if i % 100 == 0:
            print('%d images' % (i))
        i += 1

        image_save_name = Path(im_name)
        feat_json_path = Path(im_npy)
        feat_save_path = Path(target_file) / Path(im_npy)
        np.save(f'{feat_save_path}', np.array(feat))

        results.append({
            'name' : str(image_save_name),
            'feature' : str(feat_json_path),
            'label' : str(label),
            'splitted' : str(splitted)
        })
            
    #Saving the img paths, features and their labels to a .json file
    if save_json:
        with open(Path(target_file).parent.parent.parent / Path(f'data{"_" + splitted if splitted else ""}.json'), 'w') as f:
            json.dump(results, f)

    print('Extracted all...')
    

def extract_mlsp_feats_from_rand(model, path):
    feats = []
    i = 1
    
    im = np.load(path)
    
    for img in im:
        img = img / 255

        feat = model.predict(img[None], verbose=0)
        feats.append(feat)

        if i % 100 == 0:
            print('%d images' % (i))   
        i += 1  

    print('Done...')    
    return np.squeeze(np.array(feats), axis=1)


def extract_static_val_data(data, perc = 0.1):
    np.random.seed(0)
    np.random.shuffle(data)
    lensplit = int( len(data) * perc )
    data_val = data[:lensplit]
    data = data[lensplit:]
    return data, data_val


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


def extract_mlsp_feats_paths(model, paths, resize_func=None, size=None):
    '''
    extract_mlsp_feats using only paths
    no need for csv
    paths -> list of paths of images
    '''
    i = 1
    feats = []
    for path in paths:
        x = read_img(path, preprocess=None)
                
        if resize_func:
            x = resize_func(x, size)

        x = preproccess_img(x)

        feat = model.predict(x, verbose=0)
        feats.append(feat)
        
        if i % 100 == 0:
            print('%d images' % (i))
        i += 1
    print('Done...')
    return np.squeeze(np.array(feats), axis=1)

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

def normalize_feat_vector(feature_vector, 
                          mean_path="Data/splitted/train/features/cnn_efficientnet_b7/norm_vectors/mean.npy", 
                          std_path="Data/plitted/train/features/cnn_efficientnet_b7/norm_vectors/std.npy"):
    
    mean = np.load(mean_path)
    std = np.load(std_path)
    return (feature_vector - mean) / std 

def pca_transform(vector,path = 'models/PCA/PCA_MG_8464_auto.pkl'):
    pca = pk.load(open(path,'rb'))
    return pca.transform(vector)

def data_loader(data, size=None):
    #big pictures require lots of computational data, so we resize them
    imgs = []
    skipped_image = []
    for path in data:
        try:
            img = Image.open(path).convert('RGB')
            if size:
                img = img.resize(size, Image.Resampling.LANCZOS)
            img_tensor = tf.keras.utils.img_to_array(img)
            imgs.append(img_tensor)
        except:
            print('Skip')
    
#         if img.size > (1080,720):
#             x = img_to_array(img)
#             img.close()
#             img = resize_main(x, size = (None, 512))

    imgs = tf.stack(imgs, axis=0)
    imgs = tf.squeeze(imgs)
    
    if imgs.shape[2] == 3:
        imgs = imgs[None]

    return imgs[None]


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
        f.suptitle('test on ' + label, fontsize=17)
    else: 
        f.suptitle('Predictions', fontsize=17)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()


def scheduler(epoch, lr):
    scale = 2.9/4
    if (lr * scale) <= 0.0005:
        return 0.0005
    return lr*scale*1.05

def new_scheduler(epoch, lr):
    max_lr = 0.05
    min_lr = 0.0005
    
    decr_scale = 0.76125
    incr_scale = 0.17 * (epoch + 1)
    
    if (epoch+1) % 11 == 0:
        if lr * incr_scale > max_lr:
            return max_lr
        return lr * incr_scale
    
    if lr * decr_scale < min_lr:
        return min_lr
    
    return lr * decr_scale

def calc_metrics(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
#     print(f'TN - {tn}, FP - {fp}, FN - {fn}, TP - {tp}\n')
    
    precision = tp / (tp + fp)
#     precision_negative = tn / (tn + fn)
    recall = tp / (tp + fn)
#     recall_negative = tn / (tn + fp)
#     f1 = 2 * precision * recall / (precision + recall)
#     acc = (tn + tp) / (tn + tp + fn + fp)
    
    precision = np.round(precision * 100, 2)
    recall = np.round(recall * 100, 2)
#     acc = np.round(acc * 100, 2)
    
    print(f'Precision: {precision} %')
    print(f'Recall: {recall} %')
#     print(f'Accuracy \t {acc} %\n')

def generate_root_path():
    if glob('Data/splitted/train/images/good/good1/*'): #or if os.path.exists('D:Data/AesthAI')
        return '.'
    else:
        return ''
    
def calc_acc(model, weights_path, X_test, y_test, batch_size):
    '''
    Compares Max classes with targets, getting mean class precision
    '''
    model.load_weights(weights_path)
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=0)
    acc = (np.argmax(y_pred,axis=-1) == y_test).mean()
    return np.round(acc * 100, 2)

def lr_exp_decay(epoch, lr):
    k = 0.048
    return lr * np.exp(-k*epoch)


def get_train_pairs(features_bad, features_good, train_size=0.9, shuffle=True):
    train_data_bad = features_bad
    print('train_data_bad shape', train_data_bad.shape)
    train_data_good = features_good
    print('train_data_good shape', train_data_good.shape)

    input_data = np.concatenate( (train_data_bad,train_data_good),axis=0)
    target_data = np.concatenate( (np.zeros(train_data_bad.shape[0]),np.ones(train_data_good.shape[0])) , axis=0 )
    
    X_train, X_test, y_train, y_test = train_test_split(input_data,target_data, train_size=train_size, shuffle=shuffle)
    
    return X_train, X_test, y_train, y_test

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

def preproccess_img(x):
    x = np.expand_dims(x, axis=0)
    x = x / 255
    
    return x