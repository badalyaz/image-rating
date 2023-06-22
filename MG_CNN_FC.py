import numpy as np
import tensorflow as tf
import pickle as pk
from tensorflow import keras
import tensorflow_hub as hub
from final_utils import *
from keras.layers import Concatenate


root_path = generate_root_path()

model_cnn = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1",trainable=False) ])  
model_fc = fc_model_softmax(input_num=9744)
model_fc.load_weights('models/Softmax/MG_CNN/model_fc_softmax_MG_8k_B7_1k_600x600.hdf5')#model_fc_softmax_MG_8k_B7_1k_600x600
model_base=model_inceptionresnet_multigap()

pca_mg_path ='models/PCA/PCA_MG_8464_auto.pkl'
pca_cnn_path= 'models/PCA/PCA_CNN_1280_auto.pkl'
pca_mg = pk.load(open(pca_mg_path,'rb'))
pca_cnn =  pk.load(open(pca_cnn_path,'rb'))

def resize_main_tf(img_arr, size, method = "bilinear"):
    dim = None
    height, width = size
    h, w, _ = img_arr.shape

    #if not given width or height return the original image
    if width is None and height is None:
        return img_arr / 255
    #give both height and width to do regular resize without keeping aspect ratio
    if width is not None and height is not None:
        resized = tf.image.resize(img_arr, (height, width), method = method)
        return resized / 255
    if width is None:
        r = height / float(h)
        dim = (height,int(w * r))
    else:
        r = width / float(w)
        dim = ( int(h * r),width)
    resized = tf.image.resize(img_arr, dim, method = method) 
  
    return resized / 255

def resize_max_tf(img_arr, size, for_all=False, method = "bilinear"):
    h, w, _ = img_arr.shape
    print("Resize max", img_arr.shape)
    if for_all:
        if h > w:
            img_arr = resize_main_tf(img_arr, size = (size[0], None),method = method)
        elif w > h:
            img_arr = resize_main_tf(img_arr, size = (None, size[1]),method = method )
        else:
            img_arr = tf.image.resize(img_arr, (size),method = method)
    elif max((h, w)) > size[0]:
        if h > w:
            img_arr = resize_main_tf(img_arr, size = (size[0], None),method = method)
        elif w > h:
            img_arr = resize_main_tf(img_arr, size = (None, size[1]),method = method)
        else:
            img_arr = tf.image.resize(img_arr, (size),method = method) / 255  
   
    return img_arr 
 
def resize_add_border_tf(img_arr, size, method = "bilinear" ):
    h, w,  _ = img_arr.shape

    if h > w:
  
        img_arr = resize_main_tf(img_arr, size=(size[0], None), method = method)
        b_size_left = int((size[1] - img_arr.shape[1])/2)
    

        img_arr = tf.image.pad_to_bounding_box(img_arr, offset_height=0,offset_width=b_size_left, 
                                              target_height = size[0], target_width = size[1]  )
    else:
        img_arr = resize_main_tf(img_arr, size=(None, size[1]),method = method)
        b_size_top = int((size[0] - img_arr.shape[0])/2)
  
        img_arr = tf.image.pad_to_bounding_box(img_arr, offset_height=b_size_top, offset_width = 0,
                                               target_height = size[0], target_width = size[1])
 
    return img_arr
    
class Resize_Max(keras.layers.Layer):
    def __init__(self,size = (996,996),for_all = False):
        super(Resize_Max, self).__init__()
        self.size = size
        self.for_all = for_all
        
        
    def call(self, inputs):
#         resized = resize_main_tf(inputs[0], self.size)
        resized = resize_max_tf(inputs[0],self.size, self.for_all) 
#         resized = resize_max(inputs[0].numpy(),self.size, self.for_all) / 255
        return resized[None]
    
    
class Resize_Add_Border(keras.layers.Layer):
    def __init__(self,size = (600,600)):
        super(Resize_Add_Border, self).__init__()
        self.size = size
             
    def call(self, inputs):
        with_borders = resize_add_border_tf(inputs[0], self.size) 
        return with_borders[None]
class PCA_MG(keras.layers.Layer):
    def __init__(self, pca_path):
        super(PCA_MG, self).__init__()
        self.pca =  pk.load(open(pca_path,'rb'))
        self.pca_matrix =  self.pca.components_
        self.pca_mean = self.pca.mean_
        
    def call(self, inputs):
        return tf.matmul(inputs - self.pca_mean ,self.pca_matrix.T) #self.pca_matrix.transform(inputs.numpy())
    

class PCA_CNN(keras.layers.Layer):
    def __init__(self, pca_path):
        super(PCA_CNN, self).__init__()
        self.pca =  pk.load(open(pca_path,'rb'))
        self.pca_matrix =  self.pca.components_
        self.pca_mean = self.pca.mean_
        
    def call(self, inputs):
        return tf.matmul(inputs - self.pca_mean ,self.pca_matrix.T) #self.pca_matrix.transform(inputs.numpy())

class model_mg_cnn_fc(tf.keras.Model):
    def __init__(self, model_base=model_base, model_cnn=model_cnn, model_fc=model_fc, pca_mg = False, pca_cnn = False,
                 pca_mg_path=pca_mg_path, pca_cnn_path=pca_cnn_path ):
        super(model_mg_cnn_fc, self).__init__()
        
        self.layer_multigap = model_base
        self.layer_cnn = model_cnn
        self.layer_fc = model_fc
        self.concat = Concatenate()
        self.layer_resize_max = Resize_Max(size = (996, 996), for_all = False) 
        self.layer_resize_add_border = Resize_Add_Border(size = (600, 600))
        
        self.layer_pca_mg =  PCA_MG(pca_mg_path)
        self.layer_pca_cnn =  PCA_CNN(pca_cnn_path)
        
        self.pca_mg = pca_mg
        self.pca_cnn = pca_cnn
 
    def call(self, img):
        print('image shape', img.shape)
        x1 = self.layer_resize_max(img)
        print('image shape after resize max', x1.shape)
        x1 = self.layer_multigap(x1)
        print('mg feature vector shape', x1.shape)
        
        x2 = self.layer_resize_add_border(img)
        print('image shape after resize border', x2.shape)
        x2 = self.layer_cnn(x2)
        print('cnn feature vector shape', x2.shape)
        if self.pca_mg:
            x1 = self.layer_pca_mg(x1)
            print('mg feature vector shape after pca', x1.shape)
        if self.pca_cnn:
            x2 = self.layer_pca_cnn(x2)
            print('cnn feature vector shape after pca', x2.shape)
           
        x = self.concat([x1, x2])
        print('feature vector shape after concat', x.shape)
        x = self.layer_fc(x)
        print('feature vector shape after fc', x.shape)
        print('print before return')
        return x



