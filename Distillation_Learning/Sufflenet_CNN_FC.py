from shufflenetV2_tf2 import ShufflenetV2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from keras.layers import Concatenate
from keras.layers import *
from keras.models import Model
import time

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

# model_cnn = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1",trainable=False) ])
# model_cnn = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",trainable=False) ]) 18208
model_fc = fc_model_softmax(input_num=8464)
# model_fc.load_weights('models/Softmax/model_fc_softmax_16k_2k.hdf5')
model_shufflenet = ShufflenetV2(num_classes=8464, training=True)
# model_shufflenet.build([32, 600, 600, 3]) #2 --> batch_size
# model_shufflenet.load_weights('models/Shufflenet/Shufflenet_on_600x600_labels_MG_all_res_996_16.09.h5')
# print(model_shufflenet.summary())

class shufflenet_cnn_fc(tf.keras.Model):
    def __init__(self, model_shufflenet=model_shufflenet,  model_fc=model_fc): #model_cnn=model_cnn,
        super(shufflenet_cnn_fc, self).__init__()
        
        self.layer_shufflenet = model_shufflenet
#         self.layer_cnn = model_cnn
        self.layer_fc = model_fc
#         self.concat = Concatenate()

    def call(self, img):

        x = self.layer_shufflenet(img)
#         x2 = self.layer_cnn(img)
#         x = self.concat([x1, x2])
        x = self.layer_fc(x)
            
        return x