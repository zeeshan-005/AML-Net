from google.colab import drive
drive.mount('/content/drive')

!nvidia-smi

import tensorflow as tf
import numpy as np
import cv2
import glob

mask_id = []
for infile in sorted(glob.glob('/content/drive/MyDrive/Datasets/lgg/*/*_mask*')):
    mask_id.append(infile)

print(mask_id[0:5])

print(len(mask_id))

"""Split images into training 70% (961), validation 150% (206), and holdout test images 15% (206)."""

import random

mask_V = []
for i in range(206):
  random_value = random.choice(mask_id)
  mask_V.append(random_value)
  mask_id.remove(random_value)

mask_T = []
for i in range(206):
  random_value = random.choice(mask_id)
  mask_T.append(random_value)
  mask_id.remove(random_value)

print("Train: ", len(mask_id))
print("Validation: ",len(mask_V))
print("Test: ", len(mask_T))

image_ = []
for img_path in mask_id:
    image_.append(img_path.replace('_mask',''))

image_V = []
for img_path in mask_V:
    image_V.append(img_path.replace('_mask',''))

image_T = []
for img_path in mask_T:
    image_T.append(img_path.replace('_mask',''))

print("Train: ", len(image_))
print("Validation: ",len(image_V))
print("Test: ", len(image_T))

height=1536
width=1536

def DataGen4():
    img_ = []
    mask_  = []
    for i in range(len(image_)):
        image = cv2.imread(image_[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height//4,width//4), interpolation = cv2.INTER_AREA)
        image=image/255
        mask = cv2.imread(mask_id[i],0)
        mask = cv2.resize(mask, (height//4,width//4), interpolation = cv2.INTER_AREA)
        mask=mask/255
        F = np.expand_dims(mask, axis=-1)
        img_.append(image)
        mask_.append(F)
    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,mask_

def DataGen3():
    img_ = []
    mask_  = []
    for i in range(len(image_)):
        image = cv2.imread(image_[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height//8,width//8), interpolation = cv2.INTER_AREA)
        image=image/255
        mask = cv2.imread(mask_id[i],0)
        mask = cv2.resize(mask, (height//8,width//8), interpolation = cv2.INTER_AREA)
        mask=mask/255
        F = np.expand_dims(mask, axis=-1)
        img_.append(image)
        mask_.append(F)
    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,mask_

def DataGen2():
    img_ = []
    mask_  = []
    for i in range(len(image_)):
        image = cv2.imread(image_[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height//16,width//16), interpolation = cv2.INTER_AREA)
        image=image/255
        mask = cv2.imread(mask_id[i],0)
        mask = cv2.resize(mask, (height//16,width//16), interpolation = cv2.INTER_AREA)
        mask=mask/255
        F = np.expand_dims(mask, axis=-1)
        img_.append(image)
        mask_.append(F)
    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,mask_

def DataGen1():
    img_ = []
    mask_  = []
    for i in range(len(image_)):
        image = cv2.imread(image_[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height//32,width//32), interpolation = cv2.INTER_AREA)
        image=image/255
        mask = cv2.imread(mask_id[i],0)
        mask = cv2.resize(mask, (height//32,width//32), interpolation = cv2.INTER_AREA)
        mask=mask/255
        F = np.expand_dims(mask, axis=-1)
        img_.append(image)
        mask_.append(F)
    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,mask_

def DataGenV():
    #img_ = []
    #mask_  = []
    #c1=[]
    c2=[]
    c3=[]
    c4=[]
    c5=[]
    #y1=[]
    y2=[]
    y3=[]
    y4=[]
    y5=[]

    for i in range(len(image_V)):
        image = cv2.imread(image_V[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA)
        image=image/255
        #cc1 = cv2.resize(image, (height//2,width//2), interpolation = cv2.INTER_AREA)
        cc2 = cv2.resize(image, (height//4,width//4), interpolation = cv2.INTER_AREA)
        cc3 = cv2.resize(image, (height//8,width//8), interpolation = cv2.INTER_AREA)
        cc4 = cv2.resize(image,(height//16,width//16), interpolation = cv2.INTER_AREA)
        cc5 = cv2.resize(image, (height//32,width//32), interpolation = cv2.INTER_AREA)
        mask = cv2.imread(mask_V[i],0)
        mask = cv2.resize(mask, (height,width), interpolation = cv2.INTER_AREA)
        mask=mask/255
        #yy1 = cv2.resize(mask, (height//2,width//2), interpolation = cv2.INTER_AREA)
        yy2 = cv2.resize(mask, (height//4,width//4), interpolation = cv2.INTER_AREA)
        yy3 = cv2.resize(mask, (height//8,width//8), interpolation = cv2.INTER_AREA)
        yy4 = cv2.resize(mask, (height//16,width//16), interpolation = cv2.INTER_AREA)
        yy5 = cv2.resize(mask, (height//32,width//32), interpolation = cv2.INTER_AREA)
        F = np.expand_dims(mask, axis=-1)
        #yy1 = np.expand_dims(yy1, axis=-1)
        yy2 = np.expand_dims(yy2, axis=-1)
        yy3 = np.expand_dims(yy3, axis=-1)
        yy4 = np.expand_dims(yy4, axis=-1)
        yy5 = np.expand_dims(yy5, axis=-1)
        #img_.append(image)
        #mask_.append(F)
        #c1.append(cc1)
        c2.append(cc2)
        c3.append(cc3)
        c4.append(cc4)
        c5.append(cc5)
        #y1.append(yy1)
        y2.append(yy2)
        y3.append(yy3)
        y4.append(yy4)
        y5.append(yy5)
    #C1=np.array(c1)
    C2=np.array(c2)
    C3=np.array(c3)
    C4=np.array(c4)
    C5=np.array(c5)
    #Y1=np.array(y1)
    Y2=np.array(y2)
    Y3=np.array(y3)
    Y4=np.array(y4)
    Y5=np.array(y5)
    #img_ = np.array(img_)
    #mask_  = np.array(mask_)
    return C2,C3,C4,C5,Y2,Y3,Y4,Y5

def DataGenT():
    #img_ = []
    #mask_  = []
    #c1=[]
    c2=[]
    c3=[]
    c4=[]
    c5=[]

    #y1=[]
    y2=[]
    y3=[]
    y4=[]
    y5=[]

    for i in range(len(image_T)):
        image = cv2.imread(image_T[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA)
        image=image/255
        #cc1 = cv2.resize(image, (height//2,width//2), interpolation = cv2.INTER_AREA)
        cc2 = cv2.resize(image, (height//4,width//4), interpolation = cv2.INTER_AREA)
        cc3 = cv2.resize(image, (height//8,width//8), interpolation = cv2.INTER_AREA)
        cc4 = cv2.resize(image,(height//16,width//16), interpolation = cv2.INTER_AREA)
        cc5 = cv2.resize(image, (height//32,width//32), interpolation = cv2.INTER_AREA)

        mask = cv2.imread(mask_T[i],0)
        mask = cv2.resize(mask, (height,width), interpolation = cv2.INTER_AREA)
        mask=mask/255
        #yy1 = cv2.resize(mask, (height//2,width//2), interpolation = cv2.INTER_AREA)
        yy2 = cv2.resize(mask, (height//4,width//4), interpolation = cv2.INTER_AREA)
        yy3 = cv2.resize(mask, (height//8,width//8), interpolation = cv2.INTER_AREA)
        yy4 = cv2.resize(mask, (height//16,width//16), interpolation = cv2.INTER_AREA)
        yy5 = cv2.resize(mask, (height//32,width//32), interpolation = cv2.INTER_AREA)

        #F = np.expand_dims(mask, axis=-1)
        #yy1 = np.expand_dims(yy1, axis=-1)
        yy2 = np.expand_dims(yy2, axis=-1)
        yy3 = np.expand_dims(yy3, axis=-1)
        yy4 = np.expand_dims(yy4, axis=-1)
        yy5 = np.expand_dims(yy5, axis=-1)

        #img_.append(image)
        #mask_.append(F)
        #c1.append(cc1)
        c2.append(cc2)
        c3.append(cc3)
        c4.append(cc4)
        c5.append(cc5)

        #y1.append(yy1)
        y2.append(yy2)
        y3.append(yy3)
        y4.append(yy4)
        y5.append(yy5)

    #C1=np.array(c1)
    C2=np.array(c2)
    C3=np.array(c3)
    C4=np.array(c4)
    C5=np.array(c5)

    #Y1=np.array(y1)
    Y2=np.array(y2)
    Y3=np.array(y3)
    Y4=np.array(y4)
    Y5=np.array(y5)

    #img_ = np.array(img_)
    #mask_  = np.array(mask_)
    return C2,C3,C4,C5,Y2,Y3,Y4,Y5
    #return C2,Y2

"""Loss Function"""

import tensorflow.keras.backend as K
def dice_coef(y_true, y_pred, smooth=2):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

"""Evaluation Metrics"""

import numpy as np
from sklearn.metrics import confusion_matrix
def Evaluation_Metrics(result,GT):
    Y=np.reshape(GT,(result.shape[0]*result.shape[2]*result.shape[2],1))
    Y=Y.astype(int)
    P=np.reshape(result,(result.shape[0]*result.shape[2]*result.shape[2],1))
    P=P.astype(int)
    tn, fp, fn, tp=confusion_matrix(Y, P,labels=[0,1]).ravel()
    F1=2*tp/(2*tp+fp+fn)
    iou=tp/(tp+fn+fp)
    Sensitivity=tp/(tp+fn)
    print("IoU  is:  ",iou)
    print("F1_Score is:  ",F1)
    print("Sensitivity  is:  ",Sensitivity)
    return iou,F1,Sensitivity

"""Plot Graphs"""

def plot_graph(history):
  #  "Accuracy"
  plt.rcParams["figure.figsize"] = (15,10)
  f, ax = plt.subplots(1)
  ax.set_ylim(bottom=0)
  plt.rcParams.update({'font.size': 18})
  plt.plot(history.history['dice_coef'])
  plt.plot(history.history['val_dice_coef'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()

  # "Loss"
  f, ax = plt.subplots(1)
  ax.set_ylim(bottom=0)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()

def plot_level_train_graph(history1, history2, history3, history4):
  #  "Accuracy"
  plt.rcParams["figure.figsize"] = (15,10)
  plt.rcParams.update({'font.size': 18})
  plt.plot(history1.history['dice_coef'])
  plt.plot(history2.history['dice_coef'])
  plt.plot(history3.history['dice_coef'])
  plt.plot(history4.history['dice_coef'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['Level 1', 'Level 2', 'Level 3', 'Level 4'], loc='upper left')
  plt.show()

  # "Loss"
  plt.plot(history1.history['loss'])
  plt.plot(history2.history['loss'])
  plt.plot(history3.history['loss'])
  plt.plot(history4.history['loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['Level 1', 'Level 2', 'Level 3', 'Level 4'], loc='upper left')
  plt.show()

def plot_level_val_graph(history1, history2, history3, history4):
  #  "Accuracy"
  plt.rcParams["figure.figsize"] = (15,10)
  plt.rcParams.update({'font.size': 18})
  plt.plot(history1.history['val_dice_coef'])
  plt.plot(history2.history['val_dice_coef'])
  plt.plot(history3.history['val_dice_coef'])
  plt.plot(history4.history['val_dice_coef'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['Level 1', 'Level 2', 'Level 3', 'Level 4'], loc='upper left')
  plt.show()

  # "Loss"
  plt.plot(history1.history['val_loss'])
  plt.plot(history2.history['val_loss'])
  plt.plot(history3.history['val_loss'])
  plt.plot(history4.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['Level 1', 'Level 2', 'Level 3', 'Level 4'], loc='upper left')
  plt.show()

"""Models"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

class spatial_attention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        self.kernel_size = kernel_size
        super(spatial_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv2d = tf.keras.layers.Conv2D(filters=1, kernel_size=self.kernel_size,
                                             strides=1, padding='same', activation='sigmoid',
                                             kernel_initializer='he_normal', use_bias=False)
        super(spatial_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True))(inputs)
        max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=-1, keepdims=True))(inputs)
        concat = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
        feature = self.conv2d(concat)

        return tf.keras.layers.multiply([inputs, feature])

import tensorflow.keras as keras
def MYMODEL1():
    inputs1 = keras.layers.Input((48, 48, 3))
    conv1 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs1)
    conv1 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    up7 = keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(pool3))


    d31 = keras.layers.Conv2D(21, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same') (conv3)
    d32 = keras.layers.Conv2D(21, (3, 3), dilation_rate=4, activation='relu', kernel_initializer='he_normal', padding='same') (d31)
    d33 = keras.layers.Conv2D(21, (3, 3), dilation_rate=8, activation='relu', kernel_initializer='he_normal', padding='same') (d32)
    sa3 = spatial_attention(64)(d33)
    #se3 = ChannelAttention(64, 8)(conv3)
    #se3 = SpatialAttention(7)(se3)

    merge7 = keras.layers.concatenate([sa3,up7], axis = 3)
    conv7 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    up8 = keras.layers.Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv7))

    d21 = keras.layers.Conv2D(21, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same') (conv2)
    d22 = keras.layers.Conv2D(21, (3, 3), dilation_rate=4, activation='relu', kernel_initializer='he_normal', padding='same') (d21)
    d23 = keras.layers.Conv2D(21, (3, 3), dilation_rate=8, activation='relu', kernel_initializer='he_normal', padding='same') (d22)
    sa2 = spatial_attention(32)(d23)
    #se2 = ChannelAttention(32, 8)(conv2)
    #se2 = SpatialAttention(7)(se2)

    merge8 = keras.layers.concatenate([sa2,up8], axis = 3)
    conv8 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    up9 = keras.layers.Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv8))

    d11 = keras.layers.Conv2D(21, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same') (conv1)
    d12 = keras.layers.Conv2D(21, (3, 3), dilation_rate=4, activation='relu', kernel_initializer='he_normal', padding='same') (d11)
    d13 = keras.layers.Conv2D(21, (3, 3), dilation_rate=8, activation='relu', kernel_initializer='he_normal', padding='same') (d12)
    sa1 = spatial_attention(16)(d13)
    #se1 = ChannelAttention(16, 8)(conv1)
    #se1 = SpatialAttention(7)(se1)

    merge9 = keras.layers.concatenate([sa1,up9], axis = 3)
    conv9 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 =keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)
    model1 = keras.models.Model(inputs=inputs1, outputs=conv10)
    return model1

def MYMODEL2():
    inputs1 = keras.layers.Input((48,48, 1))
    inputs2 = keras.layers.Input((96,96, 3))
    #inputs1 = keras.layers.Input((48,48, 1))
    #inputs2 = keras.layers.Input((48,48, 3))
    #c_1x1 = keras.layers.Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs2)
    Up_s=keras.layers.UpSampling2D(size = (2,2))(inputs1)
    x = keras.layers.concatenate([inputs2, Up_s],axis=-1)
    #x = keras.layers.concatenate([inputs2, inputs1],axis=-1)

    conv1 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    conv1 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    up7 = keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(pool3))

    d31 = keras.layers.Conv2D(21, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same') (conv3)
    d32 = keras.layers.Conv2D(21, (3, 3), dilation_rate=4, activation='relu', kernel_initializer='he_normal', padding='same') (d31)
    d33 = keras.layers.Conv2D(21, (3, 3), dilation_rate=8, activation='relu', kernel_initializer='he_normal', padding='same') (d32)
    sa3 = spatial_attention(64)(d33)
    #se3 = ChannelAttention(64, 8)(conv3)
    #se3 = SpatialAttention(7)(se3)

    merge7 = keras.layers.concatenate([sa3,up7], axis = 3)
    conv7 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    up8 = keras.layers.Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv7))

    d21 = keras.layers.Conv2D(21, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same') (conv2)
    d22 = keras.layers.Conv2D(21, (3, 3), dilation_rate=4, activation='relu', kernel_initializer='he_normal', padding='same') (d21)
    d23 = keras.layers.Conv2D(21, (3, 3), dilation_rate=8, activation='relu', kernel_initializer='he_normal', padding='same') (d22)
    sa2 = spatial_attention(32)(d23)
    #se2 = ChannelAttention(32, 8)(conv2)
    #se2 = SpatialAttention(7)(se2)

    merge8 = keras.layers.concatenate([sa2,up8], axis = 3)
    conv8 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    up9 = keras.layers.Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv8))

    d11 = keras.layers.Conv2D(21, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same') (conv1)
    d12 = keras.layers.Conv2D(21, (3, 3), dilation_rate=4, activation='relu', kernel_initializer='he_normal', padding='same') (d11)
    d13 = keras.layers.Conv2D(21, (3, 3), dilation_rate=8, activation='relu', kernel_initializer='he_normal', padding='same') (d12)
    sa1 = spatial_attention(16)(d13)
    #se1 = ChannelAttention(16, 8)(conv1)
    #se1 = SpatialAttention(7)(se1)

    merge9 = keras.layers.concatenate([sa1,up9], axis = 3)
    conv9 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 =keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)
    model2 = keras.models.Model(inputs = [inputs1,inputs2], outputs = conv10)
    return model2

def MYMODEL3():
    inputs1 = keras.layers.Input((96,96, 1))
    inputs2 = keras.layers.Input((192,192, 3))
    #inputs1 = keras.layers.Input((48,48, 1))
    #inputs2 = keras.layers.Input((48,48, 3))
    #c_1x1 = keras.layers.Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs2)
    Up_s1=keras.layers.UpSampling2D(size = (2,2))(inputs1)
    x = keras.layers.concatenate([inputs2,Up_s1],axis=-1)
    #x = keras.layers.concatenate([inputs2,inputs1],axis=-1)
    conv1 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    conv1 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    up7 = keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(pool3))

    d31 = keras.layers.Conv2D(21, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same') (conv3)
    d32 = keras.layers.Conv2D(21, (3, 3), dilation_rate=4, activation='relu', kernel_initializer='he_normal', padding='same') (d31)
    d33 = keras.layers.Conv2D(21, (3, 3), dilation_rate=8, activation='relu', kernel_initializer='he_normal', padding='same') (d32)
    sa3 = spatial_attention(64)(d33)
    #se3 = ChannelAttention(64, 8)(conv3)
    #se3 = SpatialAttention(7)(se3)

    merge7 = keras.layers.concatenate([sa3,up7], axis = 3)
    conv7 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    up8 = keras.layers.Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv7))

    d21 = keras.layers.Conv2D(21, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same') (conv2)
    d22 = keras.layers.Conv2D(21, (3, 3), dilation_rate=4, activation='relu', kernel_initializer='he_normal', padding='same') (d21)
    d23 = keras.layers.Conv2D(21, (3, 3), dilation_rate=8, activation='relu', kernel_initializer='he_normal', padding='same') (d22)
    sa2 = spatial_attention(32)(d23)
    #se2 = ChannelAttention(32, 8)(conv2)
    #se2 = SpatialAttention(7)(se2)

    merge8 = keras.layers.concatenate([sa2,up8], axis = 3)
    conv8 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    up9 = keras.layers.Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv8))

    d11 = keras.layers.Conv2D(21, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same') (conv1)
    d12 = keras.layers.Conv2D(21, (3, 3), dilation_rate=4, activation='relu', kernel_initializer='he_normal', padding='same') (d11)
    d13 = keras.layers.Conv2D(21, (3, 3), dilation_rate=8, activation='relu', kernel_initializer='he_normal', padding='same') (d12)
    sa1 = spatial_attention(16)(d13)
    #se1 = ChannelAttention(16, 8)(conv1)
    #se1 = SpatialAttention(7)(se1)

    merge9 = keras.layers.concatenate([sa1,up9], axis = 3)
    conv9 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 =keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)
    model3 = keras.models.Model(inputs = [inputs1,inputs2], outputs = conv10)
    return model3

def MYMODEL4():
    inputs1 = keras.layers.Input((192,192, 1))
    inputs2 = keras.layers.Input((384,384, 3))
    #inputs1 = keras.layers.Input((48,48, 1))
    #inputs2 = keras.layers.Input((96,96, 3))
    #c_1x1 = keras.layers.Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs2)
    Up_s1=keras.layers.UpSampling2D(size = (2,2))(inputs1)
    x = keras.layers.concatenate([inputs2,Up_s1],axis=-1)
    conv1 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    conv1 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    up7 = keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(pool3))

    d31 = keras.layers.Conv2D(21, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same') (conv3)
    d32 = keras.layers.Conv2D(21, (3, 3), dilation_rate=4, activation='relu', kernel_initializer='he_normal', padding='same') (d31)
    d33 = keras.layers.Conv2D(21, (3, 3), dilation_rate=8, activation='relu', kernel_initializer='he_normal', padding='same') (d32)
    sa3 = spatial_attention(64)(d33)
    #se3 = ChannelAttention(64, 8)(conv3)
    #se3 = SpatialAttention(7)(se3)

    merge7 = keras.layers.concatenate([sa3,up7], axis = 3)
    conv7 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    up8 = keras.layers.Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv7))

    d21 = keras.layers.Conv2D(21, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same') (conv2)
    d22 = keras.layers.Conv2D(21, (3, 3), dilation_rate=4, activation='relu', kernel_initializer='he_normal', padding='same') (d21)
    d23 = keras.layers.Conv2D(21, (3, 3), dilation_rate=8, activation='relu', kernel_initializer='he_normal', padding='same') (d22)
    sa2 = spatial_attention(32)(d23)
    #se2 = ChannelAttention(32, 8)(conv2)
    #se2 = SpatialAttention(7)(se2)

    merge8 = keras.layers.concatenate([sa2,up8], axis = 3)
    conv8 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    up9 = keras.layers.Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv8))

    d11 = keras.layers.Conv2D(21, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same') (conv1)
    d12 = keras.layers.Conv2D(21, (3, 3), dilation_rate=4, activation='relu', kernel_initializer='he_normal', padding='same') (d11)
    d13 = keras.layers.Conv2D(21, (3, 3), dilation_rate=8, activation='relu', kernel_initializer='he_normal', padding='same') (d12)
    sa1 = spatial_attention(16)(d13)
    #se1 = ChannelAttention(16, 8)(conv1)
    #se1 = SpatialAttention(7)(se1)

    merge9 = keras.layers.concatenate([sa1,up9], axis = 3)
    conv9 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 =keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)
    model4 = keras.models.Model(inputs = [inputs1,inputs2], outputs = conv10)
    return model4

"""Train Models"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import optimizers

epochs=50

"""Stage 1"""

C2V,C3V,C4V,C5V,Y2V,Y3V,Y4V,Y5V=DataGenV()

C5,Y5=DataGen1()

Adam = optimizers.Adam(learning_rate=0.0001,  beta_1=0.9, beta_2=0.99)

##       Stage_1      ##
Adam = optimizers.Adam(lr=0.0001,  beta_1=0.9, beta_2=0.99)
model1=MYMODEL1()
model1.summary()
model1.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
history1 = model1.fit(C5,Y5,validation_data=(C5V, Y5V),batch_size=2,epochs=epochs)
model1.save_weights("/content/drive/MyDrive/PMED_Net/final/trained_weights/proposed_model_stage_1.h5")
result1 = model1.predict(C5)
result1V = model1.predict(C5V)

plot_graph(history1)

"""Stage 2"""

C4,Y4=DataGen2()

##       Stage_2      ##
model2=MYMODEL2()
model2.summary()
model2.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
history2 = model2.fit([result1,C4],Y4,validation_data=([result1V,C4V], Y4V),batch_size=2,epochs=epochs)
model2.save_weights("/content/drive/MyDrive/PMED_Net/final/trained_weights/proposed_model_stage_2.h5")
result2 = model2.predict([result1,C4])
result2V = model2.predict([result1V,C4V])

plot_graph(history2)

"""Stage 3"""

C3,Y3=DataGen3()

##       Stage_3      ##
model3=MYMODEL3()
model3.summary()
model3.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
history3 = model3.fit([result2,C3],Y3,validation_data=([result2V,C3V], Y3V),batch_size=2,epochs=epochs)
model3.save_weights("/content/drive/MyDrive/PMED_Net/final/trained_weights/proposed_model_stage_3.h5")
result3 = model3.predict([result2,C3])
result3V = model3.predict([result2V,C3V])

plot_graph(history3)

"""Stage 4"""

C2,Y2=DataGen4()

##       Stage_4     ##
model4=MYMODEL4()
model4.summary()
model4.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
history4 = model4.fit([result3,C2],Y2,validation_data=([result3V,C2V], Y2V),batch_size=2,epochs=epochs)
model4.save_weights("/content/drive/MyDrive/PMED_Net/final/trained_weights/proposed_model_stage_4.h5")
result4 = model4.predict([result3,C2])
result4V = model4.predict([result3V,C2V])

plot_graph(history4)

plot_level_train_graph(history1,history2,history3, history4)

plot_level_val_graph(history1,history2,history3, history4)

"""save history"""

import pandas as pd

# convert the history.history dict to a pandas DataFrame:
hist_df = pd.DataFrame(history1.history)

# save to json:
hist_json_file = '/content/drive/MyDrive/PMED_Net/final/history/proposed_model_stage_1_histoy.json'
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# or save to csv:
hist_csv_file = '/content/drive/MyDrive/PMED_Net/final/history/proposed_model_stage_1_histoy.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

#----------Stage # 2----------#
hist_df = pd.DataFrame(history2.history)

# save to json:
hist_json_file = '/content/drive/MyDrive/PMED_Net/final/history/proposed_model_stage_2_histoy.json'
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# or save to csv:
hist_csv_file = '/content/drive/MyDrive/PMED_Net/final/history/proposed_model_stage_2_histoy.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


#----------Stage # 3----------#
hist_df = pd.DataFrame(history3.history)

# save to json:
hist_json_file = '/content/drive/MyDrive/PMED_Net/final/history/proposed_model_stage_3_histoy.json'
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# or save to csv:
hist_csv_file = '/content/drive/MyDrive/PMED_Net/final/history/proposed_model_stage_3_histoy.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


#----------Stage # 2----------#
hist_df = pd.DataFrame(history4.history)

# save to json:
hist_json_file = '/content/drive/MyDrive/PMED_Net/final/history/proposed_model_stage_4_histoy.json'
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# or save to csv:
hist_csv_file = '/content/drive/MyDrive/PMED_Net/final/history/proposed_model_stage_4_histoy.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

"""Testing"""

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

Threshold=0.5

model1=MYMODEL1()
model2=MYMODEL2()
model3=MYMODEL3()
model4=MYMODEL4()

model1.load_weights("/content/drive/MyDrive/PMED_Net/final/trained_weights/proposed_model_stage_1.h5")
model2.load_weights("/content/drive/MyDrive/PMED_Net/final/trained_weights/proposed_model_stage_2.h5")
model3.load_weights("/content/drive/MyDrive/PMED_Net/final/trained_weights/proposed_model_stage_3.h5")
model4.load_weights("/content/drive/MyDrive/PMED_Net/final/trained_weights/proposed_model_stage_4.h5")

C2T,C3T,C4T,C5T,Y2T,Y3T,Y4T,Y5T=DataGenT()

result1 = model1.predict(C5T)
result1[np.where(result1[:,:,:,0]>.5)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result1,Y5T)
result2 = model2.predict([result1,C4T])
result2[np.where(result2[:,:,:,0]>0.5)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result2,Y4T)
result3 = model3.predict([result2,C3T])
result3[np.where(result3[:,:,:,0]>0.5)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result3,Y3T)
result4 = model4.predict([result3,C3T])
result4[np.where(result4[:,:,:,0]>0.5)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result4,Y3T)

"""Predictions"""

prediction_dataset = []
for infile in sorted(glob.glob('/content/drive/MyDrive/PMED_Net/final/predictions/dataset/*')):
    prediction_dataset.append(infile)

def DataGen():
    img_ = []
    #mask_  = []
    for i in range(len(prediction_dataset)):
        image = cv2.imread(prediction_dataset[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image, (height//4,width//4), interpolation = cv2.INTER_AREA)
        image=image/255
        #mask = cv2.imread(mask_id[i],0)
        #mask = cv2.resize(mask, (height//4,width//4), interpolation = cv2.INTER_AREA)
        #mask=mask/255
        #F = np.expand_dims(mask, axis=-1)
        img_.append(image)
        #mask_.append(F)
    img_ = np.array(img_)
    #mask_  = np.array(mask_)
    return img_
    #mask_

dataset = DataGen()

import os

### Save the Results ###
P1="/content/drive/MyDrive/PMED_Net/final/predictions/proposed_model/s1/"
P2="/content/drive/MyDrive/PMED_Net/final/predictions/proposed_model/s2/"
P3="/content/drive/MyDrive/PMED_Net/final/predictions/proposed_model/s3/"
P4="/content/drive/MyDrive/PMED_Net/final/predictions/proposed_model/s4/"

for i in range(6):
    cv2.imwrite(os.path.join(P1 , str(i)+".png"),result1[i,:,:,0]*255)
    cv2.imwrite(os.path.join(P2 , str(i)+".png"),result2[i,:,:,0]*255)
    cv2.imwrite(os.path.join(P3 , str(i)+".png"),result3[i,:,:,0]*255)
    cv2.imwrite(os.path.join(P4 , str(i)+".png"),result4[i,:,:,0]*255)

