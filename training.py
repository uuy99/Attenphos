from tensorflow.keras import * 
from tensorflow.keras.layers import Layer
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, AveragePooling1D
from keras.layers.pooling import GlobalAveragePooling2D,GlobalAveragePooling1D
from keras.layers import Input, merge, Flatten,Reshape

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

import tensorflow.python.keras.engine

from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.regularizers import l2
import keras.backend as K

from tensorflow.keras.layers import Layer
from keras.layers import Conv1D,Conv2D, MaxPooling2D,concatenate
from tensorflow.keras.layers import * 
from tensorflow.keras.layers import Layer
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.optimizers import Adam,SGD
from cProfile import label
import csv
import numpy as np
import tensorflow.keras.utils as kutils
from tensorflow.python.keras.initializers import get
from tensorflow import keras
import script.model as model



def getMatrixLabel(positive_position_file_name,sites, window_size=49  , empty_aa = '*'):
    prot = []  # list of protein name
    pos = []  # list of position with protein name
    rawseq = []
    all_label = []

    short_seqs = []
    half_len = (window_size - 1) / 2

    with open(positive_position_file_name, 'r') as rf:
        reader = csv.reader(rf)
            
        for row in reader:
            # print("row[0]:",int(row[0]))
            position = int(row[2])
            sseq = row[3]
            rawseq.append(row[3])
            center = sseq[position - 1]
            if center in sites:
                all_label.append(int(row[0]))
                # print("length of all_label",len(all_label))
                prot.append(row[1])
                pos.append(row[2])

                
                if position - half_len > 0:
                    start = position - half_len
                    start = int(start)
                    position = int(position)
                    left_seq = sseq[start - 1:position - 1]
                else:
                    
                    left_seq = sseq[0:position - 1]

                end = len(sseq)
                if position + half_len < end:
                    end = position + half_len
                    end = int(end)
                right_seq = sseq[position:end]

                if len(left_seq) < half_len:
                    nb_lack = half_len - len(left_seq)
                    nb_lack = int(nb_lack)
                    left_seq = ''.join([empty_aa for count in range(nb_lack)]) + left_seq

                if len(right_seq) < half_len:
                    nb_lack = half_len - len(right_seq)
                    nb_lack = int(nb_lack)
                    right_seq = right_seq + ''.join([empty_aa for count in range(nb_lack)])
                shortseq = left_seq + center + right_seq
                short_seqs.append(shortseq)
        targetY = kutils.to_categorical(all_label)
        letterDict = {}
        letterDict["A"] = 0
        letterDict["C"] = 1
        letterDict["D"] = 2
        letterDict["E"] = 3
        letterDict["F"] = 4
        letterDict["G"] = 5
        letterDict["H"] = 6
        letterDict["I"] = 7
        letterDict["K"] = 8
        letterDict["L"] = 9
        letterDict["M"] = 10
        letterDict["N"] = 11
        letterDict["P"] = 12
        letterDict["Q"] = 13
        letterDict["R"] = 14
        letterDict["S"] = 15
        letterDict["T"] = 16
        letterDict["V"] = 17
        letterDict["W"] = 18
        letterDict["Y"] = 19
        letterDict["*"] = 20
        # letterDict["?"] = 21
        Matr = np.zeros((len(short_seqs), window_size))
        samplenumber = 0
        for seq in short_seqs:
            AANo = 0
            for AA in seq:
                if AA not in  letterDict:
                    # AANo += 1
                    continue
                Matr[samplenumber][AANo] = letterDict[AA]
                AANo = AANo+1
            samplenumber = samplenumber + 1
    # print('data process finished')
    print("matr.shape",Matr.shape)
    return Matr, targetY ,all_label
    
class Position_Embedding(Layer):

    def __init__(self, size=None, mode='concat', **kwargs):

        self.size = size  

        self.mode = mode

        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):

        if (self.size == None) or (self.mode == 'concat'):

            self.size = int(x.shape[-1])


        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)

        position_j = K.expand_dims(position_j, 0)

        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  

        position_i = K.expand_dims(position_i, 2)

        position_ij = K.dot(position_i, position_j)
#         print(K.cos(position_ij).shape, K.sin(position_ij).shape)

        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
#         print(position_ij.shape)

        if self.mode == 'sum':

            return position_ij + x

        elif self.mode == 'concat':

            return K.concatenate([position_ij, x], 2)
        
    def compute_output_shape(self, input_shape):

        if self.mode == 'sum':

            return input_shape

        elif self.mode == 'concat':

            return (input_shape[0], input_shape[1], input_shape[2] + self.size)
            
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size': self.size,
            'mode': self.mode,
        })
        return config
    
class Self_Attention(Layer):     
    def __init__(self, output_dim, **kwargs):        
        self.output_dim = output_dim        
        super(Self_Attention, self).__init__(**kwargs)   
        
    def build(self, input_shape):        
                    
        self.kernel = self.add_weight(name='kernel',                                      
                                      shape=(3,input_shape[2], self.output_dim),                                      
                                      initializer='uniform',                                      
                                      trainable=True)     
        
        super(Self_Attention, self).build(input_shape)  
           
    def call(self, x):        
        WQ = K.dot(x, self.kernel[0])        
        WK = K.dot(x, self.kernel[1])        
        WV = K.dot(x, self.kernel[2])  
        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))               
        QK = QK / (self.output_dim**0.5)        
        QK = K.softmax(QK)         
        V = K.batch_dot(QK,WV)         
        return V     
    
    def compute_output_shape(self, input_shape):         
        return (input_shape[0],input_shape[1],self.output_dim)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
        })
        return config    
        
    
def getmatrix(x,w,L):
    matrix=[]
    n=[]
    for j in range(w):
        n=tf.reshape(x[:,:,j*w:(j+1)*w],(-1,L*w))
        print(n.shape)
        matrix.append(n)
    matrix = tf.concat(matrix,axis=1)
    matrix=tf.reshape(matrix,(-1,L*w,w)) 
    return matrix

def train():
  max_features=21
  batch_size=16
#   train_file_name="/storage/yq/dataset/PELM_S_data.csv"
  max_features=21

  win1=21
  win2=33
  win3=51
  dropout_dense=0.3
  train_file_name="/home/2021/21yq/attenphos/dataset/PELM_Y_data.csv"
  
  x_train1,y_train,z1 = getMatrixLabel(train_file_name, ('Y'), win1)
  x_train2, _,z = getMatrixLabel(train_file_name, ('Y'), win2)
  x_train3, _,z = getMatrixLabel(train_file_name, ('Y'), win3)
  
  inputs1 = Input(shape=(win1,), dtype='int32')
  x1 = Embedding(max_features,16)(inputs1)
  
  x1=Position_Embedding( mode='sum')(x1)
  print(x1.shape)
  self_number=16
  x1 = Self_Attention(self_number)(x1)
  # x1=getmatrix(x1,4,win1)
  x1=tf.reshape(x1,(-1,win1*4,4))
  print(x1.shape)
  
  
  
  
  inputs2 = Input(shape=(win2,), dtype='int32')
  x2 = Embedding(max_features,16)(inputs2)
  
  x2=Position_Embedding( mode='sum')(x2)
  self_number=16
  x2 = Self_Attention(self_number)(x2)
  
  # x2=getmatrix(x2,4,win2)
  x2=tf.reshape(x2,(-1,win2*4,4))
  print(x2.shape)
  
  
  
  inputs3 = Input(shape=(win3,), dtype='int32')
  x3 = Embedding(max_features,16)(inputs3)
  
  x3=Position_Embedding( mode='sum')(x3)
  print(x1.shape)
  self_number=16
  x3 = Self_Attention(self_number)(x3)
  
  # x3=getmatrix(x3,4,win3)
  x3=tf.reshape(x3,(-1,win3*4,4))
  
  print(x3.shape)
  
  init_form = 'RandomUniform'
  weight_decay = 0.0001
  x1.shape
  
  x1 = Conv1D(64, 12,kernel_initializer = init_form,strides=4,activation='relu',
                          padding='valid',
                          use_bias=False,
                          kernel_regularizer=l2(weight_decay))(x1)
  
  
  x2 = Conv1D(64, 12,kernel_initializer = init_form,strides=4,activation='relu',
                          padding='valid',
                          use_bias=False,
                          kernel_regularizer=l2(weight_decay))(x2)
  x3 = Conv1D(64, 12,kernel_initializer = init_form,strides=4,activation='relu',
                          padding='valid',
                          use_bias=False,
                          kernel_regularizer=l2(weight_decay))(x3)
                          
  x = keras.layers.concatenate((x1,x2,x3),axis=1)
  
  
  x1=Flatten()(x)
  x=Dense(256, activation='sigmoid')(x1)
  x=Dense(84, activation='sigmoid')(x)
  x=Dense(2,activation='softmax')(x)
  model = Model(inputs=[inputs1,inputs2,inputs3], outputs=x,name="res-net")
  print(model.summary())
  opt = Adam(learning_rate=0.0001, decay=0.00001)
  loss = 'categorical_crossentropy'
  model.compile(loss=loss,
              optimizer=opt,
              metrics=['accuracy','AUC','Precision'], 
               )
  print('Train...')
  history = model.fit([x_train1,x_train2,x_train3],y_train,
          batch_size=1,
          epochs=3  ,
          validation_split=0.2)
  modelname = "/home/2021/21yq/attenphos/model/auc_Y_no_attention/auc_Y_no_attention_"
  model.save(modelname+'.h5')
  

if __name__ =="__main__":
    train()