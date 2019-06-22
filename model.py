from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop, Nadam
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D, MaxPooling2D)

class Model:

    def __init__(self, model_type, input_shape = (20, 120, 120 ,3)):
        self.input_shape = input_shape

        if model_type == 'lrcn': 
            self.model = self.lrcn()
        
        optimizer = Adam(lr=1e-5, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
        
        print(self.model.summary())

    def lrcn(self):
        model = Sequential()

        model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2,2), activation='relu', padding='same'), input_shape = self.input_shape))
        # model.add(TimeDistributed(Conv2D(32, (3, 3),
        # kernel_initializer="he_normal", activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
        # model.add(TimeDistributed(Conv2D(64, (3, 3),
        # padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu')))
        # model.add(TimeDistributed(Conv2D(128, (3, 3),
        # padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu')))
        # model.add(TimeDistributed(Conv2D(256, (3, 3),
        # padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu')))
        # model.add(TimeDistributed(Conv2D(256, (3, 3),
        # padding='same', activation='relu')))
        # model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        
        model.add(TimeDistributed(Flatten()))

        model.add(Dropout(0.5))
        model.add(LSTM(512, return_sequences=False, dropout=0.5))
        model.add(Dense(1024, activation='relu'))
        
        # model.add(Dropout(0.5))
        # model.add(Dense(512, activation='relu'))
        
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        return model