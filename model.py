
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam

from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, BatchNormalization, Concatenate
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv2D, MaxPooling3D, Conv3D, MaxPooling2D

class Model:

    def __init__(self, model_type, input_shape = (20, 120, 120 ,3)):
        self.input_shape = input_shape

        if model_type == 'lrcn': 
            self.model = self.lrcn()
        if model_type == 'parallel_lrcn': 
            self.model = self.parallel_lrcn()
        
        self.model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-5, decay=1e-6), metrics = ['accuracy'])
        
        print(self.model.summary())
    
    def parallel_lrcn(self):
        or_base_model = self.lrcn()
        fn_base_model = self.lrcn(parallel_fn_input = True)

        or_model = Model(inputs = or_base_model.input, outputs = or_base_model.get_layer(name = 'parallel_dense_output').output)
        fn_model = Model(inputs = fn_base_model.input, outputs = fn_base_model.get_layer(name = 'parallel_dense_output').output)

        merged = Concatenate()([or_model.output, fn_model.output])
        merged = Dropout(0.5)(merged)
        merged = Dense(512, activation='relu')(merged)
        merged = Dropout(0.5)(merged)
        merged = (Dense(2, activation = 'softmax'))(merged)

        model = Model([or_model.input, fn_model.input], merged)

        return model

    def lrcn(self, parallel_fn_input = False):
        model = Sequential()
        if not parallel_fn_input:
            model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2,2), activation='relu', padding='same'), input_shape = self.input_shape))
        else:
            input_shape = self.input_shape
            input_shape[0] -= 1
            model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2,2), activation='relu', padding='same'), input_shape = input_shape))
        
        # model.add(TimeDistributed(Conv2D(32, (3, 3), kernel_initializer="he_normal", activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
        # model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu')))
        # model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu')))
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
        model.add(LSTM(512, return_sequences = False, dropout = 0.5))
        model.add(Dense(1024, activation = 'relu'), name = 'parallel_dense_output')
        
        # model.add(Dropout(0.5))
        # model.add(Dense(512, activation='relu'))
        
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        return model