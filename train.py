from model import Model
from data import DataSet
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, Callback

import os

model_type = 'lrcn'

batch_size = 32
epochs = 50

sequence_length = 20
frame_shape = (120, 120, 3)

def train(model_type, batch_size, sequence_length, frame_shape):
    model = Model(model_type, input_shape = (20, 120, 120, 3))
    data = DataSet(sequence_length, frame_shape)

    checkpoint = ModelCheckpoint(filepath = os.path.join('checkpoints', (model_type + '-.{epoch:03d}-{val_loss:.3f}.hdf5'), verbose = 1, save_best_only = True))
    tensorBoard = TensorBoard(log_dir = os.path.join('checkpoints', 'logs', model_type))

    if 'parallel' not in model_type:
        tri_generator = data.generator('train', 'fn', batch_size)
        val_generator = data.generator('test', 'fn', batch_size)
    else:
        tri_generator = data.parallel_generator('train', batch_size)
        val_generator = data.parallel_generator('test', batch_size)

    model.model.fit_generator(generator = tri_generator, 
                              steps_per_epoch = data.size('train') // batch_size, 
                              epochs = epochs,
                              verbose = 1,
                              callbacks = [tensorBoard, checkpoint],
                              validation_data = val_generator, 
                              validation_steps = 32, 
                              workers = 1)

