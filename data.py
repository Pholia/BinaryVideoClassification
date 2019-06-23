import os
import random

import numpy as np
import cv2

from keras.utils import to_categorical

def norm(image_sequence):
    image_sequence -= np.mean(image_sequence)
    image_sequence /= np.max(image_sequence)
    return image_sequence

def load_clip(clip_path):
    cap = cv2.VideoCapture(clip_path)
    frames = []
    while (1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
            
    cap.release()
    return frames

class DataSet:

    def __init__(self, num_frame = 20, frame_shape = (120, 120, 3), parallel_model = False):
        self.num_frame = num_frame
        self.frame_shape = frame_shape

        self.data = {}
        self.data['positive'] = {}
        self.data['negative'] = {}

        classes = os.listdir('data')
        for class_name in classes:
            if class_name not in ('positive','negative'):
                raise Exception
            
            self.data[class_name]['or_train'] = ['data/' + class_name + '/train/' + cp for cp in os.listdir('data/' + class_name + '/train') if 'or' in cp]
            self.data[class_name]['or_test'] = ['data/' + class_name + '/test/' + cp for cp in os.listdir('data/' + class_name + '/test') if 'or' in cp]
            self.data[class_name]['fn_train'] = ['data/' + class_name + '/train/' + cp for cp in os.listdir('data/' + class_name + '/train') if 'fn' in cp]
            self.data[class_name]['fn_test'] = ['data/' + class_name + '/test/' + cp for cp in os.listdir('data/' + class_name + '/test') if 'fn' in cp]
        
        self.create_label()
    
    def size(self, op_type):
        return len(self.data['positive'][op_type] + self.data['negative'][op_type])

    # Helper func
    def create_label(self):
        self.labels = {}
        self.labels['negative'] = to_categorical(0, 2)
        self.labels['positive'] = to_categorical(1, 2)
    
    def choose_label(self, sample_path):
        class_name = sample_path.split('/')[1]
        return self.labels[class_name]
    
    def process_frame(self, frame):
        if frame.shape != self.frame_shape:
            raise Exception
        x = frame.astype(np.float32)
        return x
    
    def build_image_sequence(self, sample_path):
        clip = load_clip(sample_path)
        if len(clip) != self.num_frame:
            raise Exception
        
        return [self.process_frame(frame) for frame in clip]
    
    def build_parallel_image_sequence(self, or_sample_path, fn_sample_path):
        or_clip = load_clip(or_sample_path)
        fn_clip = load_clip(fn_sample_path)

        if (len(or_clip) != self.num_frame) or (len(fn_clip) != self.num_frame - 1):
            raise Exception
        
        return [self.process_frame(frame) for frame in or_clip], [self.process_frame(frame) for frame in fn_clip]
    
    def print_data(self, op_type, data_arr):
        print(data_arr)
        print(op_type + ' ' + str(len(data_arr)) + ' (* 2 if parallel) samples')
    
    def generator(self, op_type, data_type, batch_size):
        data_arr = np.array(self.data['positive'][data_type + '_' + op_type] + self.data['negative'][data_type + '_' + op_type])
        self.print_data(op_type, data_arr)

        while(1):
            np.random.shuffle(data_arr)
            x, y = [], []

            for _ in range(batch_size):
                sample_path = random.choice(data_arr)
                image_sequence = self.build_image_sequence(sample_path)
            
                x.append(norm(image_sequence))
                y.append(self.choose_label(sample_path))
        
            yield np.array(x), np.array(y)
    
    def parallel_generator(self, op_type, batch_size):
        or_data_arr = np.array(self.data['positive']['or_' + op_type] + self.data['negative']['or_' + op_type])
        self.print_data(op_type, or_data_arr)

        while(1):
            or_data_arr = np.random.shuffle(or_data_arr)
            x1, x2, y = [], [], []

            for _ in range(batch_size):
                or_sample_path = random.choice(or_data_arr)
                fn_sample_path = or_sample_path.replace('or', 'fn')

                or_image_sequence, fn_image_sequence = self.build_parallel_image_sequence(or_sample_path, fn_sample_path)
                
                x1.append(norm(or_image_sequence))
                x2.append(norm(fn_image_sequence))
                y.append(self.choose_label(or_sample_path))
            
            yield [np.array(x1), np.array(x2)], np.array(y)
