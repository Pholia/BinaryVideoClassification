import os
import random

import numpy as np
import cv2

from keras.utils import to_categorical

def norm(clip):
    clip -= np.mean(clip)
    clip /= np.max(clip)
    return clip

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

    def __init__(self, num_frame = 20, frame_shape = (50, 50, 3)):
        self.num_frame = num_frame
        self.frame_shape = frame_shape

        self.data = {}
        self.data['positive'] = {}
        self.data['negative'] = {}

        classes = os.listdir('data')
        for class_name in classes:
            if class_name not in ('positive','negative'):
                raise Exception
            
            self.data[class_name]['train'] = os.listdir('data/' + class_name + '/train')
            self.data[class_name]['test'] = os.listdir('data/' + class_name + '/test')
        
        self.create_label()

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
    
    def generator(self, op_type, data_type, batch_size):
        data_arr = np.array(self.data['positive'][op_type] + self.data['negative'][op_type])

        while(1):
            np.random.shuffle(data_arr)
            x, y = [], []

            for _ in range(batch_size):
                sample_path = random.choice(data_arr)
                
                if data_type == 'frame':
                    image_sequence = self.build_image_sequence(sample_path)
                # TODO: feature data type
            
                x.append(norm(image_sequence))
                y.append(self.choose_label(sample_path))
        
        yield np.array(x), np.array(y)