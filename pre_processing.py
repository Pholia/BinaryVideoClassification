import cv2
import numpy as np

from TF_Flownet2.src.flownet2.single_chunk_convert import FN2

fn = FN2(memory_fraction = 1., model = 'flownet_c')

def flownet(original):
    feature = []
    
    prev = original[0]
    for frame in original[1:]:
        feature.append(fn.compute_feature(prev, frame))
        prev = frame
    
    return feature

