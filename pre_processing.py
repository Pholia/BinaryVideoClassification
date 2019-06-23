import cv2
from TF_Flownet2.src.flownet2.single_chunk_convert import FN2

fn = FN2(memory_fraction = 1., model = 'flownet_c')