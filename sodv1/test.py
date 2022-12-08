from PIL import Image
import numpy as np 
import tensorflow as tf
from datasets import ImageTransform, VOCDataset
import os  
import time

a = np.array([0,0,1,0,0])
print(np.argmax(a))