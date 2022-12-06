from PIL import Image
import numpy as np 
import tensorflow as tf
from datasets import ImageTransform, VOCDataset
import os  
import time

def g(n):
    def f(x):
        return n * x
    return f

print(g(3)(2))




