import tensorflow as tf
from datasets import VOCDataset, ImageTransform
from model import SODv1
import numpy as np 
from losses import my_loss_v1, my_loss_v3
tf.keras.mixed_precision.set_global_policy('mixed_float16')

images_path = "../../../datasets/voc2012/images"
labels_path = "../../../datasets/voc2012/labels"

t1 = ImageTransform.image_resize((448,448))
transform = ImageTransform([t1])
dataset = VOCDataset(images_path, labels_path, 7, 20, transform=transform)

X = []
Y = []
for i in range(1600):
    image, cell_bboxes = dataset[i]
    X.append(image)
    Y.append(cell_bboxes)
X = np.array(X)
Y = np.array(Y)

model = SODv1((448,448,3), 7, 20)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001), loss = my_loss_v3)
model.fit(X, Y, batch_size = 32, epochs = 100, verbose = 1)
#model.save("./SODv1")

