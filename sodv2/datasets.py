import os
import pandas as pd
from PIL import Image
import numpy as np

class VOCDataset:
    def __init__(
        self, img_dir, label_dir, S=13, C=20, transform=None,
    ):
        self.img_list = os.listdir(img_dir)
        self.label_list = os.listdir(label_dir)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.C = C

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.label_list[index])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])


        img_path = os.path.join(self.img_dir, self.img_list[index])
        image = Image.open(img_path)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        cell_bboxes = np.zeros((self.S, self.S, 5 + self.C ))
        for box in boxes:
            class_label, x, y, width, height = box
            class_label = int(class_label)

            # i,j represents the cell row and cell column that the bounding box belong to
            i, j = int(self.S * y), int(self.S * x)
            
            #cordinates relative to the cell
            x_cell, y_cell = x, y
            width_cell, height_cell = width, height

            # If no object already found for specific cell i,j
            # This means we restrict to one object per cell
            if cell_bboxes[i, j, 0] == 0:
                # Set that there exists an object
                cell_bboxes[i, j, 0] = 1

                # Box coordinates
                box_coordinates = np.array(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                cell_bboxes[i, j, 1:5] = box_coordinates

                # Set one hot encoding for class_label
                cell_bboxes[i, j, class_label + 5] = 1
        
        image = np.asarray(image, dtype=np.uint8)
        return image, cell_bboxes

    def __call__(self, index):
        return self.__getitem__(index)

class ImageTransform:
    def __init__(self, transforms) -> None:
        self.transforms = transforms

    def __call__(self, image, bboxes) :
        for transform in self.transforms:
            image, bboxes = transform(image), bboxes
        return image, bboxes

    @staticmethod
    def image_resize(size = (448,448)):
        return lambda image: image.resize(size)
    
