import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


# load images to build and train the model
#                       ....                                     /    img1.jpg
#             test      Hand            patient0000   positive  --   img2.png
#           /                /                         \    .....
#   Dataset   -         Elbow  ------   patient0001
#           \ train               \         /                           img1.png
#                       Shoulder        patient0002     negative --      img2.jpg
#                       ....                   \
#
def load_path(path):
    """
    load X-ray dataset
    """
    dataset = []
    for folder in os.listdir(path):
        folder = path + '/' + str(folder)
        if os.path.isdir(folder):
            for body in os.listdir(folder):
                path_p = folder + '/' + str(body)
                for id_p in os.listdir(path_p):
                    patient_id = id_p
                    path_id = path_p + '/' + str(id_p)
                    for lab in os.listdir(path_id):
                        if lab.split('_')[-1] == 'positive':
                            label = 'fractured'
                        elif lab.split('_')[-1] == 'negative':
                            label = 'normal'
                        path_l = path_id + '/' + str(lab)
                        for img in os.listdir(path_l):
                            img_path = path_l + '/' + str(img)
                            dataset.append(
                                {
                                    'label': body,
                                    'image_path': img_path
                                }
                            )
    return dataset


# load data from path
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
image_dir = "D://reaserch//data//MURA-v1.1//Dataset"
data = load_path(image_dir)
labels = []
filepaths = []
print(image_dir)