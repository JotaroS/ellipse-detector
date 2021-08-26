from network import EllipseNet
from dataset import EllipseDataset
from tensorflow.keras import backend
from tensorflow import keras

import time
import cv2 as cv
import numpy as np

def preview(filename='./cropped/0000000000.jpg', center = [10, 20]):
    thickness  = 0
    cen        = (int(center[0])+64, int(center[1])+64)
    radius     = 3
    axisLength = (radius, radius)
    angle      = 0
    color      = (0, 255, 0)
    
    src = cv.imread(filename)
    ret = cv.ellipse(src, cen, axisLength, angle, 0, 360, color, -1)
    cv.imshow("preview", ret)
    cv.waitKey(0)

def load_model():
    model = keras.models.load_model('my_model')
    return model

def preview_slideshow(pages = 20):
    model = load_model()
    dataset = EllipseDataset()
    for i in range(0, pages):
        X,f       = dataset.load_data(i)
        y_pred    = model.predict(X) # approx. 0.022[sec] = 45[Hz]
        preview(f, center = y_pred[0]);

def main():
    preview_slideshow()

if __name__ == '__main__':
    # backend.set_learning_phase(0)
    main()


# TODO:
# https://www.reddit.com/r/learnmachinelearning/comments/9yom7p/how_to_reduce_prediction_time_of_keras_cnn/
# Try to set learning_phase to 0. it tells Keras that you will be using predict only and not teaching your CNN. keras.backend.set_learning_phase(0)