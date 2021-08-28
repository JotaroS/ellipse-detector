import numpy  as np
import glob
import json

from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img

class EllipseDataset():
    def __init__(self):
        pass
    
    def load_dataset(self, DEBUG = False, color_mode = 'rgb'):
        X = []
        Y = []
        
        # X
        files = glob.glob('./cropped/*.jpg')
        if DEBUG is True:files = files[:100]
        files.sort()
        for f in files:
            temp_img = load_img(f, color_mode=color_mode)
            arr      = img_to_array(temp_img)
            X.append(arr)
        X = np.asarray(X, dtype=np.float)
        X = X.astype('float32')
        X = X / 255.0
        
        # Y
        with open('labels.json') as f:
            data = json.load(f)
        if DEBUG is True: data = data[:100]

        for d in data:
            y = [(d['center'][0]-64), (d['center'][1]-64)]
            Y.append(y)
        Y = np.asarray(Y)
        Y = Y.astype('float32')

        return X, Y
    
    def load_data(self, index = 0):
        files = glob.glob('./cropped/*.jpg')
        f = files[index]
        X = []
        temp_img = load_img(f, color_mode=color_mode)
        arr      = img_to_array(temp_img)
        X.append(arr)
        X = np.asarray(X)
        X = X.astype('float32')
        X = X / 255.0
        return X, f

# dataset = EllipseDataset()
# X = dataset.load_data()
# print(X)