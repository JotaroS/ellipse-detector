import numpy  as np
import glob
import json

from network                              import EllipseNet
from sklearn.model_selection              import train_test_split
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img


DEBUG = True

def generate_dataset():
    X = []
    Y = []
    
    # X
    files = glob.glob('./cropped/*.jpg')
    if DEBUG is True:files = files[:100]
    files.sort()
    for f in files:
        temp_img = load_img(f, color_mode='grayscale')
        arr      = img_to_array(temp_img)
        X.append(arr)
    X = np.asarray(X)
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


def train():
    X,Y   = generate_dataset()
    model = EllipseNet()
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics='mean_squared_error')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=111)

    history = model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data = (X_test, y_test))

    model.save('my_model', save_format='tf')
    pass

def main():
    train()
    pass

if __name__ == '__main__':
    main()
