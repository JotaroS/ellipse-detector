import numpy  as np
import glob
import json

from dataset                              import EllipseDataset
from network                              import EllipseNet, EllipseResNet
from sklearn.model_selection              import train_test_split
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img


def train():
    dataset = EllipseDataset()
    X, Y    = dataset.load_dataset(DEBUG=False)

    model   = EllipseResNet()
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics='mean_squared_error')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=111)

    history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data = (X_test, y_test))

    model.save('my_model_resnet', save_format='tf')
    pass

def main():
    train()
    pass

if __name__ == '__main__':
    main()
