from network import EllipseNet
from dataset import EllipseDataset

import time

def load_model():
    model = EllipseNet()
    model.load_weights('my_model')
    return model

def main():
    model = load_model()
    
    dataset = EllipseDataset()
    X       = dataset.load_data()
    y_pred  = model.predict(X) # approx. 0.022[sec] = 45[Hz]
    
    print(y_pred)

if __name__ == '__main__':
    main()


# TODO:
# https://www.reddit.com/r/learnmachinelearning/comments/9yom7p/how_to_reduce_prediction_time_of_keras_cnn/
# Try to set learning_phase to 0. it tells Keras that you will be using predict only and not teaching your CNN. keras.backend.set_learning_phase(0)