import tensorflow as tf
from   tensorflow.keras                       import Model
from   tensorflow.keras                       import layers
from   tensorflow.keras.layers                import Conv2D, MaxPooling2D, Input
from   tensorflow.keras.layers                import Dense, Dropout, Flatten
from   tensorflow.keras.layers                import BatchNormalization
from   tensorflow.keras.applications.resnet50 import ResNet50

# CNN Block with convolutional layers and processing
class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size=(3, 3)):
        super(CNNBlock, self).__init__()
        self.convIn = layers.Conv2D(out_channels, kernel_size, padding = 'same')
        self.conv   = layers.Conv2D(out_channels, kernel_size)
        self.bn     = layers.BatchNormalization()
        self.mp     = layers.MaxPooling2D(pool_size = (2, 2))
        self.dr     = layers.Dropout(0.25)
    
    def call(self, input_tensor, training = False):
        x = self.convIn(input_tensor)
        x = tf.nn.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        x = tf.nn.relu(x)
        x = self.mp(x)
        x = self.dr(x)
        return x

# Prediction model
class EllipseNet(Model):
    def __init__(self, num_classes = 2):
        super(EllipseNet, self).__init__()
        self.block1  = CNNBlock(16)
        self.block2  = CNNBlock(32)
        self.block3  = CNNBlock(64)
        self.flatten = layers.Flatten()
        self.dense   = layers.Dense(num_classes)

    def call(self, input_tensor, training = False):
        x = self.block1(input_tensor, training = training)
        x = self.block2(x, training = training)
        x = self.block3(x, training = training)
        x = self.flatten(x)
        return self.dense(x)

class EllipseResNet(Model):
    def __init__(self, num_classes = 2):
        super(EllipseResNet, self).__init__()
        input_tensor = Input(shape=(128, 128, 3))
        self.resnet  = ResNet50(input_tensor = input_tensor,weights = 'imagenet', include_top = False)
        self.flatten = layers.Flatten(input_shape=self.resnet.output_shape[1:])
        self.dense   = layers.Dense(num_classes)

    def call(self, input_tensor, training = False):
        x = self.resnet(input_tensor, training = training)
        x = self.flatten(x)
        return self.dense(x)

# model = EllipseNet()
# model.compile(optimizer = 'adam', loss='mean_squared_error')
# model.build(input_shape=(None,128, 128, 1))
# model.summary()