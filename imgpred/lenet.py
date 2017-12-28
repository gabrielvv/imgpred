from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        """
        INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC
        http://cs231n.github.io/convolutional-networks/#overview
        https://blog.dataiku.com/deep-learning-with-dss
        https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
        Softmax:
         - https://en.wikipedia.org/wiki/Softmax_function
        FC: fully-connected
         - http://cs231n.github.io/convolutional-networks/#fc
        CONV: convolution
         - http://cs231n.github.io/convolutional-networks/#conv
         - https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
        POOL: pooling layer
         - http://cs231n.github.io/convolutional-networks/#pool
        RELU: Rectified Linear Unit
         - https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        """
        #initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using 'channels first', update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

		# second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten()) # https://keras.io/layers/core/#flatten
        model.add(Dense(500)) # https://keras.io/layers/core/#dense
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
