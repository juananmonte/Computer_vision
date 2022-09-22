from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Flatten, Dropout,Dense

class SimpleCNN:

    @staticmethod

    def build(width, height, depth, classes):
        #Create the input layer
        Inlayer = Input(shape=(height, width, depth))

        #first CONV -> RELU -> BN layers
        x = Conv2D(32, (3,3), strides=(2,2), padding="same")(Inlayer)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)

        #Second CONV-> RELU -> BN layer
        x = Conv2D(32, (3,3), strides=(2,2), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x) 

        #FC layer with two Dense 
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        #Add the classifier
        x = Dense(classes)(x)
        output  = Activation("softmax")(x)
        
        #Get te constructed network achitecture

        model = Model(inputs=Inlayer, outputs=output, name="adversarial_mnist")

        return model
