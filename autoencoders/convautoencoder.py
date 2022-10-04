from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, \
                                    LeakyReLU, Activation, Flatten, Dense, Reshape, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np

class ConvAutoencoder:
    @staticmethod
    def build(width, height, depth, filters=(32,64), latentDim=16):

        #Make the input as "channels last" 
        inputShape = (height, width, depth)
        chanDim = -1

        #-----Start building the encoder

        #Define the inputs of the encoder
        inputs = Input(shape=inputShape)
        x = inputs

        #loop over the filters
        for f in filters:
            #Conv -> ReLU -> BN
            x = Conv2D(f, (3,3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha = 0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)

        #flatten the network and then construct the latent space (vector)
        #(i.e., the “compressed” data representation)
        volumSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latentDim)(x) #compress those digits into a vector of only 16 values

        #build the encoder model
        encoder = Model(inputs, latent, name="encoder")


        #-----Start building the decoder model

        #Go backwards
        latentInputs = Input(shape=(latentDim,))
        x = Dense(np.prod(volumSize[1:]))(latentInputs)
        x = Reshape((volumSize[1], volumSize[2], volumSize[3]))(x)

        #Loop over the number of filters again but using Conv2dTranspose for reverse
        for f in filters[::-1]:
			# apply a CONV_TRANSPOSE => RELU => BN operation
            x = Conv2DTranspose(f, (3, 3), strides=2,padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)

        #Recover the original depth of the image
        x = Conv2DTranspose(depth, (3,3), padding="same")(x)
        outputs = Activation('sigmoid')(x)

        #build the decoder
        decoder = Model(latentInputs, outputs, name='decoder')

        #Create tht autoencoder
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')

        #Return a 3 tuple of the encoder, decoder and autoencoder
        return (encoder, decoder, autoencoder)
