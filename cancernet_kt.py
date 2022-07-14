# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

class CancerNet_kt:
	@staticmethod
	def build(hp):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself

		lr = hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3])

		model = Sequential()
		inputShape = (48, 48, 3)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (3, 48, 48)
			chanDim = 1

		# CONV => RELU => POOL
		model.add(SeparableConv2D(
            hp.Int("sep_conv_1", min_value=32, max_value=96, step=32), (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# (CONV => RELU => POOL) * 2
		model.add(SeparableConv2D(
            hp.Int("sep_conv_2", min_value=64,max_value=128, step=32), (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(SeparableConv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# (CONV => RELU => POOL) * 3
		model.add(SeparableConv2D(
            hp.Int("sep_conv_3", min_value=128, max_value=256, step=32), (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(SeparableConv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))

		model.add(SeparableConv2D(
            hp.Int("sep_conv_4", min_value=128, max_value=256, step=32), (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(
            hp.Int("dense_units", min_value=256, max_value=768, step=256)
            ))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# binary classifier
		model.add(Dense(1, activation="sigmoid",
			bias_initializer="zeros"))
		model.compile(optimizer=Adam(learning_rate=lr), loss = "catgorical_crossentropy", metrics=["accuracy"])
		# return the constructed network architecture
		return model