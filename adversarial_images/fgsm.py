from tensorflow.keras.losses import MSE
import tensorflow as tf


#Eps: controls the size of the perturbation applied to the images
def generate_image_adversary_model(model, image, label, eps=2/255.0):

    image  = tf.cast(image, tf.float32)

    #record the gradients
    with tf.GradientTape() as tape:
        #We indicate that the image should be tracked for gradient updates
        tape.watch(image)

        #Make predictions on the input image and get the loss
        pred= model(image)
        loss= MSE(label, pred)

    #claculate the gradients of loss with respect to the image
    gradient = tape.gradient(loss, image)
    #Get the sign of the resultant gradient. Same as using np.sign()
    signedGrad = tf.sign(gradient)

    #construct the image adversary. And convert that to numpy
    adversary = (image + (signedGrad * eps)).numpy()

    return adversary

