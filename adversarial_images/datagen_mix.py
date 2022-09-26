from operator import ge
from .fgsm import generate_image_adversary
from sklearn.utils import shuffle
import numpy as np

#Create the function for the MIXED dataset of normal images and adversarial ones

def generate_mixed_adversarial_batch(model, total, images, labels, dims, eps=0.1, split=0.5):
    #Same process
    (h,w,c) = dims

    #But, we get the amount of data we will get for the normal and adversary images
    totalNormal = int(total * split)
    totalAdv = int(total * (1-split))

    #Construct the data generator

    while True:

        #get random samples from the input data and use those indexes to sample NORMAL data
        idxs = np.random.choice(range(0, len(images)), size=totalNormal, replace=False)
        normalImages = images[idxs]
        normalLabels = labels[idxs]

        #now, again randomly sample indexes from the input data but do it for the ADVERSAY data
        idxs = np.random.choice(range(0, len(images)), size=totalAdv, replace=False)

        #loop with the indexes that are for the adversary images

        for i in idxs:
            #get the imagesto converto to adversary and their corresponding labels
            image = images[i]
            label = labels[i]

            #convert the image to adversary
            adversary = generate_image_adversary(model, image.reshape(1,h,w,c), label, eps=eps)

            #Put the normal and adverasry images together
            mixedImages = np.vstack([normalImages,adversary])
            mixedLabels = np.vstack([normalLabels, label])

        #once the process is done, shuffle them to avoid the model 
        #from first learning the normal images and then the adversarial ones
        (mixedImages, mixedLabels) = shuffle(mixedImages, mixedLabels)

        #Yield the mixed images and labels to generate the generator
        yield (mixedImages, mixedLabels)
