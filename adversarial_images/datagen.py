from .fgsm import generate_image_adversary
import numpy as np
def generate_adversarial_batch(model, total, images, labels, dims, eps=0.1):
    #unpack the image dimensrions
    (h,w,c) = dims

    #When creating a data generator, we need to loop things indefinitely 
    while True:
        #list to stor the perturbed images
        perturbImages = []
        perturbLabels = []

        #Random sample indexes (withouth replacement (replace=False)) from the input data
        idxs = np.random.choice(range(0, len(images)), size=total, replace=False)

        #Go through all indexes
        for i in idxs:
            #grab current image and label
            image = images[i]
            label = labels[i]

            #Generate an adversarial image (add batch channel to image)
            adversary = generate_image_adversary(model, image.reshape(1, h,w,c), label, eps=eps)

            #Add the adversary images to the lists and the labels
            perturbImages.append(adversary.reshape(h,w,c))
            perturbLabels.append(label)

        #Yield the image and the labels as numpy arrays
        yield(np.array(perturbImages), np.array((perturbLabels)))
