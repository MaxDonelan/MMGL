import os
import argparse
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_sample_obs(indices, n_obs = 2000):
    """
    Creates an array of n_obs randomly selected slices of randomly selected images.
    """
    length = len(indices)

    sample_images = np.random.randint(0, length, size=n_obs)
    sample_slices = np.random.rand(n_obs)

    samples = np.empty((n_obs, 512, 512))
    for i, sample_image in enumerate(sample_images):
        image_to_take = indices[sample_image]
        image_path = os.path.join("/scratch/jacks.local/mrdonelan/radfusion/multimodalpulmonaryembolismdataset/images/" + str(image_to_take) + ".npy")
        image = np.load(image_path)
        slice_to_take = int(sample_slices[i] * image.shape[0])
        samples[i, :, :] = image[slice_to_take, :, :]
    
    return samples

def get_samples(indices, n_obs = 2000):
    """
    Try to load the "labeling_samples.npy" file, and if it doesn't exist, call create_sample_obs to create it.
    """
    try:
        samples = np.load("labeling_samples.npy")
        if samples.shape[0] != n_obs:
            samples = create_sample_obs(indices, n_obs)
            np.save("data/RADFUSION/labeling_samples.npy", samples)
    except FileNotFoundError:
        samples = create_sample_obs(indices, n_obs)
        np.save("data/RADFUSION/labeling_samples.npy", samples)

    return samples

def threshold_image(image, lower_bound, upper_bound):
    """
    This function applies the following filter to all pixels in a greyscale image:
    - if the pixel's value is outside the bounds of the interval (lower_bound, upper_bound), set the pixel's value to 0.
    - else set the pixel's value to 1.
    """
    return np.where((image > upper_bound) | (image < lower_bound), 0, 1)

def normalize_slice(slc):
    """
    Normalize a slice of a CT scan by subtracting the mean and divided by the standard deviation
    so that the resulting slice has mean 0 and standard deviation 1. Importantly, this function
    ignores the filler pixels that surround the true image.  
    """
    null_value = slc[0, 0]
    in_scan = np.array([slc > null_value]).squeeze()
    mu = np.mean(slc, where=in_scan)
    sigma = np.std(slc, where=in_scan)
    return (slc - mu) / sigma

def create_training_slices(n_obs):
    """
    Obtains a random sample of slices from random images, then prompts the user to label each one.
    Saves both the set of sample slices as well as the labels. Slice sample is saved after being created,
    and will be reloaded without recomputation if the function is called again. Meanwhile, labels
    are only saved after *all* slices have been assigned a label.

    # Parameters
    - n_obs: the number of slices to randomly obtain.
    """
    indices = pd.read_csv('/scratch/jacks.local/mrdonelan/radfusion/multimodalpulmonaryembolismdataset/Labels.csv', index_col=0).idx

    samples = get_samples(indices, n_obs)
    lung_presence = np.empty(n_obs, dtype=np.int64)    

    for i in range(samples.shape[0]):
        sample = samples[i, :, :]

        fig, ax = plt.subplots(1, 2)
        ax[0].matshow(sample)
        thresholded_sample = threshold_image(normalize_slice(sample), 
                                             lower_bound=-1.25, 
                                             upper_bound=-0.5)
        ax[1].matshow(thresholded_sample)
        plt.savefig("current_slice.pdf")
        plt.close(fig)

        valid = {"yes": 1, "y": 1, "ye": 1, "no": 0, "n": 0}
        unanswered = True
        while unanswered:
            sys.stdout.write(f"Are the lungs present in slice {i+1}? (y/n)  ")
            choice = input().lower()
            if choice in valid:
                lung_presence[i] = valid[choice]
                unanswered = False
            elif choice == "exit":
                return 
            else:
                sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

    np.save("data/RADFUSION/lung_presence_key.npy", lung_presence)

if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_obs", type=int, default=100, 
                        help="Number of samples to take from the original data.")
    
    args = parser.parse_args()
    n_obs = args.n_obs
    create_training_slices(n_obs)