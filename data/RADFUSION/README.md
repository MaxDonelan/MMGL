## Overview

#### labeling.py
* This script is for manually creating a data set to train the lung slice classifier.

#### lung_detection.py
* Contains the SliceClassifier class for binary classification on CT scan slices.
* Call script to train a SliceClassifier using 10-fold CV on the test data set created by `labeling.py`.

#### hyperparameter_tuning.py
* Tune the hyperparameters for the SliceClassifier

#### train_slice_classifier
* Train the SliceClassifier and report performance metrics

#### preprocessing.py
* The full data preprocessing architecture
* Once you've run the script from scratch once, you can skip the image processing (which takes the longest amount of time by far) using the `--checkpoint` flag if you so wish.

---

We have also included the model weights (`trained_cnn.pt`) for the SliceClassifier used in `preprocessing.py`, so it should be possible to call `preprocessing.py` without first training the SliceClassifier.
