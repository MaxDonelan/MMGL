To initialize an inductive MMGL model, provide
* the number of features in each modality
* A list of hyperparameters (see `main.py` and/or `mmgl.py`)
* A CUDA device

From there, call the `.fit()` function, providing your data and labels, as well as the train/test splits.

Once the model is trained, you can use `.predict_class_prob()` to get model outputs (log probabilities) on new data. Or, you can use `.predict()` to obtain absolute predictions from the model. For `.predict()`, the default cutoff to use is 0.5, but you can change that with the cutoff argument.

To quickly run MMGL on the ABIDE or RadFusion data sets:

```bash MMGL_inductive/ABIDE-simple-2-concat-weighted-cosine.sh```

```bash RADFUSION.sh```
