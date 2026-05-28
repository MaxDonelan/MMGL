import random
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import hyperopt
from hyperopt import hp, STATUS_OK
from lung_detection import *


def train_parameter_combo(params, X_train, X_val, y_train, y_val, X_test, y_test):
    batch_sizes = [8, 16, 32, 64, 128, 256]
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    batch_size = batch_sizes[params["batch_size"]]
    learning_rate = learning_rates[params["learning_rate"]]
    momentum = params["momentum"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nBatch Size: {batch_size} | Learning Rate: {learning_rate} | Momentum: {momentum}\n")

    train = SliceSample(X_train, y_train)
    val = SliceSample(X_val, y_val)

    train_loader = DataLoader(train, batch_size=batch_size)
    val_loader = DataLoader(val, batch_size=batch_size)

    model = SliceClassifier(epochs=25, 
                            learning_rate=learning_rate, 
                            momentum=momentum, 
                            device=device)
    
    model_summary, cnn = model.best_fit(train_loader, val_loader, metric="loss")
    val_loss = np.min(model_summary["Test Loss"])

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    test_out = cnn(X_test)
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = criterion(test_out, y_test).item()

    print(f"\nTest Loss: {test_loss}\n")

    return {"loss": val_loss, "status": STATUS_OK}


if __name__ == "__main__":
    random.seed(47)

    samples = np.load("data/RADFUSION/labeling_samples.npy")
    labels = np.load("data/RADFUSION/lung_presence_key.npy")
    

    for i in range(samples.shape[0]):
        samples[i, :, :] = normalize_slice(samples[i, :, :])

    X_train, X_test, y_train, y_test = train_test_split(samples, labels, stratify=labels, test_size=0.2, random_state=47)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.125, random_state=47)

    best_choice = hyperopt.fmin(partial(train_parameter_combo, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val, X_test=X_test, y_test=y_test), 
                                space={
                                    "batch_size": hp.randint("batch_size", 6),
                                    "learning_rate": hp.randint("learning_rate", 4),
                                    "momentum": hp.uniform("momentum", 0, 1)
                                },
                                algo=hyperopt.tpe.suggest,
                                max_evals=25)
    
    print(best_choice)