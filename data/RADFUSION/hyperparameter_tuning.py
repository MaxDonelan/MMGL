import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import hyperopt
from hyperopt import hp, STATUS_OK
from lung_detection import *


def train_parameter_combo(batch_size, learning_rate, momentum):
    samples = np.load("data/RADFUSION/labeling_samples.npy")
    labels = np.load("data/RADFUSION/lung_presence_key.npy")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i in range(samples.shape[0]):
        samples[i, :, :] = normalize_slice(samples[i, :, :])

    X_train, X_test, y_train, y_test = train_test_split(samples, labels, stratify=labels, test_size=0.2)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.125)

    train = SliceSample(X_train, y_train)
    val = SliceSample(X_val, y_val)

    train_loader = DataLoader(train, batch_size=batch_size)
    val_loader = DataLoader(val, batch_size=batch_size)

    model = SliceClassifier(epochs=25, 
                            learning_rate=learning_rate, 
                            momentum=momentum, 
                            device=device)
    
    model_summary, cnn = model.best_fit(train_loader, val_loader, metric="loss")
    val_loss = np.min(model_summary["test_loss"])

    X_test = torch.Tensor(X_test).to(device)
    test_out = cnn(X_test)
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = criterion(test_out, y_test).item()

    print(f"\nTest Loss: {test_loss}\n")

    return {"loss": val_loss, "status": STATUS_OK}


if __name__ == "__main__":
    random.seed(47)

    best_choice = hyperopt.fmin(train_parameter_combo, 
                                space={
                                    "batch_size": hp.choice("batch_size", [8, 16, 32, 64, 128, 256]),
                                    "learning_rate": hp.loguniform("learning_rate", 1e-4, 1e-1),
                                    "momentum": hp.uniform("momentum", 0, 1)
                                },
                                algo=hyperopt.tpe.suggest,
                                max_evals=25)
    
    print(best_choice)