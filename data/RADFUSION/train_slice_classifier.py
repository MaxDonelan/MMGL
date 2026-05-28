import random
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, DetCurveDisplay, PrecisionRecallDisplay, roc_curve, precision_recall_curve, det_curve
from lung_detection import *

def explore_cutoffs(test_pred: np.ndarray, test_labels: np.ndarray) -> plt.Figure:
    fpr, tpr, thres_roc = roc_curve(test_labels, test_pred)
    precision, recall, thres_pr = precision_recall_curve(test_labels, test_pred)
    fpr_det, fnr, thres_det  = det_curve(test_labels, test_pred)

    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name="CNN")
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall, estimator_name="CNN")
    det_display = DetCurveDisplay(fpr=fpr_det, fnr=fnr, estimator_name="CNN")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8))

    roc_display.plot(ax=ax1)
    pr_display.plot(ax=ax2)
    det_display.plot(ax=ax3)

    return fig


def train_classifier(samples_path: str, labels_path: str) -> tuple[pd.DataFrame, CNN]:
    samples = np.load(samples_path)
    labels = np.load(labels_path)

    for i in range(samples.shape[0]):
        samples[i, :, :] = normalize_slice(samples[i, :, :])

    X_train, X_test, y_train, y_test = train_test_split(samples, labels, stratify=labels)

    train = SliceSample(X_train, y_train)
    test = SliceSample(X_test, y_test)

    batch_size = 64
    train_loader = DataLoader(train, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)

    epochs = 25
    learning_rate = 0.01
    momentum = 0.9
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SliceClassifier(epochs=epochs, learning_rate=learning_rate, momentum=momentum, device=dev)

    model_summary, best_fit_model = model.best_fit(train_loader=train_loader, test_loader=test_loader, metric="loss")

    pred = model.get_class_probabilities(torch.Tensor(X_test).unsqueeze(1))[:, 1]

    fig = explore_cutoffs(test_pred=pred, test_labels=y_test)
    plt.savefig("data/RADFUSION/cutoff_exploration.png")
    plt.close(fig)

    return model_summary, best_fit_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, help="The path of the file for the samples")
    parser.add_argument("--labels", type=str, help="The path of the file for the response labels")
    parser.add_argument("--seed", type=int, help="The random seed to use for all processes")

    args = parser.parse_args()
    samples_path = args.samples
    labels_path = args.labels
    seed = args.seed
    
    if seed is not None:
        random.seed(seed)

    summary, model = train_classifier(samples_path=samples_path, labels_path=labels_path) 
    torch.save(model.state_dict(), "data/RADFUSION/trained_cnn.pt")

    summaries = [summary]
    fig = plot_summaries(summaries)
    plt.savefig("data/RADFUSION/trained_model_summary.png")
    plt.close(fig)