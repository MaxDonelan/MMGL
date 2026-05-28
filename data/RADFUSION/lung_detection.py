import random
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from labeling import normalize_slice


class SliceSample(Dataset):
    """
    A Dataset class for the CT scan slices so that they can be passed into a DataLoader.
    """
    def __init__(self, slices, key):
        super(Dataset, self).__init__()
        self.slices = slices
        self.key = key
    
    def __getitem__(self, index):
        return torch.tensor(self.slices[index], dtype=torch.float32).unsqueeze(0), torch.tensor(self.key[index], dtype=torch.long)

    def __len__(self):
        return np.shape(self.slices)[0]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 32, stride=4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 8, 8, stride=2)
        self.linear1 = nn.Linear(13*13*8, 128)
        self.linear2 = nn.Linear(128, 16)
        self.linear3 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 512x512x1 -> 121x121x32
        x = self.pool(x) # 121x121x32 -> 60x60x32
        x = F.relu(self.conv2(x)) # 60x60x32 -> 27x27x8
        x = self.pool(x) # 27x27x8 -> 13x13x8
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    

class SliceClassifier:
    """
    A class to initialize, fit, and store a CNN trained to classify CT scan slices.
    """
    def __init__(self, epochs: int, learning_rate: float, momentum: float, device: str, model: CNN | None = None):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.device = device
        if model is None:
            self.model = CNN().to(self.device)
        else:
            self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.model_summary = pd.DataFrame({"Train Acc": [], "Train Loss": [], "Train AUC": [], 
                                           "Test Acc": [], "Test Loss": [], "Test AUC": []})
    

    def fit(self, train_loader: DataLoader, test_loader: DataLoader, verbose: bool = True) -> tuple[pd.DataFrame, CNN]:
        """
        Fits the SliceClassifer given a training and test dataset.

        Parameters
        ----------
        **train_loader** : *DataLoader*

            A training set provided as a torch.utils.data.DataLoader constructed from a SliceSample.

        **test_loader** : *DataLoader*

            A test set provided as a torch.utils.data.DataLoader constructed from a SliceSample.

        **verbose** : *bool, default True*
        
            Whether to print each row (epoch) of the model summary as it is computed.

        Returns
        -------
        A tuple of a model summary DataFrame and the fitted model.
        """
        if verbose:
            print("| Epoch | Train Acc. | Train Loss | Train AUC | Test Acc. | Test Loss | Test AUC |")

        for epoch in range(self.epochs):
            train_pred, train_actual, train_prob, train_loss = [], [], [], 0
            test_pred, test_actual, test_prob, test_loss = [], [], [], 0
            n_train_batches, n_test_batches = len(train_loader), len(test_loader)

            self.model.train()
            for _, (train_data, train_labels) in enumerate(train_loader):
                train_data, train_labels = train_data.to(self.device), train_labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(train_data)
                loss = self.criterion(output, train_labels)
                loss.backward()
                self.optimizer.step()

                # slowly collected predictions for the whole training data by adding each batch one-by-one
                train_pred.extend(torch.argmax(output, dim=1).cpu().numpy())
                train_prob.extend(torch.softmax(output, dim=1)[:, 1].cpu().detach().numpy()) # roc_auc_score() expects the probabilities of the "1" class
                train_actual.extend(train_labels.cpu().numpy())
                train_loss += loss.item()

            self.model.eval()
            with torch.no_grad():
                for _, (test_data, test_labels) in enumerate(test_loader):
                    test_data, test_labels = test_data.to(self.device), test_labels.to(self.device)
                    output = self.model(test_data)
                    test_pred.extend(torch.argmax(output, dim=1).cpu().numpy())
                    test_prob.extend(torch.softmax(output, dim=1)[:, 1].cpu().detach().numpy())
                    test_actual.extend(test_labels.cpu().numpy())
                    test_loss += self.criterion(output, test_labels).item()
            
            train_acc = np.mean(np.array(train_pred) == np.array(train_actual))
            test_acc = np.mean(np.array(test_pred) == np.array(test_actual))
            train_loss = train_loss / n_train_batches
            test_loss = test_loss / n_test_batches
            train_auc = roc_auc_score(train_actual, train_prob)
            test_auc = roc_auc_score(test_actual, test_prob)

            epoch_summary = pd.DataFrame({"Train Acc": [train_acc], "Train Loss": [train_loss], "Train AUC": [train_auc], 
                                          "Test Acc": [test_acc], "Test Loss": [test_loss], "Test AUC": [test_auc]})
            self.model_summary = pd.concat([self.model_summary, epoch_summary], ignore_index=True)
            if verbose:
                print(f"|   {epoch+1}   |    {train_acc:.5f} |    {train_loss:.5f} |   {train_auc:.5f} |   {test_acc:.5f} |   {test_loss:.5f} |  {test_auc:.5f} |")

        return self.model_summary, self.model
    

    def best_fit(self, train_loader: DataLoader, test_loader: DataLoader, metric: str = "auc", verbose: bool = True) -> tuple[pd.DataFrame, CNN]:
        """
        Fits the SliceClassifer given a training and test dataset, but yields the model from the epoch with the highest 
        test AUC instead of the last epoch.

        Parameters
        ----------
        **train_loader** : *DataLoader*

            A training set provided as a torch.utils.data.DataLoader constructed from a SliceSample.

        **test_loader** : *DataLoader*

            A test set provided as a torch.utils.data.DataLoader constructed from a SliceSample.

        **metric** : *str, default "auc"*

            The metric by which to select the best epoch. One of "auc" (default), "acc", or "loss".

        **verbose** : *bool, default True*
        
            Whether to print each row (epoch) of the model summary as it is computed.

        Returns
        -------
        A tuple of a model summary DataFrame and the best fitted model.
        """
        if verbose:
            print("| Epoch | Train Acc. | Train Loss | Train AUC | Test Acc. | Test Loss | Test AUC |")

        if metric == "loss":
            best_test_metric = np.inf
        elif metric == "acc" or metric == "auc":
            best_test_metric = 0
        else:
            raise ValueError

        best_model = None

        for epoch in range(self.epochs):
            train_pred, train_actual, train_prob, train_loss = [], [], [], 0
            test_pred, test_actual, test_prob, test_loss = [], [], [], 0
            n_train_batches, n_test_batches = len(train_loader), len(test_loader)

            self.model.train()
            for _, (train_data, train_labels) in enumerate(train_loader):
                train_data, train_labels = train_data.to(self.device), train_labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(train_data)
                loss = self.criterion(output, train_labels)
                loss.backward()
                self.optimizer.step()

                # slowly collected predictions for the whole training data by adding each batch one-by-one
                train_pred.extend(torch.argmax(output, dim=1).cpu().numpy())
                train_prob.extend(torch.softmax(output, dim=1)[:, 1].cpu().detach().numpy()) # roc_auc_score() expects the probabilities of the "1" class
                train_actual.extend(train_labels.cpu().numpy())
                train_loss += loss.item()

            self.model.eval()
            with torch.no_grad():
                for _, (test_data, test_labels) in enumerate(test_loader):
                    test_data, test_labels = test_data.to(self.device), test_labels.to(self.device)
                    output = self.model(test_data)
                    test_pred.extend(torch.argmax(output, dim=1).cpu().numpy())
                    test_prob.extend(torch.softmax(output, dim=1)[:, 1].cpu().detach().numpy())
                    test_actual.extend(test_labels.cpu().numpy())
                    test_loss += self.criterion(output, test_labels).item()
            
            train_acc = np.mean(np.array(train_pred) == np.array(train_actual))
            test_acc = np.mean(np.array(test_pred) == np.array(test_actual))
            train_loss = train_loss / n_train_batches
            test_loss = test_loss / n_test_batches
            train_auc = roc_auc_score(train_actual, train_prob)
            test_auc = roc_auc_score(test_actual, test_prob)

            if metric == "loss":
                if best_test_metric >= test_loss:
                    best_model = self.model
                    best_test_metric = test_loss
            elif metric == "acc":
                if best_test_metric <= test_auc:
                    best_model = self.model
                    best_test_metric = test_acc
            else:
                if best_test_metric <= test_auc:
                    best_model = self.model
                    best_test_metric = test_auc

            epoch_summary = pd.DataFrame({"Train Acc": [train_acc], "Train Loss": [train_loss], "Train AUC": [train_auc], 
                                          "Test Acc": [test_acc], "Test Loss": [test_loss], "Test AUC": [test_auc]})
            self.model_summary = pd.concat([self.model_summary, epoch_summary], ignore_index=True)
            if verbose:
                print(f"|   {epoch+1}   |    {train_acc:.5f} |    {train_loss:.5f} |   {train_auc:.5f} |   {test_acc:.5f} |   {test_loss:.5f} |  {test_auc:.5f} |")

        self.model = best_model

        return self.model_summary, self.model
    

    def get_class_probabilities(self, x: torch.Tensor) -> np.ndarray:
        """
        Obtain class probabilities for a new observation x.

        Parameters
        ----------
        **x** : *torch.Tensor*

        A new observation for which to calculate class probabilities.

        Returns
        -------
        Class probabilites for x.
        """
        x = x.to(self.device)
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(x), dim=1).cpu().detach().numpy()


    def predict(self, x: torch.Tensor | np.ndarray, cutoff: float = 0.5) -> np.ndarray:
        """
        Obtain model predictions either from a new observation or the 
        class probabilities for a given observation.

        Parameters
        ----------
        **x** : *torch.Tensor | np.ndarray*

        Either a torch.Tensor containing a new observation or a ndarray of 
        class probailities as returned by get_class_probabilities().

        **cutoff** : *float, default 0.5*

        An alternative cutoff to use for prediction. If not supplied defaults to 0.5.

        Returns
        -------
        The positive or negative predictions for x.
        """
        if isinstance(x, torch.Tensor):
            pred = self.get_class_probabilities(x)[:, 1]
            return np.where(pred > cutoff, 1, 0) 
        elif isinstance(x, np.ndarray):
            return np.where(x[:, 1] > cutoff, 1, 0)
        else:
            raise TypeError


def plot_summaries(model_summaries: list[pd.DataFrame]) -> plt.Figure:
    """
    Create the model summary plots.

    Parameters
    ----------
    **model_summaries** : *list[pd.DataFrame]*

        The list of model summaries, each as it is generated by SliceClassifier.fit()

    Returns
    -------
    A 2x3 grid of plots of all the model summaries per epoch for training and test data.
    """
    labels = [f"CV Fold {i+1}" for i in range(len(model_summaries))]

    metrics = [
        ("Train Acc", "Train Accuracy"),
        ("Train Loss", "Train Loss"),
        ("Train AUC", "Train AUC"),
        ("Test Acc", "Test Accuracy"),
        ("Test Loss", "Test Loss"),
        ("Test AUC", "Test AUC")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, metrics):
        n_epochs = len(model_summaries[0])
        for model_summary, label in zip(model_summaries, labels):
            ax.plot(model_summary.index, model_summary[col], label=label)

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(n_epochs))
        ax.set_ylabel(title)
        legend_loc = "upper right" if ax in axes[[1, 4]] else "lower right"
        ax.legend(loc=legend_loc, fontsize=5)
        ax.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle("Model Summaries", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def cv_train_lung_detector(samples_path: str, labels_path: str) -> list[pd.DataFrame]:
    """
    Load the training data, normalize the slices, set the hyperparameters for the model, set up 10-fold cross validation,
    and train one SliceClassifier model on each fold, saving the model summaries.

    Parameters
    ----------
    **samples_path** : *str*

    The file path to the sample data.

    **labels_path** : *str*

    The file path to the labels for the samples.

    Returns
    -------
    A list of model summaries as returned by SliceClassifier().fit().
    """
    samples = np.load(samples_path)
    labels = np.load(labels_path)

    for i in range(samples.shape[0]):
        samples[i, :, :] = normalize_slice(samples[i, :, :])

    cv = StratifiedKFold(n_splits=10)
    n_samples = samples.shape[0]

    epochs = 25
    batch_size = 128
    learning_rate = 0.01
    momentum = 0.9    
    model_summaries = []
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, (train, test) in enumerate(cv.split(X=np.zeros(n_samples), y=labels)):
        train_sample  = SliceSample(samples[train, :, :], labels[train])
        test_sample = SliceSample(samples[test, :, :], labels[test])
        
        train_loader = DataLoader(train_sample, batch_size=batch_size)
        test_loader = DataLoader(test_sample, batch_size=batch_size)

        print(f"\n\n---- Cross Validation Fold {i+1} ----\n")

        model = SliceClassifier(epochs=epochs, learning_rate=learning_rate, momentum=momentum, device=dev)
        model_summary, fitted_model = model.fit(train_loader=train_loader, test_loader=test_loader)

        model_summaries.append(model_summary)

    return model_summaries


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

    model_summaries = cv_train_lung_detector(samples_path, labels_path)

    fig = plot_summaries(model_summaries)
    plt.savefig("data/RADFUSION/model_summaries.png")
    plt.close(fig)
