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


def train_CNN():
    ...

def plot_summaries(model_summaries: list[pd.DataFrame]) -> plt.Figure:
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


def train_lung_detector(samples_path: str, key_path: str) -> tuple[list[pd.DataFrame], CNN]:
    # read training data and key
    samples = np.load(samples_path)
    key = np.load(key_path)

    for i in range(samples.shape[0]):
        samples[i, :, :] = normalize_slice(samples[i, :, :])

    # create cross validation splits
    cv = StratifiedKFold(n_splits=10)
    n_samples = samples.shape[0]

    epochs = 25
    batch_size = 128
    learning_rate = 0.01
    momentum = 0.9    
    model_summaries = []
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, (train, test) in enumerate(cv.split(X=np.zeros(n_samples), y=key)):
        train_sample  = SliceSample(samples[train, :, :], key[train])
        test_sample = SliceSample(samples[test, :, :], key[test])
        
        train_loader = DataLoader(train_sample, batch_size=batch_size)
        test_loader = DataLoader(test_sample, batch_size=batch_size)

        # move this to a separate training function so that we can train the model without CV if we want?
        model = CNN().to(dev)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        model_summary = pd.DataFrame({"Train Acc": [], "Train Loss": [], "Train AUC": [], "Test Acc": [], "Test Loss": [], "Test AUC": []})

        print(f"\n\n\n---- Cross Validation Fold {i+1} ----\n")
        print("| Epoch | Train Acc. | Train Loss | Train AUC | Test Acc. | Test Loss | Test AUC |")

        for epoch in range(epochs):
            train_pred, train_actual, train_prob, train_loss = [], [], [], 0
            test_pred, test_actual, test_prob, test_loss = [], [], [], 0
            n_train_batches, n_test_batches = len(train_loader), len(test_loader)

            model.train()
            for _, (train_data, train_labels) in enumerate(train_loader):
                train_data, train_labels = train_data.to(dev), train_labels.to(dev)
                optimizer.zero_grad()
                output = model(train_data)
                loss = criterion(output, train_labels)
                loss.backward()
                optimizer.step()

                train_pred.extend(torch.argmax(output, dim=1).cpu().numpy())
                train_prob.extend(torch.softmax(output, dim=1)[:, 1].cpu().detach().numpy())
                train_actual.extend(train_labels.cpu().numpy())
                train_loss += loss.item()

            model.eval()
            with torch.no_grad():
                for _, (test_data, test_labels) in enumerate(test_loader):
                    test_data, test_labels = test_data.to(dev), test_labels.to(dev)
                    output = model(test_data)
                    test_pred.extend(torch.argmax(output, dim=1).cpu().numpy())
                    test_prob.extend(torch.softmax(output, dim=1)[:, 1].cpu().detach().numpy())
                    test_actual.extend(test_labels.cpu().numpy())
                    test_loss += criterion(output, test_labels).item()
            
            train_acc = np.mean(np.array(train_pred) == np.array(train_actual))
            test_acc = np.mean(np.array(test_pred) == np.array(test_actual))
            train_loss = train_loss / n_train_batches
            test_loss = test_loss / n_test_batches
            train_auc = roc_auc_score(train_actual, train_prob)
            test_auc = roc_auc_score(test_actual, test_prob)

            epoch_summary = pd.DataFrame({"Train Acc": [train_acc], "Train Loss": [train_loss], "Train AUC": [train_auc], "Test Acc": [test_acc], "Test Loss": [test_loss], "Test AUC": [test_auc]})
            model_summary = pd.concat([model_summary, epoch_summary], ignore_index=True)
            print(f"|   {epoch+1}   |    {train_acc:.5f} |    {train_loss:.5f} |   {train_auc:.5f} |   {test_acc:.5f} |   {test_loss:.5f} |  {test_auc:.5f} |")

        model_summaries.append(model_summary)

    fig = plot_summaries(model_summaries)
    plt.savefig("data/RADFUSION/model_summaries.png")
    plt.close(fig)

    return model_summaries, model


if __name__ == "__main__":
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, help="The path of the file for the samples")
    parser.add_argument("--key", type=str, help="The path of the file for the response key")

    args = parser.parse_args()
    samples_path = args.samples
    key_path = args.key

    train_lung_detector(samples_path, key_path)