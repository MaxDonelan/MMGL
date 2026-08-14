import random
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import normalize 
from sklearn.ensemble import RandomForestClassifier

from lung_detection import *
from train_slice_classifier import *
from preprocessing import process_scan, load_EHR


def assemble_data(dev, path):
    labels_path = path / "Labels.csv"
    labels = pd.read_csv(labels_path, index_col=0).head(500) # No longer robust to patterns in the indices of the data
    
    cnn = torch.load("data/RADFUSION/trained_cnn.pt", map_location=dev)
    slice_classifier = SliceClassifier(epochs=25,
                                       learning_rate=0.01,
                                       momentum=0.9,
                                       device=dev,
                                       model_state=cnn)
    with open("data/RADFUSION/balanced_threshold.pkl", "rb") as f:
        cutoff = pickle.load(f)["thres"]

    print("Creating CT scan paths...")
    ct_scan_paths = []
    for idx in labels.idx:
        scan_path = path.joinpath("images", str(idx) + ".npy")
        ct_scan_paths.append(scan_path)

    print("Processing images...")
    slices = []
    slice_level_labels = []
    slice_level_idx = []
    slice_level_split = []
    total_slices = 0
    for i, scan_path in enumerate(ct_scan_paths):
        slc_set, original_n_slices = process_scan(scan_path, slice_classifier, cutoff, reshape=False)
        total_slices += original_n_slices
        if slc_set is not None:
            slice_level_labels.extend([labels.label[i]] * slc_set.shape[0])
            slice_level_idx.extend([labels.idx[i]] * slc_set.shape[0])
            slice_level_split.extend([labels.split[i]] * slc_set.shape[0])
            slices.extend(slc_set)
        if i % 10 == 9:
            print(f"Processed scan {i+1}")

    slices = np.stack(slices, axis=0)
    print(f"Original Number of Slices: {total_slices}")
    print(f"Shape of slice-level data: {slices.shape}")

    np.save(path / "slices.npy", slices)
    np.save(path / "all_slice_level_idx.npy", slice_level_idx)
    np.save(path / "all_slice_level_split.npy", slice_level_split)
    np.save(path / "all_slice_level_labels.npy", slice_level_labels)

    return labels


def get_balanced_cutoff(labels, prob) -> tuple[float, float, float]:
    """
    Find the threshold which best balances the true positive rate (TPR) 
    and true negative rate (TNR) and returns that threshold along with
    the associated false positive rate and true positive rate.
    """
    fpr, tpr, thres = roc_curve(labels, prob)
    tnr = 1 - fpr
    balanced_thres = np.argmin(abs(tpr - tnr))
    return fpr[balanced_thres], tpr[balanced_thres], thres[balanced_thres]


def b_baseline(labels, scratch_dir):
    print("Loading EHR...")    
    ehr = load_EHR(scratch_dir, labels).drop_duplicates("idx")
    print(f"EHR Shape: {ehr.shape}")
    print(f"Missing: {ehr.isna().values.any()}")

    train_idx = labels.loc[labels["split"] == "train", "idx"].unique()
    test_idx = labels.loc[labels["split"] == "test", "idx"].unique()
    train_mask = ehr["idx"].isin(train_idx)
    test_mask = ehr["idx"].isin(test_idx)
    ehr_train = ehr[train_mask]
    ehr_test = ehr[test_mask]

    print("Performing feature selection...")
    estimator = RidgeClassifier()
    predictors_train = normalize(ehr_train.drop(columns=["idx", "label", "split_x", "split", "split_y"]))
    predictors_test = normalize(ehr_test.drop(columns=["idx", "label", "split_x", "split", "split_y"]))
    response = ehr_train["label"]

    selector = RFE(estimator, n_features_to_select=512, step=25, verbose=1)
    selector = selector.fit(X=predictors_train, y=response)
    #features_selected = selector.get_feature_names_out()
    transformed_train = selector.transform(predictors_train)
    transformed_test = selector.transform(predictors_test)
    #transformed_train = pd.DataFrame(transformed_train, columns=features_selected)

    rf = RandomForestClassifier()
    rf = rf.fit(X=transformed_train, y=response)
    prob = rf.predict_proba(X=transformed_test)[:, 1]

    return rf, prob


def a_baseline(scratch_dir, checkpoint):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if checkpoint == False:
        labels = assemble_data(dev, scratch_dir)

    print("Loading Data...")
    slices = np.load(scratch_dir / "slices.npy")
    slice_level_idx = np.load(scratch_dir / "all_slice_level_idx.npy")
    slice_level_split = np.load(scratch_dir / "all_slice_level_split.npy")
    slice_level_labels = np.load(scratch_dir / "all_slice_level_labels.npy")

    train_mask = slice_level_split == "train"
    val_mask = slice_level_split == "val"
    test_mask = slice_level_split == "test"

    train_index = np.array(range(slices.shape[0]))[train_mask]
    val_index = np.array(range(slices.shape[0]))[val_mask]
    test_index = np.array(range(slices.shape[0]))[test_mask]

    rf, rf_test_prob = b_baseline(labels, scratch_dir)

    train_sample = SliceSample(slices[train_index], slice_level_labels[train_index])
    val_sample = SliceSample(slices[val_index], slice_level_labels[val_index])
    #test_set = slices[test_index]

    train_loader = DataLoader(train_sample, batch_size=64)
    val_loader = DataLoader(val_sample, batch_size=64)
    print(f"Number of Training Batches: {len(train_loader)}")
    print(f"Number of Validation Batches: {len(val_loader)}")

    print("Training Classifier")
    a_baseline = SliceClassifier(epochs=50, learning_rate=0.01, momentum=0.9, device=dev)

    model_summary, best_fit_model = a_baseline.best_fit(train_loader, val_loader, metric = "loss")

    # test on test set (see train_slice_classifier)
    test_prob = a_baseline.get_class_probabilities(torch.Tensor(slices[test_index]).unsqueeze(1))[:, 1]
    test_actual = slice_level_labels[test_index]
    fpr, tpr, cutoff = get_balanced_cutoff(test_actual, test_prob)
    test_pred = np.where(test_prob > cutoff, 1, 0)
    test_acc = np.mean(test_pred == test_actual)
    test_auc = roc_auc_score(test_actual, test_prob)

    print(f"\nBalanced Cutoff of {cutoff:.4f}:  Test Set Accuracy: {test_acc:.4f} | Test Set AUC: {test_auc:.4f} | FPR: {fpr:.4f} | TPR: {tpr:.4f}")

    # grouped:
    to_group = pd.DataFrame({"prob": test_prob, "idx": slice_level_idx[test_index], "label": test_actual})
    grouped_prob = to_group.groupby(["idx"])["prob"].mean()
    grouped_labels = to_group.groupby(["idx"])["label"].mean()
    g_fpr, g_tpr, grouped_cutoff = get_balanced_cutoff(grouped_labels, grouped_prob)
    grouped_pred = np.where(grouped_prob > grouped_cutoff, 1, 0)
    grouped_acc = np.mean((np.array(grouped_pred) == np.array(grouped_labels)))
    grouped_auc = roc_auc_score(grouped_labels, grouped_prob)

    print("\n Results After Regrouping to the Image Level:")
    print(f"Grouped: FPR: {g_fpr:.4f} | TPR {g_tpr:.4f} | Cutoff: {grouped_cutoff:.4f}")
    print(f"Grouped Balanced Cutoff Test Acc: {grouped_acc:.4f} | Grouped Test AUC: {grouped_auc:.4f}")

    multi_test_prob = (grouped_prob + rf_test_prob) / 2
    m_fpr, m_tpr, multi_cutoff = get_balanced_cutoff(grouped_labels, multi_test_prob)
    multi_test_pred = np.where(multi_test_prob > multi_cutoff, 1, 0)
    mulit_test_acc = np.mean(multi_test_pred == grouped_labels)
    multi_test_auc = roc_auc_score(grouped_labels, multi_test_prob)

    print("\n Results After Regrouping to the Image Level + RF EHR Model:")
    print(f"Multi-Modal: FPR: {m_fpr:.4f} | TPR {m_tpr:.4f} | Cutoff: {multi_cutoff:.4f}")
    print(f"Multi-Modal Balanced Cutoff Test Acc: {mulit_test_acc:.4f} | Multi-Modal Test AUC: {multi_test_auc:.4f}")

    return model_summary, best_fit_model, cutoff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="The random seed to use for all processes")
    parser.add_argument("--checkpoint", type=bool, default=False, help="Start after image preprocessing")

    args = parser.parse_args()
    seed = args.seed
    checkpoint = args.checkpoint
    
    if seed is not None:
        random.seed(seed)

    scratch_dir = Path("/scratch/jacks.local/mrdonelan/radfusion/multimodalpulmonaryembolismdataset/")

    summary, model, balanced_thres = a_baseline(scratch_dir, checkpoint) 
    torch.save(model.state_dict(), scratch_dir / "trained_radfusion_cnn.pt")

    with open(scratch_dir / "balanced_threshold.pkl", "wb") as f:
        pickle.dump(balanced_thres, f)
