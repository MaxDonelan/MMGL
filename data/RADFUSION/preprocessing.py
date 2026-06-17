import argparse
import random
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import normalize
from lung_detection import *


def process_scan(path: Path, classifier: SliceClassifier, cutoff, reshape = True) -> np.ndarray | None:
    """
    Load a CT scan, normalize it, obtain the classifier's prediction, then resize the slices from
    512x512 to 1x(256*256).
    """
    scan = np.load(path)
    for i in range(scan.shape[0]):
        scan[i, :, :] = normalize_slice(scan[i, :, :])
        original_n_slices = scan.shape[0]
    try:
        scan = torch.tensor(scan, dtype=torch.float32).unsqueeze(1)
    except TypeError:
        print(f"Invalid Type: Skipping scan at {path}")
        return None, 0
    pred = classifier.predict(scan, cutoff=cutoff)
    keep = pred == 1
    print(f"Original # Slices: {scan.shape[0]} | Kept Slices: {sum(pred)}")
    if sum(pred) != 0:
        if reshape:
            scan = scan[keep, :, ::4, ::4].numpy()
            scan = scan.reshape(scan.shape[0], -1)
        else:
            scan = scan[keep].squeeze(1).numpy()
    else:
        print(f"No valid lung slices were found out of {scan.shape[0]} slices for scan at path {path}")
        return None, 0
    return scan, original_n_slices


def load_EHR(dir: Path, labels):
    """
    Load the EHR data from the provided directory, keeping only indexes in labels.idx.
    """
    demographics = pd.read_csv(dir / "Demographics.csv", index_col=0)
    demographics = demographics[demographics.idx.isin(labels.idx)].drop(columns=["Male", "SMOKER_N"])

    icd = pd.read_csv(dir / "ICD.csv", index_col=0)
    icd = icd[icd.idx.isin(labels.idx)]

    inp_med = pd.read_csv(dir / "INP_MED.csv", index_col=0)
    inp_med = inp_med[inp_med.idx.isin(labels.idx)].drop(columns=["split"])
    inp_med = inp_med.add_prefix("in_").rename(columns={"in_idx": "idx"})

    labs = pd.read_csv(dir / "LABS.csv", index_col=0)
    labs = labs[labs.idx.isin(labels.idx)]

    out_med = pd.read_csv(dir / "OUT_MED.csv", index_col=0)
    out_med = out_med[out_med.idx.isin(labels.idx)].drop(columns=["split"])
    out_med = out_med.add_prefix("out_").rename(columns={"out_idx": "idx"})

    dfs = [demographics, icd, inp_med, labs, out_med, labels[['idx', 'label']]]
    ehr = reduce(lambda left, right: pd.merge(left, right, on='idx', how='inner'), dfs)
    return ehr


def preprocessing(slice_limit, checkpoint):
    """
    Perform data preprocessing on the RadFusion dataset to format it for MMGL.
    """
    print("Starting preprocessing...")
    scratch_dir = Path("/scratch/jacks.local/mrdonelan/radfusion/multimodalpulmonaryembolismdataset/")
    labels_path = scratch_dir / "Labels.csv"
    labels = pd.read_csv(labels_path, index_col=0).head(5)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = torch.load("data/RADFUSION/trained_cnn.pt", map_location=device)
    slice_classifier = SliceClassifier(epochs=25,
                                       learning_rate=0.01,
                                       momentum=0.9,
                                       device=device,
                                       model_state=cnn)
    with open("data/RADFUSION/balanced_threshold.pkl", "rb") as f:
        cutoff = pickle.load(f)["thres"]

    print("Creating CT scan paths...")
    ct_scan_paths = []
    for idx in labels.idx:
        scan_path = scratch_dir.joinpath("images", str(idx) + ".npy")
        ct_scan_paths.append(scan_path)

    if checkpoint == False:
        print("Processing images...")
        slices = []
        slice_level_labels = []
        slice_level_idx = []
        slice_level_split = []
        total_slices = 0
        for i, path in enumerate(ct_scan_paths):
            slc_set, original_n_slices = process_scan(path, slice_classifier, cutoff)
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

        print("Performing PCA...")
        pca_model = PCA(n_components=512, copy=False) # could use better intuition on this value
        slices_transformed = pca_model.fit_transform(X=slices)

        np.save("data/RADFUSION/slices_transformed.npy", slices_transformed)
        np.save("data/RADFUSION/all_slice_level_idx.npy", slice_level_idx)
        np.save("data/RADFUSION/all_slice_level_split.npy", slice_level_split)
        np.save("data/RADFUSION/all_slice_level_labels.npy", slice_level_labels)

    slices_transformed = np.load("data/RADFUSION/slices_transformed.npy")
    slice_level_idx = np.load("data/RADFUSION/all_slice_level_idx.npy")
    slice_level_split = np.load("data/RADFUSION/all_slice_level_split.npy")
    slice_level_labels = np.load("data/RADFUSION/all_slice_level_labels.npy")

        # take only a random subset of slices for each scan
    if slice_limit is not None:
        slice_level_idx_arr = np.column_stack([np.arange(len(slice_level_idx)), np.array(slice_level_idx)])
        selection = []
        for idx in labels.idx:
            mask = slice_level_idx_arr[:, 1] == idx
            subset = slice_level_idx_arr[mask, :]
            if subset.shape[0] >= slice_limit:
                selected_indices = random.sample(list(subset[:, 0]), slice_limit)
                selection.extend(selected_indices)
            else:
                selection.extend(list(subset[:, 0]))

        slice_level_idx = np.array(slice_level_idx)[selection]
        slice_level_labels = np.array(slice_level_labels)[selection]
        slice_level_split = np.array(slice_level_split)[selection]
        slices_transformed = slices_transformed[selection]
        print(f"Shape of slice-level data post-transformations: {slices_transformed.shape}")

    # EHR preprocessing
    print("Loading EHR...")    
    ehr = load_EHR(scratch_dir, labels).drop_duplicates("idx")
    print(f"EHR Shape: {ehr.shape}")
    print(f"Missing: {ehr.isna().values.any()}")

    print("Performing feature selection...")
    estimator = RidgeClassifier()
    predictors = normalize(ehr.drop(columns=["idx", "label", "split_x", "split", "split_y"]))
    response = ehr["label"]

    selector = RFE(estimator, n_features_to_select=512, step=25, verbose=1)
    selector = selector.fit(X=predictors, y=response)
    features_selected = selector.get_feature_names_out()
    transformed_tabular = selector.transform(predictors)
    transformed_tabular = pd.DataFrame(transformed_tabular, columns=features_selected)
    transformed_tabular["idx"] = ehr["idx"].values

    # merging modalities
    print("Merging Modalities...")
    slices_df = pd.DataFrame(slices_transformed,
                        columns=[f'img_feature_{i}' for i in range(slices_transformed.shape[1])])
    slices_df['idx'] = slice_level_idx

    radfusion = pd.merge(slices_df, transformed_tabular, how="left", on="idx")
    radfusion_labels = slice_level_labels
    radfusion_labels = [x + 1 for x in radfusion_labels] # MMGL expects class labels of 1,...,n
    radfusion.insert(radfusion.shape[1], column="label", value=radfusion_labels)
    radfusion = radfusion.drop(columns=["idx"])

    print("Creating the modality feature dictionary...")
    modal_feat_dict = {"EHR": [], "IMAGE": []}
    modal_feat_dict["EHR"] = transformed_tabular.drop(columns=["idx"]).columns
    modal_feat_dict["IMAGE"] = slices_df.drop(columns=["idx"]).columns

    print(f"Missing: {np.any(np.isnan(radfusion))}")

    return radfusion, modal_feat_dict, np.array(slice_level_idx), np.array(slice_level_split) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="The random seed to use for all processes")
    parser.add_argument("--slice_limit", type=int, default=None, help="A maximum number of slices to keep from each image")
    parser.add_argument("--checkpoint", type=bool, default=False, help="Start after image preprocessing")

    args = parser.parse_args()
    seed = args.seed
    slice_limit = args.slice_limit
    checkpoint = args.checkpoint
    print(f"Checkpoint: {checkpoint}")

    if seed is not None:
        random.seed(seed)

    prepared_data, modal_feat_dict, slice_level_idx, slice_level_split = preprocessing(slice_limit, checkpoint)
    prepared_data.to_csv("data/RADFUSION/processed_standard_data.csv")
    np.save("data/RADFUSION/modal_feat_dict.npy", modal_feat_dict)
    np.save("data/RADFUSION/slice_level_idx.npy", slice_level_idx)
    np.save("data/RADFUSION/slice_level_split.npy", slice_level_split)
    print("Done.")