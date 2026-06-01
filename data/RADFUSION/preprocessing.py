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

def process_scan(path, classifier: SliceClassifier, cutoff):
    scan = np.load(path)
    for i in range(scan.shape[0]):
        scan[i, :, :] = normalize_slice(scan[i, :, :])
    scan = torch.tensor(scan).unsqueeze(1)
    pred = classifier.predict(scan, cutoff=cutoff)
    scan = scan.squeeze()[pred, ::2, ::2].numpy() # not sure if .squeeze() is needed
    scan = scan.reshape(scan.shape[0], -1)
    return scan


def load_EHR(dir: Path, labels):
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


def preprocessing():
    print("Starting preprocessing...")
    scratch_dir = Path("/scratch/jacks.local/mrdonelan/radfusion/multimodalpulmonaryembolismdataset/")
    labels_path = scratch_dir / "Labels.csv"
    labels = pd.read_csv(labels_path, index_col=0)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = torch.load("data/RADFUSION/trained_cnn.pt")
    slice_classifier = SliceClassifier(epochs=25,
                                       learning_rate=0.01,
                                       momentum=0.9,
                                       device=device,
                                       model=cnn)
    cutoff = pickle.load("data/RADFUSION/balanced_threshold.pkl")

    print("Creating CT scan paths...")
    ct_scan_paths = []
    for idx in labels.idx:
        scan_path = scratch_dir.joinpath(str(idx), ".npy")
        ct_scan_paths.append(scan_path)

    print("Processing images...")
    slices = []
    slice_level_labels = []
    slice_level_idx = []
    for i, path in enumerate(ct_scan_paths):
        slc_set = process_scan(path, slice_classifier, cutoff)
        slice_level_labels.extend([labels.label[i]] * slc_set.shape[0])
        slice_level_idx.extend([labels.idx[i]] * slc_set.shape[0])
        slices.extend(slc_set)
        if i % 10 == 9:
            print(f"Cut non-lung slices from scan {i}")

    slices = np.stack(slices, axis=0)
    print(f"Shape of slice-level data: {slices.shape}")

    print("Performing PCA...")
    pca_model = PCA(n_components=512) # could use better intuition on this value
    slices_transformed = pca_model.fit_transform(X=slices, y=slice_level_labels)

    # EHR preprocessing
    print("Loading EHR...")    
    ehr = load_EHR(scratch_dir, labels)

    print("Performing feature selection...")
    estimator = RidgeClassifier()
    predictors = normalize(ehr.drop(columns=["idx", "label", "split_x", "split", "split_y"]))
    response = ehr["label"]

    selector = RFE(estimator, n_features_to_select=512, step=25, verbose=1)
    selector = selector.fit(X=predictors, y=response)
    features_selected = selector.get_feature_names_out()
    transformed_tabular = selector.transform(predictors)
    transformed_tabular = pd.DataFrame(transformed_tabular, columns=features_selected)
    transformed_tabular["idx"] = ehr["idx"]

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
    modal_feat_dict["EHR"] = transformed_tabular.drop(columns=["idx", "label"]).columns
    modal_feat_dict["IMAGE"] = slices_df.drop(columns=["idx"]).columns

    return radfusion, modal_feat_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="The random seed to use for all processes")

    args = parser.parse_args()
    seed = args.seed

    if seed is not None:
        random.seed(seed)

    prepared_data, modal_feat_dict = preprocessing()
    prepared_data.to_csv("data/RADFUSION/processed_standard_data.csv")
    np.save("data/RADFUSION/modal_feat_dict.npy", modal_feat_dict)
    print("Done.")