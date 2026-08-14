import argparse
import sys
import warnings
import gc

import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import pandas as pd
import torch_geometric

from network import *
from utils import *
from mmgl import *


class RedirectStdStreams:
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
    
    
def train_and_eval(datadir, datname, hyperpm):
    torch_geometric.seed_everything(hyperpm.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = datadir + datname + '/'
    modal_feat_dict = np.load(path + 'modal_feat_dict.npy', allow_pickle=True).item()
    data = pd.read_csv(path + 'processed_standard_data.csv').values
    print('data shape: ', data.shape)
    if datname == 'TADPOLE':
        hyperpm.nclass = 3
        hyperpm.nmodal = 6
    elif datname == 'ABIDE':
        hyperpm.nclass = 2
        hyperpm.nmodal = 4

    input_data_dims = []
    for i in modal_feat_dict.keys():
        input_data_dims.append(len(modal_feat_dict[i]))
    print('Modal dims ', input_data_dims)
    input_data = data[:,:-1]
    label = data[:,-1]-1
    cv = StratifiedKFold(n_splits=10, random_state=hyperpm.seed, shuffle=True)
    val_acc, tst_acc, tst_auc = [], [], []
    for fold, (train_index, test_index) in enumerate(cv.split(X=input_data, y=label)):
        mmgl = MMGL(input_data_dims=input_data_dims,
                    hyperpm=hyperpm,
                    device=dev)
        mmgl.fit(input_data, label, train_index, test_index, verbose=True)
        log_test_prob = mmgl.predict_class_prob(data[test_index])
        test_labels = np.array(label[test_index])
        test_prob = np.exp(np.array(log_test_prob))
        test_pred = np.where(test_prob > 0.5, 1, 0)
        cur_test_acc = np.mean((np.array(test_pred) == np.array(test_labels)))
        cur_test_auc = roc_auc_score(test_labels, test_prob)
        #fpr, tpr, cutoff = get_balanced_cutoff(test_labels, test_prob)

        print(f"Current Test Acc: {cur_test_acc:.4f} | Current Test AUC: {cur_test_auc:.4f}")
       # print(f"FPR: {fpr} | TPR: {tpr} | Cutoff: {cutoff}")

        tst_acc.append(cur_test_acc)
        tst_auc.append(cur_test_auc)
        if np.array(tst_acc).mean() < 0.6 and fold == 5:
            break
    
    print(f"Mean Test Acc: {np.array(tst_acc).mean():.4f} | Mean Test AUC: {np.array(tst_auc).mean():.4f}")
    print(f"Test Acc Std.: {np.array(tst_acc).std():.4f} | Test AUC Std.: {np.array(tst_auc).std():.4f}")
    
    return mmgl

def train_eval_radfusion(datadir, datname, hyperpm):
    torch_geometric.seed_everything(hyperpm.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = datadir + datname + '/'
    modal_feat_dict = np.load(path + 'modal_feat_dict.npy', allow_pickle=True).item()
    data = pd.read_csv(path + 'processed_standard_data.csv').values

    slice_level_idx = np.load(path + 'slice_level_idx.npy')
    slice_level_split = np.load(path + 'slice_level_split.npy')
    print('data shape: ', data.shape)

    hyperpm.nclass = 2
    hyperpm.nmodal = 2

    input_data_dims = []
    for i in modal_feat_dict.keys():
        input_data_dims.append(len(modal_feat_dict[i]))
    print('Modal dims ', input_data_dims)
    input_data = data[:,:-1]
    label = data[:,-1]-1
    
    train_mask = slice_level_split == "train"
    val_mask = slice_level_split == "val"
    test_mask = slice_level_split == "test"

    train_index = np.array(range(data.shape[0]))[train_mask]
    val_index = np.array(range(data.shape[0]))[val_mask]
    test_index = np.array(range(data.shape[0]))[test_mask]

    mmgl = MMGL(input_data_dims=input_data_dims,
                hyperpm=hyperpm,
                device=dev)
    mmgl.fit(input_data, label, train_index, val_index, verbose=True)
    log_test_prob = mmgl.predict_class_prob(data[test_index])
    test_labels = np.array(label[test_index])
    test_prob = np.exp(np.array(log_test_prob))
    fpr, tpr, cutoff = get_balanced_cutoff(test_labels, test_prob)
    test_pred = np.where(test_prob > cutoff, 1, 0)
    test_acc = np.mean((np.array(test_pred) == np.array(test_labels)))
    test_auc = roc_auc_score(test_labels, test_prob)

    print(f"FPR: {fpr:.4f} | TPR: {tpr:.4f} | Cutoff: {cutoff:.4f}")
    print(f"Balanced Cutoff Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f}")

    print(f"\nShape of test_prob: {test_prob.shape} | Shape of slice_level_idx: {slice_level_idx[test_index].shape}")
    to_group = pd.DataFrame({"prob": test_prob, "idx": slice_level_idx[test_index], "label": test_labels})
    grouped_prob = to_group.groupby(["idx"])["prob"].mean()
    grouped_labels = to_group.groupby(["idx"])["label"].mean()
    g_fpr, g_tpr, grouped_cutoff = get_balanced_cutoff(grouped_labels, grouped_prob)
    grouped_pred = np.where(grouped_prob > grouped_cutoff, 1, 0)
    grouped_acc = np.mean((np.array(grouped_pred) == np.array(grouped_labels)))
    grouped_auc = roc_auc_score(grouped_labels, grouped_prob)

    print("\n Results After Regrouping to the Image Level:")
    print(f"Grouped: FPR: {g_fpr:.4f} | TPR {g_tpr:.4f} | Cutoff: {grouped_cutoff:.4f}")
    print(f"Grouped Balanced Cutoff Test Acc: {grouped_acc:.4f} | Grouped Test AUC: {grouped_auc:.4f}")

    return mmgl

def main(args_str=None):
    assert float(torch.__version__[:3]) + 1e-3 >= 0.4
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./data/')
    parser.add_argument('--datname', type=str, default='TADPOLE')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Insist on using CPU instead of CUDA.')
    parser.add_argument('--nepoch', type=int, default=1000,
                        help='Max number of epochs to train.')
    parser.add_argument('--early', type=int, default=150,
                        help='Extra iterations before early-stopping.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--reg', type=float, default=0.0036,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.65,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nlayer', type=int, default=3,
                        help='Number of conv layers.')
    parser.add_argument('--n_hidden', type=int, default=16,
                        help='Number of attention head.')
    parser.add_argument('--n_head', type=int, default=8,
                        help='Number of hidden units per modal.')
    parser.add_argument('--n_iter', type=int, default=10,
                        help='Number of alternate iteration.')
    parser.add_argument('--nmodal', type=int, default=6,
                        help='Size of the sampled neighborhood.')
    parser.add_argument('--th', type=float, default=0.9,
                        help='threshold of weighted cosine')
    parser.add_argument('--GC_mode', type=str, default='adaptive-learning',
                        help='graph constrcution mode')
    parser.add_argument('--MP_mode', type=str, default='GCN',
                        help='Massage Passing mode')
    parser.add_argument('--MF_mode', type=str, default=' ',
                        help='Massage Passing mode')
    parser.add_argument('--alpha', type=float, default='0.5',
                        help='alpha for GAT')
    parser.add_argument('--theta_smooth', type=float, default='1',
                        help='graph_loss_smooth')
    parser.add_argument('--theta_degree', type=float, default='0.5',
                        help='graph_loss_degree')
    parser.add_argument('--theta_sparsity', type=float, default='0.0',
                        help='graph_loss_sparsity')
    parser.add_argument('--nclass', type=int, default=3,
                        help='class number')
    parser.add_argument('--mode', type=str, default='pre-train',
                        help='training mode')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed setting')
    if args_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_str.split())
    with RedirectStdStreams(stdout=sys.stderr):
        print('GC_mode:', args.GC_mode, 'MF_mode:', args.MF_mode)
        if args.datname == "RADFUSION":
            mmgl = train_eval_radfusion(args.datadir, args.datname, args)
        else:
            mmgl = train_and_eval(args.datadir, args.datname, args)


if __name__ == '__main__':
    # Suppress a pointless warning regarding not enabling CPU affinity, which does nothing for num_workers=0
    warnings.filterwarnings('ignore', message='.*Dataloader CPU affinity opt is not enabled.*')
    main()
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
