import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from torch_geometric.loader import NeighborLoader
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix

from network import *
from utils import *


class MultiModalDataset(Dataset):
    def __init__(self, feat, label, ind):
        super(Dataset, self).__init__()
        
        self.feat  = feat[ind]
        self.label = label[ind]
        
    def __getitem__(self, index):
        return self.feat[index], self.label[index]
    
    def __len__(self):
        return np.shape(self.feat)[0]


class MMGL:
    """
    A class to oversee the initialization, training, testing, and 
    prediction gathering, of an inductive Multi-Modal Graph Learning 
    model. 

    Initial with a list containing the number of features in each modality,
    a set of hyperparameters, and a CUDA device.

    The main change in the functionality of this MMGL class compared to previous
    classes for overseeing MMGL is that this class stores the hidden and adjacency
    matrices after _foward_model(), allowing us to indepently test and obtain
    predictions for data not provided when fit() was initially called.
    """
    def __init__(self, input_data_dims, hyperpm, device):
        self.dev = device
        self.hyperpm = hyperpm
        self.trn_mode = hyperpm.mode
        self.GC_mode = hyperpm.GC_mode
        self.MP_mode = hyperpm.MP_mode
        self.MF_mode = hyperpm.MF_mode
        self.d_v = hyperpm.n_hidden
        self.modal_num = hyperpm.nmodal
        self.n_class = hyperpm.nclass
        self.dropout = hyperpm.dropout
        self.alpha = hyperpm.alpha
        self.n_head = hyperpm.n_head
        self.th = hyperpm.th
        self.num_epoch = hyperpm.nepoch 
        self.early = hyperpm.early
        self.model_save = tempfile.TemporaryFile()
        self.out_dim = self.d_v * self.n_head + self.modal_num**2
        self.hidden_matrix = torch.Tensor()
        self.adj = torch.Tensor()

        if self.MF_mode == 'sum':
            self.ModalFusion = VLTransformer_Gate(input_data_dims, hyperpm).to(self.dev)
        else:
            self.ModalFusion = VLTransformer(input_data_dims, hyperpm).to(self.dev)
        self.GraphConstruct = GraphLearn(self.out_dim, th = self.th, mode = self.GC_mode).to(self.dev)
        
        if self.MP_mode == 'GCN':
            self.MessagePassing = GCN(self.out_dim, self.out_dim // 2, self.n_class, self.dropout).to(self.dev)
        
        self.optimizer_MF = optim.Adam(self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.optimizer_GC = optim.Adam(self.GraphConstruct.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.optimizer_MP = optim.Adam(self.MessagePassing.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        
        self.ModalFusion.apply(my_weight_init)


    def fit(self, X: np.ndarray, y: np.ndarray, train_index: np.ndarray, test_index: np.ndarray, verbose: bool = False):
        """
        Fit an MMGL model.

        Parameters
        ----------
        
        **X** : *np.ndarray*

        Data set to train with, containing both train and test indices.

        **y** : *np.ndarray*

        Response variable, as an array containing both train and test indices.

        **train_index** : *np.ndarray*

        Array of train indices.

        **test_index** : *np.ndarray*

        Array of test indices.

        **verbose** : *bool, default False*

        Whether or not to print metrics during training.
        """
        train_dataset = MultiModalDataset(X, y, train_index)
        test_dataset = MultiModalDataset(X, y, test_index)

        train_loader = DataLoader(train_dataset, batch_size = 256, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size = 256, shuffle=False)

        t = time.time()
        best_val_acc, wait_cnt = 0.0, 0
        for epoch in range(self.num_epoch):
            print(f'{epoch:3d}/{self.num_epoch:d} |', end=' ')
            self._run_epoch(train_loader, verbose)
            _, _, val_log_prob, val_labels = self._test(test_loader)
            val_prob = np.exp(np.array(val_log_prob))
            val_pred = np.where(val_prob > 0.5, 1, 0)
            cur_val_acc = np.mean((np.array(val_pred) == np.array(val_labels)))
            cur_val_auc = roc_auc_score(val_labels, val_prob)
            print(f"Val Acc: {cur_val_acc:.4f} | Val AUC: {cur_val_auc:.4f}")
            if cur_val_acc > best_val_acc:
                wait_cnt = 0
                best_val_acc = cur_val_acc
                self.model_save.close()
                self.model_save = tempfile.TemporaryFile()
                dict_list = [self.ModalFusion.state_dict(),
                             self.GraphConstruct.state_dict(),
                             self.MessagePassing.state_dict()]
                torch.save(dict_list, self.model_save)
            else:
                wait_cnt += 1
                if wait_cnt > self.early:
                    break
        print(f"time: {(time.time() - t):.4f} sec.")
        self.model_save.seek(0)
        dict_list = torch.load(self.model_save)
        self.ModalFusion.load_state_dict(dict_list[0])
        self.GraphConstruct.load_state_dict(dict_list[1])
        self.MessagePassing.load_state_dict(dict_list[2])

        _, _, log_prob, labels = self._test(test_loader)
        prob = np.exp(np.array(log_prob))
        val_pred = np.where(prob > 0.5, 1, 0)
        val_acc = np.mean((np.array(val_pred) == np.array(labels)))
        val_auc = roc_auc_score(labels, prob)
        print(f"Best Epoch Val Acc: {val_acc:.4f} | Best Epoch Val AUC: {val_auc:.4f}")


    def predict_class_prob(self, X: DataLoader | np.ndarray):
        """
        Find the class probabilities of X. Currently, returns prob[:, 1], so will only work
        correctly with binary response variables.

        Parameters
        ----------

        **X** : *DataLoader | np.ndarray*

        The data to obtain predictions for. If a DataLoader is provided, y is irrelevant 
        to calculations

        Returns
        -------

        An array of the log class probabilities for each observation.
        """
        dummy_y = np.zeros(X.shape[0])
        if type(X) != DataLoader:
            test_dataset = MultiModalDataset(X, dummy_y, range(len(dummy_y)))
            test_loader = DataLoader(test_dataset, batch_size = 256, shuffle=False)
        _, _, prob, _ = self._test(test_loader)
        return np.array(prob)


    def predict_outcome(self, X, cutoff = 0.5):
        """
        Obtain class predictions for X using a given cutoff (default 0.5).

        Parameters
        ----------

        **X** : *DataLoader | np.ndarray*

        The data to obtain predictions for. If a DataLoader is provided, y is irrelevant 
        to calculations.

        **cutoff** : *float, default 0.5*

        The classification cutoff to use for prediction.

        Returns
        -------

        The class predictions for X.
        """
        log_prob = self.predict_class_prob(X)
        return np.where(np.exp(log_prob) > cutoff, 1, 0)


    def _run_epoch(self, dataloader: DataLoader, verbose=False):
        """
        Run one epoch of training. Currently, only simple-2 is supported.
        """
        # pre-train is not yet supported for the MMGL class
        # if self.trn_mode == 'pre-train':
        #     self.ModalFusion.train()
        #     self.GraphConstruct.eval()
        #     self.MessagePassing.eval()
            
        #     self.optimizer_MF.zero_grad()
        #     self.optimizer_GC.zero_grad()
        #     self.optimizer_MP.zero_grad()
        #     trn_loss, trn_acc, trn_auc = self.forward(dataloader, self.dev)
        #     self.optimizer_MF.step()
            
        #     print('trn-loss-MF: %.4f' % trn_loss, end=' ')
        #     #print('trn-acc-MF:  %.4f' % trn_acc, end=' ')

        if self.trn_mode == 'simple-2':
            self.ModalFusion.train()
            self.GraphConstruct.train()
            
            self.optimizer_MF.zero_grad()
            self._forward_MF(dataloader)
            self.optimizer_MF.step()
            
            self.MessagePassing.train()
            
            self.optimizer_MF.zero_grad()
            self.optimizer_GC.zero_grad()
            self.optimizer_MP.zero_grad()
            self.hidden_matrix, self.adj, mf_loss, gc_loss = self._forward_MF(dataloader)
            graph_loss, log_prob, labels = self._forward_graph(self.hidden_matrix.detach(), self.adj, test=False)
            self.optimizer_MF.step()
            self.optimizer_GC.step()
            self.optimizer_MP.step()
            if verbose:
                prob = np.exp(np.array(log_prob))
                pred = np.where(prob > 0.5, 1, 0)
                train_auc = roc_auc_score(labels, prob)
                train_acc = np.mean((np.array(pred) == np.array(labels)))
                print(f'Train MF Loss: {mf_loss:.4f} | Train GC Loss : {gc_loss:.4f} | Train Graph Loss {graph_loss:.4f} |', end=' ')
                print(f'Train AUC: {train_auc:.4f} | Train Acc: {train_acc:.4f} |', end=' ')


    def _forward_MF(self, dataloader: DataLoader):
        """
        Forward overhead function for the modal fusion and graph
        construction processes.

        Returns hidden matrix, adjacency matrix + labels, total loss, and graph 
        construction loss.
        """
        loss = 0
        labels = []
        hidden_matrix = torch.empty((0)).to(self.dev)
        for feat, label in dataloader:
            feat, label = feat.float().to(self.dev), label.long().to(self.dev)
            output, hidden, attn = self.ModalFusion(feat)
            cls_loss = F.nll_loss(output, label)
            hidden_matrix = torch.cat([hidden_matrix,hidden],0)
            labels.extend(label.cpu().numpy()) 
            loss += cls_loss
        
        adj = self.GraphConstruct(hidden_matrix)
        gc_loss = GraphConstructLoss(hidden_matrix, 
                                     adj, 
                                     self.hyperpm.theta_smooth, 
                                     self.hyperpm.theta_degree, 
                                     self.hyperpm.theta_sparsity, 
                                     self.dev)
        adj = {'adj':adj, 'label':np.array(labels)}
        loss += gc_loss
        loss.backward()

        return hidden_matrix, adj, loss, gc_loss


    def _forward_graph(self, hidden_matrix, adj_dict, test=False, test_len=None):
        """
        Foward overhead function for the graphical neural network. Takes in the hidden matrix
        and adjacency matrix + labels. Importantly, _forward_graph() does not rely on the
        stored hidden matrix in self, allowing it to be called during testing.

        If test is True, only obtain predictions for test nodes.
        """
        loss = 0
        prob, labels = [], []
        adj, adj_label = adj_dict['adj'], adj_dict['label']
        normalized_adj = normalize_adj(adj + torch.eye(adj.size(0)).to(self.dev))
        # it's a little silly to convert to sp.coo_matrix just to immediately convert back
        sp_adj = sp.coo_matrix(normalized_adj.cpu().detach().numpy())
        edge_index, edge_weight = from_scipy_sparse_matrix(sp_adj)
        G = Data(x=hidden_matrix, 
                 edge_index=edge_index, 
                 edge_attr=edge_weight, 
                 y=torch.tensor(adj_label)).to(self.dev)
        if test:
            idx = list(range(G.num_nodes))[-test_len:]
        else:
            idx = list(range(G.num_nodes))
        node_loader =  NeighborLoader(G,
                                      num_neighbors= [5, 10],
                                      input_nodes=torch.tensor(idx).to(self.dev),
                                      batch_size=64,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=0)
        num_batches = len(node_loader)
        for batch in node_loader:
            batch = batch.to(self.dev)
            output = self.MessagePassing(batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr)
            num_root_nodes = batch.input_id.size(0) # Root nodes are always the first num_root_nodes in the batch
            root_output = output[:num_root_nodes]
            root_labels = batch.y[:num_root_nodes]
            cls_loss = F.nll_loss(root_output, root_labels)
            cls_loss.backward()
            prob.extend(root_output.cpu().detach().numpy()[:, 1])
            labels.extend(root_labels.cpu().numpy())
            loss += cls_loss.item()
            
        loss = loss/num_batches
        return loss, prob, labels
       

    def _test(self, dataloader: DataLoader):
        """
        Test MMGL on a data set. Requires _forward_MF() to be called first to
        initalize the hidden and adjacency matrices.

        This function uses the ModalFusion transformer to append the test data
        to the hidden and adjacency matrices, then calls _forward_graph() in
        test mode to obtain the test class probabilities and labels.

        Since this gets hidden_matrix and adj from self, while not updating self itself,
        it can be called an arbitrary number of times from the same initial model state.
        """
        self.ModalFusion.eval()
        self.GraphConstruct.eval()
        self.MessagePassing.eval()
        with torch.no_grad():
            mf_loss = 0
            prob, test_labels = [], []
            hidden_matrix = self.hidden_matrix
            labels = list(self.adj["label"])
            
            for feat, label in dataloader:
                feat, label = feat.float().to(self.dev), label.long().to(self.dev)
                output, hidden, attn = self.ModalFusion(feat)
                cls_loss = F.nll_loss(output, label)
                hidden_matrix = torch.cat([hidden_matrix,hidden],0)
                test_labels.extend(label.cpu().numpy())
                mf_loss += cls_loss.item()
            
            labels.extend(test_labels)
            adj = self.GraphConstruct(hidden_matrix.detach())
            adj = {'adj':adj, 'label':np.array(labels)}

        # _forward_graph() calls .backwards() on cls_loss, so has to be outside torch.no_grad()
        graph_loss, prob, labels = self._forward_graph(hidden_matrix, adj, test = True, test_len=len(test_labels))
        return mf_loss, graph_loss, prob, labels
    

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