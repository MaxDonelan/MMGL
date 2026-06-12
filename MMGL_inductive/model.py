import os
import random
import sys
import tempfile

import networkx as nx
import numpy as np
import scipy.sparse as spsprs
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from collections import Counter
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.cm
import networkx as nx 
from sklearn.metrics import confusion_matrix
import torch_geometric as pyg
from torch_geometric.loader import NeighborLoader
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix

from network import *
from utils import *


class disease_dataset(Dataset):
    def __init__(self, feat, label, ind):
        super(Dataset, self).__init__()
        
        self.feat  = feat[ind]
        self.label = label[ind]
        
    def __getitem__(self, index):
        return self.feat[index], self.label[index]
    
    def __len__(self):
        return np.shape(self.feat)[0]


class EvalHelper:
    def __init__(self, input_data_dims, feat, label, hyperpm, train_index, val_index, test_index, device):
        #feat = torch.from_numpy(feat).float().to(dev)
        #label = torch.from_numpy(label).long().to(dev)
        self.dev = device
        self.hyperpm = hyperpm
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
        self.feat = feat
        self.targ = label
        self.best_acc = 0
        self.best_acc_2 = 0
        self.MF_sav = tempfile.TemporaryFile()
        self.GCMP_sav = tempfile.TemporaryFile()
        num = train_index.shape[0]
        self.trn_idx = np.array(train_index)
        self.val_idx = np.array(val_index)
        self.tst_idx = np.array(test_index)
        
        self.trn_dataset = disease_dataset(feat, label, self.trn_idx)
        self.val_dataset = disease_dataset(feat, label, self.val_idx)
        self.tst_dataset = disease_dataset(feat, label, self.tst_idx)
        
        self.trn_loader = torch.utils.data.DataLoader(self.trn_dataset, batch_size = 256, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size = 256, shuffle=False)
        self.tst_loader = torch.utils.data.DataLoader(self.tst_dataset, batch_size = 256, shuffle=False)
        
        trn_label = label[self.trn_idx]
        
        counter = Counter(trn_label)
        print(counter)
        weight = len(trn_label)/np.array(list(counter.values()))/self.n_class
        
        self.out_dim = self.d_v * self.n_head + self.modal_num**2
        self.weight = torch.from_numpy(weight).float().to(self.dev)
        if self.MF_mode == 'sum':
            self.ModalFusion = VLTransformer_Gate(input_data_dims, hyperpm).to(self.dev)
        else:
            self.ModalFusion = VLTransformer(input_data_dims, hyperpm).to(self.dev)
        self.GraphConstruct = GraphLearn(self.out_dim, th = self.th, mode = self.GC_mode).to(self.dev)
        
        if self.MP_mode == 'GCN':
            self.MessagePassing = GCN(self.out_dim, self.out_dim // 2, self.n_class, self.dropout).to(self.dev)
        # Not implemented
        # elif self.MP_mode == 'GAT':
        #     self.MessagePassing = GAT(self.out_dim, self.out_dim // 2, self.n_class, self.dropout, self.alpha, nheads = 2).to(self.dev)
        
        self.optimizer_MF = optim.Adam(self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.optimizer_GC = optim.Adam(self.GraphConstruct.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.optimizer_MP = optim.Adam(self.MessagePassing.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        
        self.ModalFusion.apply(my_weight_init)
        
    def forward(self, dataloader, dev):
        loss, prob, pred, targ = 0, [], [], []
        num_batches = len(dataloader)
        
        for i, (feat, label) in enumerate(dataloader):
            feat, label = feat.float().to(dev), label.long().to(dev)
            output, hidden, attn = self.ModalFusion(feat)
            cls_loss = F.nll_loss(output, label)
            cls_loss.backward()
            prob.extend(output.cpu().detach().numpy()[:, 1])
            pred.extend(output.argmax(1).cpu().numpy())
            targ.extend(label.cpu().numpy())
            loss += cls_loss.item()
        
        avg_loss = loss / num_batches
        acc = np.mean((np.array(pred) == np.array(targ)))
        auc = roc_auc_score(targ, prob)
        return avg_loss, acc, auc
    
    def forward_MF(self, dev, test=False):
        loss, prob, pred, targ = 0, [], [], []
        dataloader = self.trn_loader
        num_batches = len(dataloader)
        hidden_matrix = torch.empty((0)).to(dev)
        for i, (feat, label) in enumerate(dataloader):
            feat, label = feat.float().to(dev), label.long().to(dev)
            output, hidden, attn = self.ModalFusion(feat)
            cls_loss = F.nll_loss(output, label)
            prob.extend(output.cpu().detach().numpy()[:, 1])
            pred.extend(output.argmax(1).cpu().numpy())
            targ.extend(label.cpu().numpy())
            loss += cls_loss
            
            hidden_matrix = torch.cat([hidden_matrix,hidden],0)
            
        trn_acc = np.mean((np.array(pred) == np.array(targ)))
        trn_auc = roc_auc_score(targ, prob)
        trn_loss = loss.item()/num_batches
        
        adj = self.GraphConstruct(hidden_matrix)
        graph_loss = GraphConstructLoss(hidden_matrix, adj, self.hyperpm.theta_smooth, self.hyperpm.theta_degree, self.hyperpm.theta_sparsity, dev)
        adj = {'adj':adj, 'label':np.array(targ)}
        loss += graph_loss
        loss.backward()

    
        val_acc, val_auc, val_loss = None, None, None
        
        if test != False:
            loss, prob, pred, tst_targ = 0, [], [], []
            if test == 'val':
                val_loader = self.val_loader
            else:
                val_loader = self.tst_loader
            num_batches = len(val_loader)
            
            for i, (feat, label) in enumerate(val_loader):
                feat, label = feat.float().to(dev), label.long().to(dev)
                output, hidden, attn = self.ModalFusion(feat)
                cls_loss = F.nll_loss(output, label)
                prob.extend(output.cpu().detach().numpy()[:, 1])
                pred.extend(output.argmax(1).cpu().numpy())
                tst_targ.extend(label.cpu().numpy())
                loss += cls_loss.item()
                
                hidden_matrix = torch.cat([hidden_matrix,hidden],0)
            val_acc = np.mean((np.array(pred) == np.array(tst_targ)))
            val_auc = roc_auc_score(tst_targ, prob)
            val_loss = loss/num_batches
            
            targ.extend(tst_targ)
            adj = self.GraphConstruct(hidden_matrix.detach())
            adj = {'adj':adj, 'label':np.array(targ)}
            
        return (trn_acc, trn_auc, trn_loss, graph_loss.item()), (val_acc, val_auc, val_loss), hidden_matrix, adj


    def forward_graph(self, hidden_matrix, adj_dict, dev, test=False):
        loss, prob, pred, targ = 0, [], [], []
        adj, label = adj_dict['adj'], adj_dict['label']
        np.save('adj.npy', adj.cpu().detach().numpy())
        normalized_adj = normalize_adj(adj + torch.eye(adj.size(0)).to(dev))
        sp_adj = sp.coo_matrix(normalized_adj.cpu().detach().numpy())
        edge_index, edge_weight = from_scipy_sparse_matrix(sp_adj)
        G = Data(x=hidden_matrix, 
                 edge_index=edge_index, 
                 edge_attr=edge_weight, 
                 y=torch.tensor(label)).to(dev)
        
        if test:
            idx = list(range(G.num_nodes))[-len(self.tst_idx):]
        else:
            idx = list(range(G.num_nodes))
        node_loader =  NeighborLoader(G,
                                      num_neighbors= [5, 10],
                                      batch_size=64,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=0)
        num_batches = len(node_loader)
        for batch in node_loader: # batch is of type torch_geometric.data.Batch and inherits Data
            batch = batch.to(dev)
            label = batch.y
            # print(num_batches, batch.x.shape, batch.edge_index.shape, batch.edge_attr.shape, batch.y.shape)
            output = self.MessagePassing(batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr)
            num_root_nodes = batch.input_id.size(0) # Root nodes are always the first num_root_nodes in the batch
            root_output = output[:num_root_nodes]
            root_labels = batch.y[:num_root_nodes]
            cls_loss = F.nll_loss(root_output, root_labels)
            cls_loss.backward()
            prob.extend(root_output.cpu().detach().numpy()[:, 1])
            pred.extend(root_output.argmax(1).cpu().numpy())
            targ.extend(root_labels.cpu().numpy())
            loss += cls_loss.item()
            
        
        acc = np.mean((np.array(pred) == np.array(targ)))
        auc = roc_auc_score(targ, prob)
        loss = loss/num_batches
        
        return acc, auc, loss, prob, targ
            
        
    def run_epoch(self, mode, end = ''):
        dev = self.dev
        if mode == 'pre-train':
            self.ModalFusion.train()
            self.GraphConstruct.eval()
            self.MessagePassing.eval()
            
            self.optimizer_MF.zero_grad()
            self.optimizer_GC.zero_grad()
            self.optimizer_MP.zero_grad()
            trn_loss, trn_acc, trn_auc = self.forward(self.trn_loader, dev)
            self.optimizer_MF.step()
            
            print('trn-loss-MF: %.4f' % trn_loss, end=' ')
            #print('trn-acc-MF:  %.4f' % trn_acc, end=' ')
        
        if mode == 'simple-2':
            self.ModalFusion.train()
            self.GraphConstruct.train()
            
            self.optimizer_MF.zero_grad()
            (trn_acc, trn_auc, trn_loss, GC_loss), _, hidden_matrix, _ = self.forward_MF(dev)
            self.optimizer_MF.step()
            #print('trn-loss-MF: %.4f ' % trn_loss, end=' ')
            
            self.MessagePassing.train()
            
            self.optimizer_MF.zero_grad()
            self.optimizer_GC.zero_grad()
            self.optimizer_MP.zero_grad()
            _, _, hidden_matrix, adj = self.forward_MF(dev)
            acc, auc, loss, pred, targ = self.forward_graph(hidden_matrix.detach(), adj, dev)
            self.optimizer_MF.step()
            self.optimizer_GC.step()
            self.optimizer_MP.step()
            print('trn-loss: %.4f trn-loss-GC: %.4f' % (loss, GC_loss), end=' ')
            #print('acc: ', acc, 'auc: ', auc)
            

    
    def print_trn_acc(self, mode = 'pre-train'):
        print('trn-', end='')
        trn_acc, trn_auc = self._print_acc(self.trn_loader, mode, tst = False, end=' val-')
        val_acc, val_auc = self._print_acc(self.val_loader, mode, tst = 'val')
        #print('pred:',pred_val[:10], 'targ:',targ_val[:10])
        return trn_acc, val_acc

    def print_tst_acc(self, mode = 'pre-train'):
        print('tst-', end='')
        tst_acc, tst_auc = self._print_acc(self.tst_loader, mode, tst = True)
        #conf_mat = confusion_matrix(targ_tst.detach().cpu().numpy(), pred_tst.detach().cpu().numpy())
        return tst_acc, tst_auc
    

    def _print_acc(self, eval_idx, mode, tst = False, end='\n'):
        self.ModalFusion.eval()
        self.GraphConstruct.eval()
        self.MessagePassing.eval()
        if mode == 'pre-train':
            loss, acc, auc = self.forward(eval_idx, self.dev)
        elif mode == 'simple-2':
            _, _, hidden_matrix, adj = self.forward_MF(self.dev, test = tst)
            acc, auc, loss = self.forward_graph(hidden_matrix.detach(), adj, self.dev, test = tst)
            
            
        print('auc: %.4f  acc: %.4f' % (auc, acc), end=end)
        return acc, auc
        
        
    def get_balanced_cutoff(targ, prob):
        fpr, tpr, thres = roc_curve(targ, prob)
        tnr = 1 - fpr
        balanced_thres = np.argmin(abs(tpr - tnr))
        return fpr[balanced_thres], tpr[balanced_thres], thres[balanced_thres]
    
    def get_class_probabilities(self):
        self.ModalFusion.eval()
        self.GraphConstruct.eval()
        self.MessagePassing.eval()
        with torch.no_grad():
            _, _, hidden_matrix, adj = self.forward_MF(self.dev, test = "val")
            _, _, _, pred, targ = self.forward_graph(hidden_matrix.detach(), adj, self.dev, test = "val")
            return pred, targ
        
