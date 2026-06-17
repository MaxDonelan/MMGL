# Multi-modal Graph learning for Disease Prediction (MMGL)

This is a fork of the original MMGL code which adds support for the RadFusion dataset on pulmonary embolism. **Our modifications solely target the inductive version of MMGL; no changes were made to the Transductive version.**

Briefly, we add/change:
* Data preprocessing code for the RadFusion dataset to prepare it for MMGL
* Support for the RadFusion dataset within the inductive version of MMGL
* Switch from Deep Graph Library to PyTorch Geometric for GraphConv
* A MMGL class which allows users to (more) easily fit, obtain prediction, and further modify, the MMGL model.

### Limitations
* Our version does not support the "pre-train" mode, only "simple-2"
* While this is not specific to our version, due to the method of creating the adjacency graph, MMGL inductive is very GPU resource intensive for data sets with large numbers of observations.
* We have also only tested the "weighted-cosine" graph construction method
* While the original MMGL inductive version did not support the GAT message passing mode, ours does not either.
* We have not tested the "sum" modal fusion mode
* We have not tested inductive MMGL on the TADPOLE dataset

### Requirements
Our work was performed on Linux. We hope that creating a python environment described by `requirements.txt` will allow our code to be easily rerun. That being said, computers are computers, and if you are having trouble recreating the environment, you should only need the following to run our code:
* Python 3.11.15
* numpy 1.26.4
* torch & torchaudio 2.2.2+cu121
* torchvision 0.17.2+cu121
* CUDA 12.1
* torch-geometric 2.7.0
* scikit-learn 1.8.0
* matplotlib 3.10.8
* hyperopt 0.2.7

### Code

For more details on code, see:
* MMGL inductive: [MMGL_inductive/README.md](https://github.com/MaxDonelan/MMGL/blob/main/MMGL_inductive/README.md)
* RadFusion data preprocessing: [data/RADFUSION/README.md](https://github.com/MaxDonelan/MMGL/blob/main/data/RADFUSION/README.md)

# (Semi-)Original README
> We made some grammar edits and removed the misleading "Requirements" section.

### Introduction
We hope MMGL can act as a flexible baseline that could help you to explore more powerful variants and perform scenario-specific multi-modal adaptive graph learning for more biomedical tasks. In this work, we propose an end-to-end Multi-modal Graph Learning framework (MMGL) for disease prediction with multi-modality data sets. To effectively exploit the rich information across modalities, modality-aware representation learning is proposed to aggregate the features of each modality by leveraging the correlation and complementarity between the modalities. Furthermore, instead of defining the graph manually, the latent graph structure is captured through an effective way of adaptive graph learning. It can be jointly optimized with the prediction model, thus revealing the intrinsic connections among samples. Our model is also applicable to the scenario of inductive learning on unseen data.

For more details about MMGL, please refer to our paper [[TMI](https://ieeexplore.ieee.org/abstract/document/9733917)] [[Arxiv](https://arxiv.org/abs/2203.05880)].
![image](https://github.com/SsGood/MMGL/blob/main/img/MMGL.png)

## Code running

For MMGL Tranductive: see [[./MMGL_transductive/README.md](https://github.com/SsGood/MMGL/blob/main/MMGL_transductive/README.md)]

## Data
The data preprocessing process are provided in [[./data/{dataset}](https://github.com/SsGood/MMGL/blob/main/data/)]

If you want to use your own data, you have to provide 
* a csv.file which contains multi-modal features, and
* a multi-modal feature dict.

If you find our work useful, please consider citing： 
```
@ARTICLE{MMGL,
  author={Zheng, Shuai and Zhu, Zhenfeng and Liu, Zhizhe and Guo, Zhenyu and Liu, Yang and Yang, Yuchen and Zhao, Yao},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Multi-Modal Graph Learning for Disease Prediction}, 
  year={2022},
  volume={41},
  number={9},
  pages={2207-2216},
  doi={10.1109/TMI.2022.3159264}}
```
