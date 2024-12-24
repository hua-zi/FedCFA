# FedCFA: Alleviating Simpson’s Paradox in Model Aggregation with Counterfactual Federated Learning
> Zhonghua Jiang*, Jimin Xu*, Shengyu Zhang†, Tao Shen, Jiwei Li, Kun Kuang, Haibin Cai, Fei Wu

<h5 align=center>

[![arXiv](https://img.shields.io/badge/Arxiv-xxxx.xxxx-red?logo=arxiv&label=Arxiv&color=red)]([https://arxiv.org/abs/2406.18139](https://github.com/hua-zi/FedCFA))
[![License](https://img.shields.io/badge/Code%20License-MIT%20License-yellow)](https://github.com/hua-zi/FedCFA/LICENSE)
![GitHub Repo stars](https://img.shields.io/github/stars/hua-zi/FedCFA)
</h5>

**Abstract:** Federated learning (FL) is a promising technology for data privacy and distributed optimization, but it suffers from data imbalance and heterogeneity among clients. Existing FL methods try to solve the problems by aligning client with server model or by correcting client model with control variables. These methods excel on IID and general Non-IID data but perform mediocrely in Simpson's Paradox scenarios. Simpson's Paradox refers to the phenomenon that the trend observed on the global dataset disappears or reverses on a subset, which may lead to the fact that global model obtained through aggregation in FL does not accurately reflect the distribution of global data. Thus, we propose FedCFA, a novel FL framework employing counterfactual learning to generate counterfactual samples by replacing local data critical factors with global average data, aligning local data distributions with the global and mitigating Simpson's Paradox effects. In addition, to improve the quality of counterfactual samples, we introduce factor decorrelation (FDC) loss to reduce the correlation among features and thus improve the independence of extracted factors. We conduct extensive experiments on six datasets and verify that our method outperforms other FL methods in terms of efficiency and global model accuracy under limited communication rounds.

## Usage

### Environment Setup
```
conda create -n fedcfa python=3.8.16
conda activate fedcfa
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

git clone https://github.com/hua-zi/FedCFA.git
cd FedCFA
pip install -r requirements.txt
```

### Running the Experiments
```
cd alg
sh run.sh
```
## Citation

#### If you find our work valuable, we would appreciate your citation: 🎈

```bibtex
@inproceedings{
jiang2024fedcfa,
title={FedCFA: Alleviating Simpson's Paradox in Model Aggregation with Counterfactual Federated Learning},
author={Anonymous},
booktitle={The 39th Annual AAAI Conference on Artificial Intelligence},
year={2024}
}
```

#### The code is still being organized.🚧

