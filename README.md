# DropNet

DNNs and E-DNNs have achieved impressive results in real-world applications, but their deployment on resource- constrained devices is challenging due to high computational costs and memory requirements. To address this issue, model compression techniques are being used to accelerate DNNs while preserving their performance. Our empirical experiments have shown that DropNet, an iterative pruning approach, is robust across a range of scenarios. We have demonstrated that DropNet can remove up to 70-80% of filters in larger networks like ResNet and VGG, with no significant accuracy degradation. Furthermore, the pruned network maintains its strong performance even after weights and biases are reinitialized. Our proposed algorithm combines DropNet with a Hierarchical Ensemble pruning method that reduces the computational costs and memory requirements of E-DNNs. HQ selects a subset of high-performing E-DNNs from a set of M models, with the goal of achieving better performance than the original set while reducing the size to S. Our experiments indicate that pruned DNNs can form better ensembles than their unpruned counterparts. By combining pruned models, we were able to achieve better accuracy while also reducing computational costs and memory requirements. This algorithm is specifically tailored for deployment on resource-constrained devices.

This work is inspired from the following paper : https://arxiv.org/abs/2207.06646

**Citation:**

@inproceedings{10.5555/3524938.3525805,
author = {Min, John Tan Chong and Motani, Mehul},
title = {DropNet: reducing neural network complexity via iterative pruning},
year = {2020},
publisher = {JMLR.org},
abstract = {Modern deep neural networks require a significant amount of computing time and power to train and deploy, which limits their usage on edge devices. Inspired by the iterative weight pruning in the Lottery Ticket Hypothesis (Frankle \& Carbin, 2018), we propose DropNet, an iterative pruning method which prunes nodes/filters to reduce network complexity. DropNet iteratively removes nodes/filters with the lowest average post-activation value across all training samples. Empirically, we show that DropNet is robust across diverse scenarios, including MLPs and CNNs using the MNIST, CIFAR-10 and Tiny ImageNet datasets. We show that up to 90\% of the nodes/filters can be removed without any significant loss of accuracy. The final pruned network performs well even with reinitialization of the weights and biases. DropNet also has similar accuracy to an oracle which greedily removes nodes/filters one at a time to minimise training loss, highlighting its effectiveness.},
booktitle = {Proceedings of the 37th International Conference on Machine Learning},
articleno = {867},
numpages = {11},
series = {ICML'20}
}

This Project has been done as a part of CS 688 Machine Learning course at George Mason University under the guidance of Dr. Carlotta Domeniconi - https://cs.gmu.edu/~carlotta/ <br />
