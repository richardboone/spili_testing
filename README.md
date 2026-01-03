<div align="center" style="font-family: charter;">
<h1>SpiLiFormer: Enhancing Spiking Transformers with Lateral Inhibition</h1>
<a href="https://arxiv.org/pdf/2503.15986" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-SpiLiFormer-red?logo=arxiv" height="20" />
</a>
<a href="https://openaccess.thecvf.com/content/ICCV2025/papers/Zheng_SpiLiFormer_Enhancing_Spiking_Transformers_with_Lateral_Inhibition_ICCV_2025_paper.pdf" target="_blank">
    <img alt="ICCV 2025" src="https://img.shields.io/badge/ICCV%202025-SpiLiFormer-0077b6?logo=adobeacrobatreader&logoColor=white" height="20" />
</a>
<a href="https://openaccess.thecvf.com/content/ICCV2025/supplemental/Zheng_SpiLiFormer_Enhancing_Spiking_ICCV_2025_supplemental.pdf" target="_blank">
    <img alt="Supplementary" src="https://img.shields.io/badge/📑_Supplementary-SpiLiFormer-ffc107?color=FFCF50&logoColor=white" height="20" />
</a>

<div>
    Zeqi Zheng*<sup></sup>,</span>
    Yanchen Huang*<sup></sup>,</span>
    Yingchao Yu<sup></sup>,</span>
    Zizheng Zhu<sup></sup>,</span>
    Junfeng Tang<sup></sup>,</span>
    Zhaofei Yu<sup></sup>,</span>
    Yaochu Jin<sup>&dagger;</sup></span>
</div>

<div>
    * Equal contribution <sup>&dagger;</sup> Corresponding author&emsp;
</div>

</div>

## News
* `Jun. 2025` Our work has been accepted by [ICCV 2025](https://iccv.thecvf.com/) 🎉.
* `Jan. 2026` We release the code and model checkpoints! 🚀

## Overview
SpiLiFormer (Spiking Transformer with Lateral Inhibition) is a novel brain-inspired spiking transformer architecture designed to enhance the performance and robustness of spiking neural networks (SNNs). 

Inspired by the lateral inhibition mechanism in the human visual system, which helps the brain focus on salient regions by suppressing responses from neighboring neurons, SpiLiFormer introduces two new attention modules:

- FF-LiDiff Attention (Feedforward-pathway Lateral Differential Inhibition): Inspired by short-range inhibition in the retina, this module reduces distraction in shallow network stages by differentially inhibiting attention responses.

- FB-LiDiff Attention (Feedback-pathway Lateral Differential Inhibition): Inspired by long-range cortical inhibition, this module incorporates feedback to refine attention allocation in deeper network stages.

![SpiLiFormer poster](assets/SpiLiFormer.svg)


## Main Results on ImageNet-1K

| Methods | Type | Architecture | Input Size | Param (M) | Power (mJ) | Time Step | Top-1 Acc (%) |  Download   |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ViT | ANN | ViT-B/16 | 384 | 86.59 | 254.84 | 1 | 77.90 |  -   |
| Swin Transformer | ANN | Swin Transformer-B | 384 | 87.77 | 216.20 | 1 | 84.50 | -  |
| Spikformer | SNN | Spikformer-8-768 | 224 | 66.34 | 21.48 | 4 | 74.81 |  -  |
| QKFormer | SNN | HST-10-768 | 384 | 64.96 | 113.64 | 4 | 85.65 | - |
| E-SpikeFormer | SNN | E-SpikeFormer | 384 | 173.0 | - | 8 | 86.2 |  -  |
| **SpiLiFormer (Ours)** | **SNN** | **SpiLiFormer-10-768** | **224** | **69.10** | **11.77** | **1** | **81.54** | [link](https://huggingface.co/KirinZheng/SpiLiFormer/resolve/main/checkpoint_spiliformer_T1_224.pth) |
| **SpiLiFormer (Ours)** | **SNN** | **SpiLiFormer-10-768** | **224** | **69.10** | **44.17** | **4** | **85.82** | [link](https://huggingface.co/KirinZheng/SpiLiFormer/resolve/main/checkpoint_spiliformer_T4_224.pth) |
| **SpiLiFormer (Ours)** | **SNN** | **SpiLiFormer-10-768\*** | **288** | **69.10** | **73.52** | **4** | **86.62** | [link](https://huggingface.co/KirinZheng/SpiLiFormer/resolve/main/checkpoint_spiliformer_T4_288.pth) |
| **SpiLiFormer (Ours)** | **SNN** | **SpiLiFormer-10-768\*\*** | **384** | **69.10** | **129.45** | **4** | **86.66** | [link](https://huggingface.co/KirinZheng/SpiLiFormer/resolve/main/checkpoint_spiliformer_T4_384.pth) |

> SpiLiFormer demonstrates performance superior to current State-of-the-Art (SOTA) SNN models and even some ANN models on ImageNet-1K, while maintaining lower energy consumption and parameter counts.



## Main Results on Other Datasets (CIFAR & Neuromorphic)
| Datasets | Methods | Architecture | Param (M) | Time Step | Top-1 Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CIFAR-10** | **SpiLiFormer (Ours)** | **SpiLiFormer-4-384** | **7.04** | **4** | **96.63** |
| | QKFormer | HST-4-384 | 6.74 | 4 | 96.18 |
| | Spikformer | Spikformer-4-384 | 9.32 | 4 | 95.51 |
| **CIFAR-100** | **SpiLiFormer (Ours)** | **SpiLiFormer-4-384** | **7.04** | **4** | **81.63** |
| | QKFormer | HST-4-384 | 6.74 | 4 | 81.15 |
| | Spikingformer | Spikingformer-4-384 | 9.32 | 4 | 79.21 |
| **CIFAR10-DVS** | **SpiLiFormer (Ours)** | **SpiLiFormer-2-256** | **1.57** | **16** | **86.7** |
| | QKFormer | HST-2-256 | 1.50 | 16 | 84.0 |
| | Spikformer | Spikformer-2-256 | 2.57 | 16 | 80.9 |
| **N-Caltech101** | **SpiLiFormer (Ours)** | **SpiLiFormer-2-256** | **1.57** | **16** | **89.18** |
| | QKFormer | HST-2-256 | 1.50 | 16 | 87.24 |
| | S-Transformer | S-Transformer-2-256 | 2.57 | 16 | 86.3 |

> SpiLiFormer also achieves SOTA performance on static image datasets (CIFAR-10/100) and neuromorphic datasets (CIFAR10-DVS/N-Caltech101)


## Quick Start

### Requirements

```
timm==0.6.12
cupy==11.4.0
torch==1.12.1
spikingjelly==0.0.0.0.12
pyyaml
tensorboard
```

### Data Preparation

* **ImageNet-1K (ILSVRC 2012)**: [https://image-net.org/download.php](https://image-net.org/download.php)

* **CIFAR-10**: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

* **CIFAR-100**: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

* **CIFAR10-DVS**: [https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671)

* **N-Caltech101**: [https://data.mendeley.com/datasets/cy6cvx3ryv/1](https://data.mendeley.com/datasets/cy6cvx3ryv/1)



### Train on CIFAR-10
```
CUDA_VISIBLE_DEVICES=0 python ./cifar10/train.py \
--output ./cifar10/outputs \
--config ./cifar10/cifar10.yml \
-data-dir /your_cifar_10_dataset_filepath \
-T 4
```






### Train on CIFAR-100
```
CUDA_VISIBLE_DEVICES=0 python ./cifar100/train.py \
--output ./cifar100/outputs/ \
--config ./cifar100/cifar100.yml \
-data-dir /your_cifar_100_dataset_filepath \
-T 4
```



### Train on CIFAR10-DVS
```
CUDA_VISIBLE_DEVICES=0 python ./cifar10dvs/train.py \
--output-dir ./cifar10dvs/outputs/ \
--data-path /your_cifar_10_dvs_dataset_filepath \
--T 16
```



### Train on N-Caltech101
```
CUDA_VISIBLE_DEVICES=0 python ./ncaltech101/train.py \
--output-dir ./ncaltech101/outputs/ \
--data-path /your_ncaltech101_dataset_filepath \
--dts_cache /your_ncaltech101_dataset_filepath/dts_cache \
--T 16
```


### Evaluation on ImageNet-1K

#### SpiLiFormer-10-768, T=1, Input_size=224 

```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./imagetnet_1k/train.py \
--output_dir ./imagetnet_1k/ouputs/ \
--log_dir ./imagetnet_1k/ouputs/ \
--data_path /your_imagenet_1k_dataset_filepath \
--model SpiLiFormer_10_768 \
--input_size 224 \
--time_step 1 \
--batch_size 64 \
--accum_iter 1 \
--resume ./your_checkpoint_filepath/spiliformer_7_768_T_1_224.pth \
--eval
```


#### SpiLiFormer-10-768, T=4, Input_size=224 
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 ./imagetnet_1k/train.py \
--output_dir ./imagetnet_1k/ouputs/ \
--log_dir ./imagetnet_1k/ouputs/ \
--data_path /your_imagenet_1k_dataset_filepath \
--model SpiLiFormer_10_768 \
--input_size 224 \
--time_step 4 \
--batch_size 64 \
--accum_iter 1 \
--resume ./your_checkpoint_filepath/spiliformer_7_768_T_4_224.pth \
--eval
```


#### SpiLiFormer-10-768, T=4, Input_size=288
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 ./imagetnet_1k/train.py \
--output_dir ./imagetnet_1k/ouputs/ \
--log_dir ./imagetnet_1k/ouputs/ \
--data_path /your_imagenet_1k_dataset_filepath \
--model SpiLiFormer_10_768 \
--input_size 288 \
--time_step 4 \
--batch_size 64 \
--accum_iter 1 \
--resume ./your_checkpoint_filepath/spiliformer_7_768_T_4_288.pth \
--eval
```


#### SpiLiFormer-10-768, T=4, Input_size=384
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 ./imagetnet_1k/train.py \
--output_dir ./imagetnet_1k/ouputs/ \
--log_dir ./imagetnet_1k/ouputs/ \
--data_path /your_imagenet_1k_dataset_filepath \
--model SpiLiFormer_10_768 \
--input_size 384 \
--time_step 4 \
--batch_size 64 \
--accum_iter 1 \
--resume ./your_checkpoint_filepath/spiliformer_7_768_T_4_384.pth \
--eval
```


## Citation
If you use our code or data in this repo or find our work helpful, please consider giving a citation:

``` bibtex
@inproceedings{zheng2025spiliformer,
  title={SpiLiFormer: Enhancing Spiking Transformers with Lateral Inhibition},
  author={Zheng, Zeqi and Huang, Yanchen and Yu, Yingchao and Zhu, Zizheng and Tang, Junfeng and Yu, Zhaofei and Jin, Yaochu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={24539--24548},
  year={2025}
}
