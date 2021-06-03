## [Joint Noise-Tolerant Learning and Meta Camera Shift Adaptation for Unsupervised Person Re-Identification](https://arxiv.org/abs/2103.04618) (CVPR'21)

### Introduction

Code for our CVPR 2021 paper "MetaCam+DSCE".

####[2021.5.24] We recorded an introduction video on [Zhidongxi](https://course.zhidx.com/c/OWYyZTMxNzJhYTA2YzEyYjZhYjM=).

![](figures/metacam.png)

### Prerequisites

- CUDA>=10.0
- At least two 1080-Ti GPUs 
- Other necessary packages listed in [requirements.txt](requirements.txt)
- Training Data
  
  (Market-1501, DukeMTMC-reID and MSMT-17. You can download these datasets from [Zhong's repo](https://github.com/zhunzhong07/ECN))

   Unzip all datasets and ensure the file structure is as follow:
   
   ```
   MetaCam_DSCE/data    
   │
   └───market1501 OR dukemtmc OR msmt17
        │   
        └───DukeMTMC-reID OR Market-1501-v15.09.15 OR MSMT17_V1
            │   
            └───bounding_box_train
            │   
            └───bounding_box_test
            | 
            └───query
            │   
            └───list_train.txt (only for MSMT-17)
            | 
            └───list_query.txt (only for MSMT-17)
            | 
            └───list_gallery.txt (only for MSMT-17)
            | 
            └───list_val.txt (only for MSMT-17)
   ```

[comment]: <> (### Before You Start)

[comment]: <> (If you are not familiar with meta-learning, I suggest that you should )

[comment]: <> (read [this code]&#40;https://github.com/AdrienLE/ANIML/blob/master/ANIML.ipynb&#41; first. )

[comment]: <> (It explains why should we use "MetaModule" &#40;"MetaConv2d", "MetaBatchNorm2d", etc.&#41; )

[comment]: <> (to replace original "Module" &#40;"Conv2d", "BatchNorm2d", etc.&#41; in Pytorch. )

[comment]: <> (Here is part of the explanation.)

[comment]: <> (![]&#40;figures/meta.png&#41;)

### Usage

See [run.sh](run.sh) for details.


### Acknowledgments
This repo borrows partially from [MWNet (meta-learning)](https://github.com/xjtushujun/meta-weight-net), 
[ECN (exemplar memory)](https://github.com/zhunzhong07/ECN) and 
[SpCL (faiss-based acceleration)](https://github.com/yxgeee/SpCL). 
If you find our code useful, please cite their papers.

```MWNet
@inproceedings{shu2019meta,
  title={Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting},
  author={Shu, Jun and Xie, Qi and Yi, Lixuan and Zhao, Qian and Zhou, Sanping and Xu, Zongben and Meng, Deyu},
  booktitle={NeurIPS},
  year={2019}
}
```

```ECN
@inproceedings{zhong2019invariance,
  title={Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identiﬁcation},
  author={Zhong, Zhun and Zheng, Liang and Luo, Zhiming and Li, Shaozi and Yang, Yi},
  booktitle={CVPR},
  year={2019},
}
```

```SpCL
@inproceedings{ge2020selfpaced,
    title={Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID},
    author={Yixiao Ge and Feng Zhu and Dapeng Chen and Rui Zhao and Hongsheng Li},
    booktitle={NeurIPS},
    year={2020}
}
```

### Citation
```yang
@inproceedings{yang2021joint,
  title={Joint Noise-Tolerant Learning and Meta Camera Shift Adaptation for Unsupervised Person Re-Identification},
  author={Yang, Fengxiang and Zhong, Zhun and Luo, Zhiming and Cai, Yuanzheng and Li, Shaozi and Nicu, Sebe},
  booktitle={CVPR},
  year={2021},
}
```


### Resources

1. Pre-trained MMT-500 models to reproduce Tab. 3 of our paper. 
   [BaiduNetDisk](https://pan.baidu.com/s/1E2d_oMBYIn5dByccLIvIAw), Passwd: nsbv.
   [Google Drive](https://drive.google.com/drive/folders/1sEi9fOeNQmrjQ4ZEQ3sbsKg26Dc6Boe1?usp=sharing).
   
2. Pedestrian images used to plot Fig.3 in our paper. 
   [BaiduNetDisk](https://pan.baidu.com/s/1c_lmWhlQ5rZ4frDJhFLwIg), Passwd: ydrf.
   [Google Drive](https://drive.google.com/file/d/1lk4DbkJR9BWpVFb_AnFUlFJPmQVrj-MX/view?usp=sharing).
   
   Please download 'marCam' and 'dukeCam', 
   put them under 'MetaCam_DSCE/data' 
   and uncomment corresponding code.
   (e.g., L#87-89, L#163-168 of train_usl_knn_merge.py)
   


### Contact Us

Email: yangfx@stu.xmu.edu.cn