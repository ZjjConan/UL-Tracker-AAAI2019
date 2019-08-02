## UL-Tracker

This repository includes Matlab code for the following paper:

**[Learning a Visual Tracker from a Single Movie without Annotation (AAAI-2019)](https://www4.comp.polyu.edu.hk/~cslzhang/paper/UL-Tracker.pdf)**

*Lingxiao Yang, David Zhang, Lei Zhang*

### Introduction
This paper proposes a method that off-line pre-learns a DCFNet [1] in an unsuperivsed manner. It formulates the optimization as a two-stage learning pipeline, which iteratively updates the tracker by alternating between target localization and network optimization. Besides, the tracker is learned from a single movie, for which the data could be easily obtained other vthan collecting thousands of video clips or millions of images.

**The trained network can be directly used in DCFNet software [2]. You can simply use the given model in 'model/' to reproduce similar results as shown in below.**

--------------------------------------------------------------------------------------

### Requirements
```
1. MatConvNet-1.0-beta24 or latest
2. git clone https://github.com/ZjjConan/UL-Tracker-AAAI2019.git
```

Please follow this [website](http://www.vlfeat.org/matconvnet/install/) to compile MatConvNet with GPU support.

***Options:*** To extract SSW proposals, we use OpenCV3.0 and [mexopencv](https://github.com/kyamagu/mexopencv), please follow the mexopencv's website to compile them with opencv extra **contrib** module.

### Preprocessed Data

We provide the extracted proposals and images for **The Shawshank Redemption** in the following links. For other movies, interested readers can pre-process these movies by the provided codes in this repository.

[BaiduYun](https://pan.baidu.com/s/1z6rxbVyTVU7sCn_BUJ-TxQ) or [GoogleDrive](https://drive.google.com/open?id=1OweBcSLZWd4QTyO3uQgOfDG96SR0QtQ1)

**Note:** This is a little difference to the original paper, we downsample each images twice for fast propossing. We found that the final performance is slightly better than original one (maybe it is because of a good seed for random initialization ^_^). Results are shown below:

|	              |   OTB-2013 (AUC) |  OTB-2015 (AUC) |  VOT-2015 (A) | VOT-2015 (R) | VOT-2015 (EAO) |
| :----           |    :----:        | :----:          | :----:        |  :----:      | :----: | 
| DCFNet [1]      |    62.24         | 58.37           | 0.53          | 1.68         | 0.2174 |
| AAAI-2019 Paper |    62.47 (1.37)  | 57.45 (0.64)    | 0.53          | 1.70         | 0.2234 |
| This Repository |    **63.22**     | **59.01**       | **0.56**      | **1.55**     | **0.2473** |    

***These comparisons are based on the DCFNet_1.0. Since DCFNet_2.0 shows significant better performance than its 1.0 version, currently we are willing to achieve more robust tracking by optimizing DCFNet_2.0.***


### How to Train

1. Get frames and proposals from the above links, and compile all required softwares.
2. Use *ul_group_frames* to get clip index for each frame.
3. Use *ul_create_imdb* to create a *imdb.mat* for training.
4. Use *ul_demo_trainnet* to train the network.

Moe detail parameter settings can be found in each *ul_demo_** files.

### New Movie
Please check *ul_demo_preprocess* to find the pre-process steps for new data.

-------------------------------------------------


### Contact

Lingxiao Yang

Email: lingxiao.yang717@gmail.com


### Citation
If you find UL-Tracker useful in your research, please consider cite:

@inproceedings{yang2019learning,
    title={Learning a Visual Tracker from a Single Movie without Annotation},
    author={Yang, Lingxiao and Zhang, David and Zhang, Lei},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    volume={33},
    pages={9095--9102},
    year={2019}
}

### Reference
[1] Wang Q, Gao J, Xing J, et al. Dcfnet: Discriminant correlation filters network for visual tracking [J]. arXiv preprint arXiv:1704.04057, 2017.

[2] https://github.com/foolwood/DCFNet
