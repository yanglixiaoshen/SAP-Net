# SAP-net

Spatial Attention-based Non-reference Perceptual Quality Prediction Network for Omnidirectional Images (Accepted by IEEE ICME 2021))


> 
This repository contains the official PyTorch implementation of the following paper:
> **Spatial Attention-based Non-reference Perceptual Quality Prediction Network for Omnidirectional Images (IEEE ICME 2021)**<br>
> Li Yang, Mai Xu, Xin Deng and Bo Feng (School of Electronic and Information Engineering, Beihang University)<br>
> **Paper link**: https://ieeexplore.ieee.org/document/9428390<br>
> 
> **Abstract**: *Due to the strong correlation between visual attention and perceptual quality, many methods attempt to use human saliency information for image quality assessment. Although this mechanism can get good performance, the networks require human saliency labels, which is not easily accessible for omnidirectional images (ODI). To alleviate this issue, we propose a spatial attention-based perceptual quality prediction network for non-reference quality assessment on ODIs (SAP-net). Without any human saliency labels, our network can adaptively estimate human perceptual quality on impaired ODIs through a self-attention manner, which significantly promotes the prediction performance of quality scores. Moreover, our method greatly reduces the computational complexity in quality assessment task on ODIs. Extensive experiments validate that our network outperforms 9 state-of-the-art methods for quality assessment on ODIs. The dataset and code have been available on https://github.com/yanglixiaoshen/SAP-Net.*

## Preparation

### Requriments 

First, install a new conda environment \<envs\> in Linux sys (Ubuntu 18.04+); Then, activate \<envs\> and run the following command:
```shell
pip install -r requirements.txt
```

### Datasets

<div align="center"><img width="93%" src="https://github.com/yanglixiaoshen/SAP-Net/blob/main/images/IQAdataSET.jpg" /></div>


**IQA-ODI**: A large-scale IQA dataset of ODIs (IQA-ODI) with 4 categories (Human, Indoor, Landscapes, Nature), containing 120 high quality reference ODIs and 960 ODIs with impairments in both JPEG compression and map projection. In our VR experiment, each ODI was viewed and scored by 20-30 subjects and we can obtain the final DMOS (0-100, higher indicates lower quality) by means of all subjects' MOS. The impairments conducted on each ODI is shown as (Take the reference "Human_P0.jpg" as an example):

```shell
# Dataset impairment example:

REF: human_P0.jpg                    IMP:                     DMOS:

           |                ├── QF5_ERP_human_P0.jpg         73.7899
           ├─────Mode 1─────├── QF15_ERP_human_P0.jpg        43.4527 
           |                ├── QF35_ERP_human_P0.jpg        31.8013
           |                ├── QF60_ERP_human_P0.jpg        32.6931
           |
           |                ├── QF15_cmp_human_P0.jpg        38.1707
           ├─────Mode 2─────├── QF15_cpp_human_P0.jpg        43.3672
           |                ├── QF15_isp_human_P0.jpg        43.7219
           |                ├── QF15_ohp_human_P0.jpg        38.7299 

```

If you want train and test over our ODI-IQA dataset, please download the ODIs form [ODI-IQA dataset](https://bhpan.buaa.edu.cn:443/link/FF704DD138E2C0A466AF99F5724B8310) and the corresponding important info .txt files [Info of ODI-IQA](https://bhpan.buaa.edu.cn:443/link/49AA896C49299B472047DD9D79F7FD7A).


## Implementation

The architecture of the proposed SAP-net is shown in the following figure, which contains three novel modules, i.e., WBRE, PQE and QR.

<div align="center"><img width="93%" src="https://github.com/yanglixiaoshen/SAP-Net/blob/main/images/framework5.png" /></div>



### Training the SAP-net













































contact with 

me if you have any questions about ODI-IQA dataset and SAP-net.

My email address is 13021041[at]buaa[dot]edu[dot]cn

###############################################
