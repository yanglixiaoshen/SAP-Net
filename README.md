# Perceptual quality prediction on omnidirectional or 360 images (SAP-net)

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
pip install -r SAP-net_requirements.txt
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

First, you should train the WBRE module for ODI enhancement to get the pseudo reference for each impaired patch of ODI, and run the command:

```shell

python train.py

```
Every two epoches, it will save the model parameters and the optimizer as "state.pkl.epochxxx" and "optimizer_state.pkl". We choose some checkpoints of WBRE in [WBRE_saved_models](https://bhpan.buaa.edu.cn:443/link/CD2D8B0865F8D686168B22DD827A3A89). Actually, we use the "epoch444" for the following training and test procedure for IQA. Of course, you can use any of your own trained WBRE models to validate the IQA performance, due to your actual training condition. 

Second, after accomplishing the WBRE training stage, you should train the PQE and QR module, which imply the main task of IQA. Run the command:

```shell

python train3.py

```
Here, we load the pre-trained WBRE model (Epoch:444) directly and only train the PQE and QR module for IQA. We save some checkpoints of PQE and QR in [PQE_QR_saved_models](https://bhpan.buaa.edu.cn:443/link/F2FABCAC180134389FB7ACE7442A8BFC). Actually, after the epoch 50, the performance of IQA can maintain a superior level (Validation PLCC>0.9). You can conduct any of experiments to test the performance. 

### Test the SAP-net

After getting the pre-trained model of WBRE and PQE+QR, you can load the pre-trained model to validate the performance of IQA on the test set. We split 960 ODIs into train, validation and test set with the ratio of 0.75:0.05:0.2 and the the corresponding ID is saved in the "train_score.txt" , "validate_score.txt" and "test_score.txt". You can download it in [train_val_test_id](https://bhpan.buaa.edu.cn:443/link/17480AC574F1939CB136227220CD634D). Note that the "test_val_score.txt" is the combination of test and validation sets. (Actually, there is no need to download these files before running the test code, since they will be generated automatically when running the test code). Moreover, please note that the ids in these files corresponds to the row index from 0-959 in "ref_imp_ID.txt", which indicate the corresponding ODI names. Just run and the test-val predicted score will be recorded in the file "predicted_score.txt":

```shell

python demo_test.py

```

### Ablation experiments

You can modify any of the components in SAP-net to verify their contribution to IQA. In our project, we have accomplished several ablation experiments, such as w and w/o wavelet, w and w/o RSAB, etc. You can find all the codes in this repository, like train1.py corresponds to the MWCNN_NonDWT.py where down/up sampling substitute DWT/IWT.


### Citation 

If this repository can offer you help in your research, please cite the paper:

```latex

@INPROCEEDINGS{Li2021Spatial,
  author={Yang, Li and Xu, Mai and Deng, Xin and Feng, Bo},
  booktitle={2021 IEEE International Conference on Multimedia and Expo (ICME)}, 
  title={Spatial Attention-Based Non-Reference Perceptual Quality Prediction Network for Omnidirectional Images}, 
  year={2021},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/ICME51207.2021.9428390}}

```

Please enjoy it and best wishes. Plese contact with me if you have any questions about ODI-IQA dataset and SAP-net.

My email address is 13021041[at]buaa[dot]edu[dot]cn




































