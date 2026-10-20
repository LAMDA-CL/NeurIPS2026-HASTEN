# Hierarchical Semantic Tree Anchoring for CLIP-Based Class-Incremental Learning

<div align="center">
    <div>
        Tao Hu&emsp;Lan Li&emsp;Zhen-Hao Xie&emsp;Da-Wei Zhou
    </div>
    <div>
        School of Artificial Intelligence, Nanjing University
    </div>
    <div>
        State Key Laboratory for Novel Software Technology, Nanjing University
    </div>
</div>

<div align="center">
    <a href="https://arxiv.org/abs/2511.15633">
        <img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper">
    </a>
</div>

The PyTorch code repository for **[Hierarchical Semantic Tree Anchoring for CLIP-Based Class-Incremental Learning](https://arxiv.org/abs/2511.15633)**. If you use this code in your research, please cite:

```bibtex
@inproceedings{hu2026hierarchical,
  title={Hierarchical Semantic Tree Anchoring for CLIP-Based Class-Incremental Learning},
  author={Tao Hu and Lan Li and Zhen-Hao Xie and Da-Wei Zhou},
  booktitle = {NeurIPS},
  year      = {2026}
}
```

## 📝 Introduction

Class-Incremental Learning (CIL) enables models to learn new classes continually while preserving past knowledge. 
Recently, vision-language models like CLIP offer transferable features via multimodal pre-training, making them well-suited for CIL. 
However, real-world visual and linguistic concepts are inherently hierarchical: a textual concept like "dog" subsumes fine-grained categories such as "Labrador" and "Golden Retriever," and each category is associated with corresponding images. 
Nevertheless, existing CLIP-based CIL methods fail to explicitly capture this inherent hierarchy, leading to feature drift of fine-grained classes during incremental updates and ultimately to catastrophic forgetting. 
To address this challenge, we propose HASTEN (Hierarchical Semantic Tree Anchoring), a hierarchy-aware framework that uses semantic structure to stabilize CLIP-based CIL. 
Rather than treating classes as isolated labels, HASTEN leverages external hierarchical knowledge as structured supervision to organize visual and textual features in hyperbolic space, helping maintain parent--child relations as new tasks arrive and mitigating feature drift. 
Since the shared hyperbolic mapper is updated across tasks, we further stabilize it by constraining its updates to a null space induced by prior-task features, reducing interference with previous mappings while retaining adaptability to new classes. 
Extensive experiments on nine benchmarks show that HASTEN consistently outperforms existing methods while reducing feature drift and catastrophic forgetting.

## 🔧 Requirements

**Environment**

1 [torch 1.13.0](https://github.com/pytorch/pytorch)

2 [torchvision 0.15.0](https://github.com/pytorch/vision)

3 [open-clip 2.30.0](https://github.com/mlfoundations/open_clip/releases/tag/v2.30.0)

**Dataset**

We provide the processed datasets as follows:

- **CIFAR100**: will be automatically downloaded by the code.
- **CUB200**: Google Drive: [link](https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb/view?usp=sharing) or OneDrive [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EVV4pT9VJ9pBrVs2x0lcwd0BlVQCtSrdbLVfhuajMry-lA?e=L6Wjsc)
- **ImageNet-R**: Google Drive: [link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW)
- **ObjectNet**: Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EZFv9uaaO1hBj7Y40KoCvYkBnuUZHnHnjMda6obiDpiIWw?e=4n8Kpy) You can also refer to the [filelist](https://drive.google.com/file/d/147Mta-HcENF6IhZ8dvPnZ93Romcie7T6/view?usp=sharing) and processing [code](https://github.com/zhoudw-zdw/RevisitingCIL/issues/2#issuecomment-2280462493) if the file is too large to download.
- **Cars**: Google Drive: [link](https://drive.google.com/file/d/1D8ReAuOPenWi6SMNUrOZhbm6ViyhDHbL/view?usp=sharing  ) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EbT1XAstg51Mpy82uHM0D2EBJLrtzmr_V64jeBRjqyyTnQ?e=h6g1rM)
- **UCF**: Google Drive: [link](https://drive.google.com/file/d/1Ng4w310_VDqpKbc7eYaumXTOiDxI02Wc/view?usp=sharing) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EU2qHQXjASdLh1jIl6ihZmcB6G2KvqmSw-sTlZKDE6xPbg?e=7ezvTr)
- **Aircraft**: Google Drive: [link](https://drive.google.com/file/d/1xI5r1fU0d6Nff51HuOo5w-e4sGEP46Z2/view?usp=drive_link) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/ETVliZnmPY9AvZZgcFFJ6jMB2c7TRvcq7-gso2Aqvdl_VQ?e=pWXqdP)
- **Food**: Google Drive: [link](https://drive.google.com/file/d/1rupzXpwrbxki4l-RVmsRawhz1Cm0lDY5/view?usp=drive_link) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/Eb4xfptD4L5Egus-SiYxrIcBDH1VewLGp4kzyACGF_Na_w?e=duA3Ia)
- **SUN**: OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EcQq1-1pFulKstYtdknB4O8BGo0hnlDRarAwB4wFEgkx0Q?e=YZ0xYV)

These subsets are sampled from the original datasets. Please note that I do not have the right to distribute these datasets. If the distribution violates the license, I shall provide the filenames instead.

Set `DATA_ROOT_PATH` in [utils/data.py](utils/data.py) to your dataset directory; its default is `data/` relative to the repository root. The prepared layout is:

```text
data/
├── cifar100/
│   └── cifar-100-python/
│       ├── train
│       ├── test
│       └── meta
├── aircraft/{train,test}/
├── cars/{train,test}/
├── cub/{train,test}/
├── food101/{train,test}/
├── imagenet-r/{train,test}/
├── objectnet/{train,test}/
├── sun/{train,test}/
└── ucf/{train,test}/
```

## 💡 Running scripts

To prepare your JSON files, refer to the settings in the `exps` folder and run the following command. All main experiments from the paper are already provided in the `exps` folder, you can simply execute them to reproduce the results found in the `logs` folder.

```
python main.py --config ./exps/hasten/[configname].json
```

## 🌳 Incremental semantic trees

We provide task-wise semantic trees for the main experiments in `knowledge/`. A generation script is also available at [utils/create_hier.py](utils/create_hier.py) to construct incremental semantic trees for different seeds or task splits.

## 🎈 Acknowledgements

This implementation builds on [CIL_Survey](https://github.com/zhoudw-zdw/CIL_Survey), [PyCIL](https://github.com/G-U-N/PyCIL), [C3Box](https://github.com/LAMDA-CL/C3Box), and [OpenCLIP](https://github.com/mlfoundations/open_clip).

