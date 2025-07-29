### This is an official repository for SegMIC.

**Dataset**

We are organizing our benchmark **UniMedDB**, and we will release the official version. Once you have downloaded **UniMedDB.tar.gz**, unzip it and place all datasets in Meddata. Specifically, for OOD datasets, you should create a new sub-folder named "not_train" in Meddata, and put all OOD datasets into "Meddata/not_train".

**Train**

You can train SegMIC by:

`bash train_train_segm_512_base.sh`

**Inference**

You can infer SegMIC by:

`bash inference_per_label.sh`

:smile:
If you find this repository help you well, please cite our paper:
```html
@article{zhao2025segmic,
  title={SegMIC: A Universal Model for Medical Image Segmentation Through In-Context Learning},
  author={Zhao, Jianwei and Yang, Fan and Li, Xin and Jiao, Zicheng and Zhai, Qiang and Li, Xiaomeng and Wu, De and Fu, Huazhu and Cheng, Hong},
  journal={Pattern Recognition},
  pages={112179},
  year={2025},
  publisher={Elsevier}
}
```
