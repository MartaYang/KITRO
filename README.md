<div align="center">
  <h1>KITRO: Refining Human Mesh by 2D Clues and Kinematic-tree Rotation <br> (CVPR 2024)</h1>
</div>

<div align="center">
  <h3><a href=https://martayang.github.io/>Fengyuan Yang</a>, <a href=https://www.comp.nus.edu.sg/~keruigu/>Kerui Gu</a>, <a href=https://www.comp.nus.edu.sg/~ayao/>Angela Yao</a></h3>
</div>

<div align="center">
  <h4> <a href=https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_KITRO_Refining_Human_Mesh_by_2D_Clues_and_Kinematic-tree_Rotation_CVPR_2024_paper.pdf>[Paper]</a>, <a href=https://openaccess.thecvf.com/content/CVPR2024/supplemental/Yang_KITRO_Refining_Human_CVPR_2024_supplemental.pdf>[Supp]</a>, <a href=http://arxiv.org/abs/2405.19833>[arXiv]</a></h4>
</div>

## 1. Requirements
* Python 3.6
* PyTorch 1.10.1


## 2. Datasets

* Download the preprocessed data from this [link](https://drive.google.com/drive/folders/1muH0UUDyVZN4DDgh_8mhi8cGcnOECQNF?usp=sharing) and put in './data' folder.
    *  the data structure KITRO desires is as following:
        ```python
        {
            'imgname'  # image name (list)
            'pred_theta'  # Predicted 3D rotation matrix (shape: [samples, 24, 3, 3])
            'pred_beta'    # Predicted body shape parameters (shape: [samples, 10])
            'pred_cam'      # Predicted camera translation (shape: [samples, 3])
            'intrinsics'  # Intrinsic camera parameters (shape: [samples, 3, 3])
            'keypoints_2d'  # Given 2D keypoints (shape: [samples, 24, 2])
            'GT_pose'        # Ground truth 3D rotation parameters (shape: [samples, 72])
            'GT_beta'        # Ground truth body shape parameters (shape: [samples, 10])
        }
        ```

## 3. Usage

* Test on 3DPW
    ```python
    python eval_KITRO.py --data_path 'data/ProcessedData_CLIFFpred_w2DKP_3dpw.pt' >> logs/runkitro_3dpw.out 2>&1
    ```
    * Corresponding output logs can found at [`logs/runkitro_3dpw.out`](logs/runkitro_3dpw.out)

* Test on Human3.6m
    ```python
    python eval_KITRO.py --data_path 'data/ProcessedData_CLIFFpred_w2DKP_HM36.pt' >> logs/runkitro_HM36.out 2>&1
    ```
    * Corresponding output logs can found at [`logs/runkitro_HM36.out`](logs/runkitro_HM36.out)

## Citation

If you find our paper or codes useful, please consider citing our paper:

```bibtex
@InProceedings{KITRO_2024,
    author    = {Yang, Fengyuan and Gu, Kerui and Yao, Angela},
    title     = {KITRO: Refining Human Mesh by 2D Clues and Kinematic-tree Rotation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {1052-1061}
}
```

## Acknowledgments

Our codes are based on [SPIN](https://github.com/nkolot/SPIN/tree/master), [CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF), [SMPLify](https://github.com/vchoutas/smplify-x/blob/master), and [HybrIK](https://github.com/Jeff-sjtu/HybrIK) and we really appreciate it. 
