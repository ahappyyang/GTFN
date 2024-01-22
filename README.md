# GTFN: GCN and Transformer Fusion Network With Spatial-Spectral Features for Hyperspectral Image Classification

Aitao Yang, Min Li, Yao Ding, Danfeng Hong, Yilong Lv, Yujie He

___________

The code in this toolbox implements the ["GTFN: GCN and Transformer Fusion Network With Spatial-Spectral Features for Hyperspectral"](https://ieeexplore.ieee.org/document/10247637). 



Citation
---------------------

**Please kindly cite the papers if this code is useful and helpful for your research.**

A. Yang, M. Li, Y. Ding, D. Hong, Y. Lv and Y. He, "GTFN: GCN and Transformer Fusion Network With Spatial-Spectral Features for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-15, 2023, Art no. 6600115, doi: 10.1109/TGRS.2023.3314616.

    @ARTICLE{10247637,
            author={Yang, Aitao and Li, Min and Ding, Yao and Hong, Danfeng and Lv, Yilong and He, Yujie},
            journal={IEEE Transactions on Geoscience and Remote Sensing},
            title={GTFN: GCN and Transformer Fusion Network With Spatial-Spectral Features for Hyperspectral Image Classification},
            year={2023},
            volume={61},
            number={},
            pages={1-15},
            doi={10.1109/TGRS.2023.3314616}}

    
System-specific notes
---------------------
The codes of networks were tested using PyTorch 1.12.1 version (CUDA 10.1) in Python 3.7 on Ubuntu system.

How to use it?
---------------------
Directly run **GTFN.py** functions with different network parameter settings to produce the results. Please note that due to the randomness of the parameter initialization, the experimental results might have slightly different from those reported in the paper.

For the datasets:
Add your dataset path to function “load_dataset” in function.py

On the Indian Pines dataset, you can either re-train by following:
 `python GTFN.py --dataset='Indian' --epoch=200 --patches=9 --n_gcn=21 --pca_band=70`

On the Salinas dataset, you can either re-train by following:
 `python GTFN.py --dataset='Salinas' --epoch=200 --patches=9 --n_gcn=21 --pca_band=70`

On the PaviaU dataset, you can either re-train by following:
 `python GTFN.py --dataset='PaviaU' --epoch=200 --patches=9 --n_gcn=15 --pca_band=50`



