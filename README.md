# LayerCAM-jittor-ViT
Added visualization support for ViT in [LayerCAM-jittor](https://github.com/PengtaoJiang/LayerCAM-jittor).

[LayerCAM-jittor](https://github.com/PengtaoJiang/LayerCAM-jittor) provided an amazing way to visualize your jittor model using activation maps. However, you cannot directly use it to visualize ViT (Vision Transformer) model. The reason is that the inputs and outputs of intermediate blocks of ViT do not follow the classical `(B, C, H, W)` shape. Therefore, we need to slightly modify the codes to add support for visualizing ViT.

## Usage
```
usage: test_vit.py [-h] [--img_path IMG_PATH] [--blocks BLOCKS]
                   [--ckpt_path CKPT_PATH]

LayerCAM for Jittor

optional arguments:
  -h, --help            show this help message and exit
  --img_path IMG_PATH   Path of test image
  --blocks BLOCKS       The cam generation blocks
  --ckpt_path CKPT_PATH
                        Path of the checkpoint to load
```

## Tips
The jittor implementation of ViT is directly converted from the pytorch implementation of ViT from timm. The source codes are contained in `vision_transformer.py`. If you are using some other versions of jittor implementation, this may not work for you.

Also remember to check if the codes fit your own ViT architecture (e.g. `num_classes`).

Have a glance on the final result!

<img src="https://github.com/LovEveRv/LayerCAM-jittor-ViT/blob/master/LayerCAM-jittor/vis/img_image_06734_block_6.png" width="100%">

## Citation
```
@article{jiang2021layercam,
  title={LayerCAM: Exploring Hierarchical Class Activation Maps For Localization},
  author={Jiang, Peng-Tao and Zhang, Chang-Bin and Hou, Qibin and Cheng, Ming-Ming and Wei, Yunchao},
  journal={IEEE Transactions on Image Processing},
  year={2021},
  publisher={IEEE}
}
```
```
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}
```

