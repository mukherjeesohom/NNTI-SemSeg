# NNTI-SemSeg

In this project, we address the computer vision task of semantic segmentation using deep learning-based approaches. In the first task, we train ENet architecture on the Pascal VOC 2012 dataset and obtain mean IoU of 27.75 \% on the validation set. For the second task and third tasks, we train R2Unet and PSPNet architectures on Cityscapes dataset respectively, and obtain mean IoU of 33.65 \% and 75.15 \% respectively, on the validation set.

## Task 1

## Task 2





### Installation Requirements

```bash
conda env create -f environment.yml
```

### Code structure
The code structure is based on [pytorch-template](https://github.com/victoresque/pytorch-template/blob/master/README.md)

```
pytorch-template/
│
├── train.py - main script to start training
├── inference.py - inference using a trained model
├── trainer.py - the main trained
├── config.json - holds configuration for training
│
├── base/ - abstract base classes
│   ├── base_data_loader.py
│   ├── base_model.py
│   ├── base_dataset.py - All the data augmentations are implemented here
│   └── base_trainer.py
│
├── dataloader/ - loading the data for different segmentation datasets
│
├── models/ - contains semantic segmentation models
│
├── saved/
│   ├── runs/ - trained models are saved here
│   └── log/ - default logdir for tensorboard and logging output
│  
└── utils/ - small utility functions
    ├── losses.py - losses used in training the model
    ├── metrics.py - evaluation metrics used
    └── lr_scheduler - learning rate schedulers 
```
  
### Training
  
```bash
python train.py --config ./config-files/config_pspnet_cityscapes.json
```
  
### Inference
  
```bash
tensorboard --logdir saved/runs/PSPNet/03-28_23-02
```

### Results

| Model  	| Backbone  	| Pretrained Weights                                                                                 	| Tensorboard                                                                                        	| Evaluation Metrics                                                                                 	|
|--------	|-----------	|----------------------------------------------------------------------------------------------------	|----------------------------------------------------------------------------------------------------	|----------------------------------------------------------------------------------------------------	|
| ENet   	|     -     	| [Google Drive](https://drive.google.com/file/d/14EdSNK7C6-h8_Amvc4TrzTGUby5ANhyH/view?usp=sharing) 	| [Google Drive](https://drive.google.com/file/d/1kPDkYR_RyF0SCklLI4Ipt1lXkHPFL6D3/view?usp=sharing) 	| [Google Drive](https://drive.google.com/file/d/1xXSfXNWPhyCnLIqYWP5yhaToFrTKfl7l/view?usp=sharing) 	|
| R2UNet 	|     -     	| [Google Drive](https://drive.google.com/file/d/1HBX-5yVPftpYgHuAf-ENTaVXUjfdMdLF/view?usp=sharing) 	| [Google Drive](https://drive.google.com/file/d/1ADUbWKzv9tlsUy61JkcBYQUfLMJ_vfzX/view?usp=sharing) 	| [Google Drive](https://drive.google.com/file/d/1Zx_LWMqLMirKPIEImUmcr-yuOJJbGZF_/view?usp=sharing) 	|
| PSPNet 	| ResNet-50 	| [Google Drive](https://drive.google.com/file/d/1UuO3wCXNJMrTNxxHBpu8KlVLFpY6jNcJ/view?usp=sharing) 	| [Google Drive](https://drive.google.com/file/d/10hn50-K0fYHtL0lc-U1pgCNelp813lem/view?usp=sharing) 	| [Google Drive](https://drive.google.com/file/d/1EPDJke8Dl7M4V7yEsrMwHiowOPexpY-a/view?usp=sharing) 	|

 ## Contributors
 
 This code was written for the Neural Networks: Theory and Implementation (NNTI) course at Saarland University for Winter Semester 2020-21. 
 
 - [Sohom Mukherjee](https://github.com/mukherjeesohom) (Student Number: 7010515)
 - [Shayari Bhattacharjee](https://github.com/shayari21) (Student Number: 7009998)
