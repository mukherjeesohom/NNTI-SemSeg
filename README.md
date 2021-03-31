# NNTI-SemSeg
NNTI Project: Semantic Segmentation using Deep Learning

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

 
