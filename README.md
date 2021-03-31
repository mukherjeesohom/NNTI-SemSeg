# NNTI-SemSeg

In this project, we address the computer vision task of semantic segmentation using deep learning-based approaches. In the first task, we train ENet architecture on the Pascal VOC 2012 dataset and obtain mean IoU of 27.75 \% on the validation set. For the second task and third tasks, we train R2Unet and PSPNet architectures on Cityscapes dataset respectively, and obtain mean IoU of 33.65 \% and 75.15 \% respectively, on the validation set.

## Task 1

The `ipynb` file for Task 1 can be found under `task-1`.We did not obtain desired plots for evaluations metrics as well as  proper visualizations for output segmentation maps in this `.ipynb` notebook.  So, we provide evaluation metrics and output segmentation maps using a  different dataloader (analogous to ones we used for Tasks 2 and 3) in our report. The code for ENet using this data loader can be found in the git repository under the folder `task-2-3`.

## Task 2 and Task 3

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
  
### Datasets

- **Pascal VOC:** We download the [original dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), and extract it to obtain a folder which contains he image sets, the XML annotation for both object detection and segmentation, and JPEG images which is named as `VOCtrainval_11-May-2012/VOCdevkit/VOC2012` Following this, we use [Semantic Contours from Inverse Detectors](http://home.bharathh.info/pubs/pdfs/BharathICCV2011.pdf)to augment the dataset.  We navigate to `/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation` and add the image sets (`train_aug`, `trainval_aug`, `val_aug` and `test_aug`) which is downloaded from this link: [Aug ImageSets](https://www.dropbox.com/sh/jicjri7hptkcu6i/AACHszvCyYQfINpRI1m5cNyta?dl=0&lst=). After this step, we add new annotatations   `VOCtrainval_11-May-2012/VOCdevkit/VOC2012` which we downloaded from [SegmentationClassAug](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0). We now use `VOCtrainval_11-May-2012` as the training path.  


- **CityScapes**: From the official website [cityscapes-dataset.com](https://www.cityscapes-dataset.com/downloads/) we download the images (images `leftImg8bit_trainvaltest.zip`)and annotations(Fine `gtFine_trainvaltest.zip` and Coarse `gtCoarse.zip annotations`) and use the same folder for extraction and then specify this path in `config.json` for training.

### Training

For training, we download the dataset and place in the directory structure outlined above. Then we choose the desired architecture
and training hyperparameters and the correct path to the dataset directory in the `config.json` file. Following is an example command
we run for training:
  
```bash
python train.py --config ./config-files/config_pspnet_cityscapes.json
```
The TensorBoard log files will be saved in `saved/runs` and the `.pth` model chekpoints in `saved/`. 
We can monitor the training using TensorBoard by running:

```bash
tensorboard --logdir saved/runs/PSPNet/03-28_23-02
```
  
### Inference

For inference, we need a trained PyTorch model, test images, and the config file used for training:

```bash
python inference.py --config config.json --model best_model.pth --images images_folder
```

Following parameters availble for inference:
```
--output       The folder where the results will be saved (default: outputs).
--extension    The extension of the images to segment (default: jpg).
--images       Folder containing the images to segment.
--model        Path to the trained model.
--mode         Mode for inference `multiscale` or `sliding`
--config       The config file used for training the model.
```


### Results

| Model  	| Backbone  	| Dataset    	| mIoU     	| Pretrained Weights                                                                                 	| Tensorboard                                                                                        	| Evaluation Metrics                                                                                 	|
|--------	|-----------	|------------	|----------	|----------------------------------------------------------------------------------------------------	|----------------------------------------------------------------------------------------------------	|----------------------------------------------------------------------------------------------------	|
| ENet   	|     -     	| Pascal VOC 	| 27.75 \% 	| [Google Drive](https://drive.google.com/file/d/14EdSNK7C6-h8_Amvc4TrzTGUby5ANhyH/view?usp=sharing) 	| [Google Drive](https://drive.google.com/file/d/1kPDkYR_RyF0SCklLI4Ipt1lXkHPFL6D3/view?usp=sharing) 	| [Google Drive](https://drive.google.com/file/d/1xXSfXNWPhyCnLIqYWP5yhaToFrTKfl7l/view?usp=sharing) 	|
| R2UNet 	|     -     	| Cityscapes 	| 33.65 \% 	| [Google Drive](https://drive.google.com/file/d/1HBX-5yVPftpYgHuAf-ENTaVXUjfdMdLF/view?usp=sharing) 	| [Google Drive](https://drive.google.com/file/d/1ADUbWKzv9tlsUy61JkcBYQUfLMJ_vfzX/view?usp=sharing) 	| [Google Drive](https://drive.google.com/file/d/1Zx_LWMqLMirKPIEImUmcr-yuOJJbGZF_/view?usp=sharing) 	|
| PSPNet 	| ResNet-50 	| Cityscapes 	| 75.15 \% 	| [Google Drive](https://drive.google.com/file/d/1UuO3wCXNJMrTNxxHBpu8KlVLFpY6jNcJ/view?usp=sharing) 	| [Google Drive](https://drive.google.com/file/d/10hn50-K0fYHtL0lc-U1pgCNelp813lem/view?usp=sharing) 	| [Google Drive](https://drive.google.com/file/d/1EPDJke8Dl7M4V7yEsrMwHiowOPexpY-a/view?usp=sharing) 	|

 ## Contributors
 
 This code has been written for the Neural Networks: Theory and Implementation (NNTI) course project at Saarland University for Winter Semester 2020-21. Following 
 are the contributors:
 
 - [Sohom Mukherjee](https://github.com/mukherjeesohom) (Student Number: 7010515)
 - [Shayari Bhattacharjee](https://github.com/shayari21) (Student Number: 7009998)
