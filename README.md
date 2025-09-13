
# 🗺️ Getting Started with Semantic Segmentation in PyTorch Using SMP

## SIBGRAPI 2025 - Bahia/BA 🌴


## 🤝 Authors
- João Fernando Mari - joaof.mari@ufv.br - [joaofmari.github.io](joaofmari.github.io) 
- Leandro Henrique Furtado Pinto Silva - leandro.furtado@ufv.br
- Mauricio Cunha Escarpinati - mauricio@ufu.br
- André Ricardo Backes - arbackes@yahoo.com.br

## 📖 Project Overview
This project demonstrates semantic segmentation using the [Segmentation Models PyTorch (SMP)](https://github.com/qubvel/segmentation_models.pytorch) library, supporting both binary and multiclass datasets.

## 🚀 Features
- Model initialization with multiple backbones
- Configurable training with command-line arguments
- Automatic logging, evaluation, and visualization
- Reproducible results with controlled seeds
- Supports binary and multiclass segmentation tasks
- Supports RGB, multispectral, and grayscale input images 

## 🛠️ Requirements
- PyTorch
- Segmentation Models Pytorch (SMP)
- Albumentations
- Scikit-learn
- Scikit-image
- Matplotlib
- Pillow
- Pandas

It is recommended to use Anaconda or Miniconda to manage the virtual enviroments and packages.
Follow the commands below to create a Conda environment with the latest versions of the required libraries:
```bash
conda create -n env-smp-py312 python=3.12
conda activate env-smp-py312
pip install torch torchvision torchaudio 
pip install segmentation-models-pytorch
pip install notebook
pip install matplotlib
pip install scikit-learn
pip install scikit-image
pip install albumentations
pip install ipywidgets
pip install pandas
```

## 🗂️ Datasets

For this tutorial, all datasets should be stored inside a directory named `Datasets` located in your home directory (`~`):


```
~/
└── Datasets/
    ├── 38-cloud/
    ├── DeepGlobe/
    └── FUSAR-Map/
```

### **38-Cloud Dataset**

1. Download the **38-Cloud Dataset** from:  
   [https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images](https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images)  
   
   
2. Copy the downloaded ZIP file (`archive.zip`) to the `38-cloud` directory inside `Datasets/` and unzip it. It contains some files and folders, including the directories:  
   - `38-Cloud-training/`  
   - `38-Cloud-test/`  

3. In this tutorial, we use **only the training set** (`38-Cloud-training/`), because the test set provides ground truth only for full scenes, not for image patches.

4. Run the provided script `preprocess_38-cloud.ipynb` to remove any patches that contain only black pixels.

5. If you follow the directory structure shown below, the code will run without modifications.  
```
~/
└── Datasets/
    └── 38-cloud/
        ├── 38-Cloud_training/
        |   ├── Entire_scene_gts/
        |   ├── Natural_False_Color/
        |   ├── train_blue/
        |   ├── train_green/
        |   ├── train_gt/
        |   ├── train_nir/ 
        |   └── train_red/
        └── 38-Cloud_test/
```

### **DeepGlobe Dataset**

1. Download the **DeepGlobe Land Cover Classification Dataset** from:  
   [https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)

2. Copy the downloaded ZIP file (`archive.zip`) to the `DeepGlobe` directory inside `Datasets/` and unzip it.

3. The extracted dataset contains three folders: `train`, `val`, and `test`.  
   In this tutorial, we use **only the training set**, since validation and test sets do not have ground truth annotations available.

4. If you follow the directory structure shown below, the code will run without modifications.  
```
~/
└── Datasets/
    └── DeepGlobe/
        |   ├── test/
        |   ├── train/
        |   ├── valid/
        |   ├── class_dict.csv
        |   ├── metadata.csv
        └── archive.zip
```

### **FUSAR-Map (Bonus Dataset)**

1. Download **FUSAR-Map v1.0** from:  
   [https://drive.google.com/file/d/1dfr2YRFppjoPZi3KKlnd-m6ko-o20sNM/view?usp=sharing](https://drive.google.com/file/d/1dfr2YRFppjoPZi3KKlnd-m6ko-o20sNM/view?usp=sharing)

2. Copy `Fusar-Map1.0.zip` to the `FUSAR-Map` directory inside `Datasets/` and unzip it.

3. FUSAR-Map does not come with predefined train/val/test splits.  
   - SAR images are located in:  
     ```
     FUSAR-Map/FUSAR-Map1.0/FUSAR-Map/SAR_1024/
     ```
   - Corresponding masks are located in:  
     ```
     FUSAR-Map/FUSAR-Map1.0/FUSAR-Map/LAB_1024/
     ```

4. If you follow the directory structure shown below, the code will run without modifications.  
```
~/
└── Datasets/
    └── FUSAR-Map/
        ├── FUSAR-Map1.0/
        |   ├── FUSAR-Map/
        |   |   ├── SAR_1024/
        |   |   ├── LAB_1024/
        |   |   └── datasets_info.mat 
        |   └── ... 
        └── FUSAR-Map1.0.zip  
```

Ob.: If you choose a different directory layout for any dataset, update the source code paths accordingly.

## 🏋️ How to Train and Test the Models

To reproduce the exact results reported in the paper, run the following command lines:

#### DeepGlobe Dataset:
```bash
python train-test.py --dataset deepglobe --n_classes 7 --in_channels 3 --h_size 512 --w_size 512 --model Unet --backbone resnet50 --loss crossentropy -- --da_train none --max_epochs 400 --batch_size 8 --lr 0.0001 --scheduler plateau --save_images --segmap_mode darker --ec 0

python train-test.py --dataset deepglobe --n_classes 7 --in_channels 3 --h_size 512 --w_size 512 --model Unet --backbone resnet50 --loss dice -- --da_train none --max_epochs 400 --batch_size 8 --lr 0.0001 --scheduler plateau --save_images --segmap_mode darker --ec 1

python train-test.py --dataset deepglobe --n_classes 7 --in_channels 3 --h_size 512 --w_size 512 --model Unet --backbone efficientnet-b2 --loss crossentropy -- --da_train none --max_epochs 400 --batch_size 8 --lr 0.0001 --scheduler plateau --save_images --segmap_mode darker --ec 2

python train-test.py --dataset deepglobe --n_classes 7 --in_channels 3 --h_size 512 --w_size 512 --model Unet --backbone efficientnet-b2 --loss dice -- --da_train none --max_epochs 400 --batch_size 8 --lr 0.0001 --scheduler plateau --save_images --segmap_mode darker --ec 3
```

#### 38-Cloud Dataset:
```bash
python train-test.py --dataset 38-cloud --n_classes 2 --in_channels 4 --h_size 384 --w_size 384 --model Unet --backbone resnet50 --loss crossentropy -- --da_train none --max_epochs 400 --batch_size 8 --lr 0.0001 --scheduler plateau --save_images  --ec 0

python train-test.py --dataset 38-cloud --n_classes 2 --in_channels 4 --h_size 384 --w_size 384 --model Unet --backbone efficientnet-b2 --loss dice -- --da_train none --max_epochs 400 --batch_size 8 --lr 0.0001 --scheduler plateau --save_images  --ec 0

python train-test.py --dataset 38-cloud --n_classes 2 --in_channels 4 --h_size 384 --w_size 384 --model Unet --backbone resnet50 --loss crossentropy -- --da_train none --max_epochs 400 --batch_size 8 --lr 0.0001 --scheduler plateau --save_images  --ec 0

python train-test.py --dataset 38-cloud --n_classes 2 --in_channels 4 --h_size 384 --w_size 384 --model Unet --backbone efficientnet-b2 --loss dice -- --da_train none --max_epochs 400 --batch_size 8 --lr 0.0001 --scheduler plateau --save_images --ec 0
```

To achieve the same results with the 38-Cloud dataset, run ```preprocess_28-cloud.ipynb``` to remove all-black images (i.e., images where all pixels = 0).

#### FUSAR-Map Dataset:
```bash
python train-test.py --dataset FUSAR-Map --n_classes 5 --in_channels 1 --h_size 512 --w_size 512 --model Unet --backbone resnet50 --loss crossentropy -- --da_train none --max_epochs 400 --batch_size 8 --lr 0.0001 --scheduler plateau --save_images --segmap_mode darker --ec 0

python train-test.py --dataset FUSAR-Map --n_classes 5 --in_channels 1 --h_size 512 --w_size 512 --model Unet --backbone efficientnet-b2 --loss dice -- --da_train none --max_epochs 400 --batch_size 8 --lr 0.0001 --scheduler plateau --save_images --segmap_mode darker --ec 0

python train-test.py --dataset FUSAR-Map --n_classes 5 --in_channels 1 --h_size 512 --w_size 512 --model Unet --backbone resnet50 --loss crossentropy -- --da_train none --max_epochs 400 --batch_size 8 --lr 0.0001 --scheduler plateau --save_images --segmap_mode darker --ec 0

python train-test.py --dataset FUSAR-Map --n_classes 5 --in_channels 1 --h_size 512 --w_size 512 --model Unet --backbone efficientnet-b2 --loss dice -- --da_train none --max_epochs 400 --batch_size 8 --lr 0.0001 --scheduler plateau --save_images --segmap_mode darker --ec 0
```

## Batch Training
- We also provided a spript to run a batch (```run-batch.py```) of structutred exeriments.
- One can combine diverse models, encoderes, lerning rates, batch sizes, data augmenation strategies, and run a number of exeperiments.
- The summary of each experiment is stored in a single csv file to simplify the analisys.

## 📈 Training Evaluation
- Loss history curves (training anfd validation)
- IoU, F1-score, and accuract history curves (training and validation)

## 📈 Visualization of Results
- IoU, F1-Score, Accuracy, Precision, and Recall metrics
- Per-image prediction overlays and segmentation maps

## 📝 Reports
- Evaluation metrics saved in CSV format
- Image-wise and overall aggregated metrics reports

## 🎤 Slides Presentation
⚠️ *Slides will be available soon.*

## 📄 Paper
- SIBGRAPI Digital Library Archive – SDLA:
[http://sibgrapi.sid.inpe.br/col/sid.inpe.br/sibgrapi/2025/09.12.19.17/doc/thisInformationItemHomePage.html](http://sibgrapi.sid.inpe.br/col/sid.inpe.br/sibgrapi/2025/09.12.19.17/doc/thisInformationItemHomePage.html)
- IEEE Xplore:
  - ⚠️ *Paper will be available soon.*

## ✍️ How to Cite
If this tutorial was usefull in your work, use the reference for our paper:
```
⚠️ *Reference will be available soon.*

```

## 📚 References

### Segmentation Models PyTorch (SMP):

P. Iakubovskii, “Segmentation models pytorch,” https://github.com/qubvel/segmentation models.pytorch, 2019.

### Albumentations:

A. Buslaev, V. I. Iglovikov, E. Khvedchenya, A. Parinov, M. Druzhinin, and A. A. Kalinin, “Albumentations: fast and flexible image augmentations,” Information, vol. 11, no. 2, p. 125, 2020.


### 38-Cloud Dataset:

S. Mohajerani, T. A. Krammer, and P. Saeedi, “A cloud detection algorithm for remote sensing images using fully convolutional neural networks,” in 2018 IEEE 20th International Workshop on Multimedia Signal Processing (MMSP), 2018, pp. 1–5.

S. Mohajerani and P. Saeedi, “Cloud-net: An end-to-end cloud detection algorithm for landsat 8 imagery,” in IGARSS 2019 - 2019 IEEE International Geoscience and Remote Sensing Symposium, 2019, pp.1029–1032.

### DeepGlobe Dataset:
I. Demir, K. Koperski, D. Lindenbaum, G. Pang, J. Huang, S. Basu,F. Hughes, D. Tuia, and R. Raskar, “Deepglobe 2018: A challenge toparse the earth through satellite images,” in Proceedings of the IEEEconference on computer vision and pattern recognition workshops, 2018,pp. 172–181.

### FUSAR-Map:
X. Shi, S. Fu, J. Chen, F. Wang, and F. Xu, “Object-level semantic segmentation on the high-resolution gaofen-3 fusar-map dataset,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 3107–3119, 2021.


## 📄 License
MIT License

#### Last update: September 13, 2025
