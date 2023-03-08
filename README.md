# csfn

A Cross-Supervised Fusion Network for 3D Object Detection based on Score Redistribution

## Installation
 - Clone this repository
   ```
   git clone git@github.com:shangjie-li/csfn.git
   ```
 - Install PyTorch environment with Anaconda (Tested on Ubuntu 16.04 & CUDA 10.2)
   ```
   conda create -n csfn python=3.6
   conda activate csfn
   cd csfn
   pip install -r requirements.txt
   ```
 - Compile external modules
   ```
   cd csfn
   python setup.py develop
   ```

## Dataset
 - Download KITTI3D dataset: [calib](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip), [velodyne](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip), [label_2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip) and [image_2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip).
 - Organize the downloaded files and detection results as follows (MVMM and LDFMM are the test models that we use as examples)
   ```
   csfn
   ├── data
   │   ├── kitti
   │   │   │── ImageSets
   │   │   │   ├──test.txt & train.txt & trainval.txt & val.txt
   │   │   │── training
   │   │   │   ├──calib & velodyne & label_2 & image_2
   │   │   │── testing
   │   │   │   ├──calib & velodyne & image_2
   │   ├── detected_objects  # detections of KITTI label format
   │   │   │── mvmm
   │   │   │   ├──train (contains 3712 txt files) & val (contains 3769 txt files)
   │   │   │── ldfmm
   │   │   │   ├──train (contains 3712 txt files) & val (contains 3769 txt files)
   ├── helpers
   ├── ops
   ├── utils
   ```
 - Display the dataset
   ```
   python dataset_player.py
   ```

## Demo
 - Run the demo with a trained model
   ```
   python demo.py --checkpoint=checkpoints/checkpoint_epoch_10.pth
   ```

## Training
 - Train your model using the following commands
   ```
   python train.py
   ```

## Evaluation
 - Evaluate your model using the following commands
   ```
   python test.py
   ```

