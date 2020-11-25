# NCTU_CS_T0828_HW2-Street View House Numbers images object detection
## Introduction
The proposed challenge is a Street View House Numbers images object detection task using the SVHN dataset, which contains 33,402 trianing images, 13,068 test images.
Train dataset | Test dataset
------------ | ------------- |
33,402 images | 13,068 images
## Hardware
The following specs were used to create the original solution.
- Ubuntu 16.04 LTS
- Intel(R) Core(TM) i7-7500U CPU @ 2.90GHz
- 2x NVIDIA 2080Ti
## Data pre-process
### Get image & bounding box informations
Firstly, use the **construct_dataset.py** to read the **digitStruct.mat**, get the img_bbox_data.
```
$ python3 construct_dataset.py
```
And then, use the **getimgdata.py** to get the images width & height, merge w & h with img_bbox_data, get the all image data:
```
$ python3 getimgdata.py
```
img_name | label | left | top | width | height | right | bottom | img_width | img_height
------------ | ------------- |------------- |------------- |------------- |------------- |------------- |------------- |------------- |------------- |
 | | | | | | | | | | | |
 
 Next, use the **get_imgtxt.py** and the data from the last step to get the labels informations corresponding to each image, saved as **.txt** format as labels_dataset. 
 ```
 $ python3 get_imgtxt.py
 ```
 Because of using yolov5 , before training data, we neen to export the labels to YOLO format, like this:
- One row per object
- Each row is class x_center y_center width height format.
- Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide x_center and width by image width, and y_center and height by image height.
- Class numbers are zero-indexed (start from 0)
```
8       0.4339622641509434      0.5892857142857143      0.07547169811320754     0.4642857142857143
2       0.5283018867924528      0.5892857142857143      0.11320754716981132     0.4642857142857143
```
### Data_classes
All images are placed in two folders：train & test. 

All labels txt file are placed in one foldr.

In order to observe the effect of our trained model more conveniently, we need to divide the train_dataset into training_dataset and validation_dataset. 

So, firstly, we devide the train_dataset into training_dataset and validation_dataset by **classdataset.py**.
```
$ python3 classdataset.py
```
After deviding, the training_data becomes like this:
```
data
+- images
|	+- train 
|		 image 1
|		 image 2 ... (total 30061 images)
|	+- val	 	 
|		 image 1
|	 	 image 2 ... (total 3341 images )
+- labels
|	+- train 
|		 txt 1
|		 txt 2 ... (total 30061 images)
|	+- val	 	 
|		 txt 1
|	 	 txt 2 ... (total 3341 images )
```
Among them, the ratio of training_dataset to validation_dataset is 9:1.


## Training
### Configure the environment
Clone this repo, download tutorial dataset, and install requirements.txt dependencies, including Python>=3.8 and PyTorch>=1.7.
```
$ git clone https://github.com/ultralytics/yolov5  # clone repo
$ cd yolov5
$ pip install -r requirements.txt  # install dependencies
```
### Create dataset.yaml
```
train: ../digit/images/train/
val: ../digit/images/val/

nc: 10

names: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```
### Model architecture
YOLOv5 has s series of models from s to x, we can train models by specifying dataset, batch-size, image size and either pretrained --weights yolov5s.pt (recommended), or randomly initialized --weights '' --cfg yolov5s.yaml (not recommended). Pretrained weights are auto-downloaded from the latest YOLOv5 release.
### Train models
To train models, run following commands.
```
$ python3 train.py --img 416 --epochs 50 --data digitdata.yaml --weights yolov5l.pt --batch 32
```
### Hyperparameters
YOLOv5 has about 25 hyperparameters used for various training settings. These are defined in yaml files in the /data directory. Better initial guesses will produce better final results, so it is important to initialize these values properly before evolving. If in doubt, simply use the default values, which are optimized for YOLOv5 COCO training from scratch.
```
# Hyperparameters for COCO training from scratch 
 # python train.py --batch 40 --cfg yolov5m.yaml --weights '' --data coco.yaml --img 640 --epochs 300 
 # See tutorials for hyperparameter evolution https://github.com/ultralytics/yolov5#tutorials 
  
  
 lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3) 
 lrf: 0.2  # final OneCycleLR learning rate (lr0 * lrf) 
 momentum: 0.937  # SGD momentum/Adam beta1 
 weight_decay: 0.0005  # optimizer weight decay 5e-4 
 warmup_epochs: 3.0  # warmup epochs (fractions ok) 
 warmup_momentum: 0.8  # warmup initial momentum 
 warmup_bias_lr: 0.1  # warmup initial bias lr 
 giou: 0.05  # box loss gain 
 cls: 0.5  # cls loss gain 
 cls_pw: 1.0  # cls BCELoss positive_weight 
 obj: 1.0  # obj loss gain (scale with pixels) 
 obj_pw: 1.0  # obj BCELoss positive_weight 
 iou_t: 0.20  # IoU training threshold 
 anchor_t: 4.0  # anchor-multiple threshold 
 # anchors: 0  # anchors per output grid (0 to ignore) 
 fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5) 
 hsv_h: 0.015  # image HSV-Hue augmentation (fraction) 
 hsv_s: 0.7  # image HSV-Saturation augmentation (fraction) 
 hsv_v: 0.4  # image HSV-Value augmentation (fraction) 
 degrees: 0.0  # image rotation (+/- deg) 
 translate: 0.1  # image translation (+/- fraction) 
 scale: 0.5  # image scale (+/- gain) 
 shear: 0.0  # image shear (+/- deg) 
 perspective: 0.0  # image perspective (+/- fraction), range 0-0.001 
 flipud: 0.0  # image flip up-down (probability) 
 fliplr: 0.5  # image flip left-right (probability) 
 mosaic: 1.0  # image mosaic (probability) 
 mixup: 0.0  # image mixup (probability) 
 ```
### Plot the training_result
**Weights & Biases** (W&B) is now integrated with YOLOv5 for real-time visualization and cloud logging of training runs. This allows for better run comparison and introspection, as well improved visibility and collaboration among team members. To enable W&B logging install wandb, and then train normally (you will be guided setup on first use).
```
$ pip install wandb
```
## Testing
### creat testyolo.yaml
Creating testyolo.yaml to get the test datasets.
```
train: /home/div/cv/hw2/yolov5/digit/images/train/
val: /home/div/cv/hw2/yolov5/digit/images/val/
test: /home/div/cv/hw2/data/test/
nc: 10

names: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```
### test model
Using the trained model to pred the testing_data.
```
$  python3 test.py --weight best.pt --data testyolov.yaml --img 416 --augment --task test --save-json
```
And get the result, save it as csv file.
### transform the test result
Transform the test result to suitable .json format.
```
result.json
[dict1, dict2, ..., dict13068]
dict{“bbox”: list of bounding boxes in (y1, x1, y2, x2). (top,left,right,bottom)
     “score”: list of probability for the class
     “label”: list of label}
```
$ python3 tranjson.py
```
## Submission
Submit the test_result csv file, get the score.

