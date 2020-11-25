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
### Model architecture
PyTorch provides several pre-trained models with different architectures. 

Among them, **ResNet152** is the architecture I adopted and I redefine the last layer to output 196 values, one for each class. As it gave the best validation accuracy upon training on our data, after running various architectures for about 10 epochs.
### Train models
To train models, run following commands.
```
$ python3 training.py
```
### Hyperparameters
Batch size=32, epochs=20
#### Loss Functions
Use the nn.**CrossEntropyLoss()**
#### Optimizer
Use the **SGD** optimizer with lr=0.01, momentum=0.9.

Use the lr_scheduler.**StepLR()** with step_size=10, gamma=0.1 to adjust learning rate. 
### Plot the training_result
Use the **plot.py** to plot the training_result

---training_data loss & acc, and validation_data loss & acc
```
$ python3 plot.py
```
## Testing
Using the trained model to pred the testing_data.
```
$ python3 test_data.py
```
And get the result, save it as csv file.
## Submission
Submit the test_result csv file, get the score.

