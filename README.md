# GROUP-7
### Contributions of each member of the group
## 1. Kuguma Victor – Model Architecture and Configuration
   - Set up the Model Architecture: He was responsible for choosing the YOLOv5 architecture and configuring it. He determined the optimal number of layers, parameters, and GFLOPs required to ensure efficient turtle face detection.
   - Tuned Hyperparameters: He experimented with various hyperparameters (learning rate, batch size.) to fine-tune the model’s accuracy and efficiency.
   - Compiled the Model Summary: He generated and reviewed the model summary data, including layers, parameters, and gradients, to keep track of the model’s complexity.

## 2.Katusiime Angella – Data Preparation and Annotation Management
   - Prepared the Dataset: She organized the dataset into train, Val, and test subsets, ensuring a balanced distribution of images in each.
   - Transformed Data to YOLO Format: She converted annotation boxes to YOLO format, normalizing coordinates and verifying that bounding boxes were correctly formatted for the model’s requirements.
   - Conducted Quality Control on Labels: She checked for any corrupt images or those with only backgrounds and maintained a cache of labels to ensure validation consistency.

## 3. Mutyaba Jonah – Model Training, Validation, and Evaluation
   - Conducted Model Training: He trained the model and monitored the process for any issues, ensuring that the model converged as expected.
   - Handled Validation and Evaluation: He managed the validation phase, assessing metrics such as precision, recall, and map scores to evaluate the model’s performance accurately.
Note: The Fastai model was as completed as a group effort, with each team member contributing small bits on different aspects including model development, data processing, and feature engineering. 
     
# Automated Detection of Sea Turtle Facial Patterns Using Bounding Box Frameworks

This project uses a convolutional neural network (CNN) to detect and localize facial patterns of sea turtles in images by predicting bounding boxes. We frame the object detection task as a regression problem where we predict the `x`, `y`, `width`, and `height` of bounding boxes, providing an end-to-end solution from data preparation to model training and evaluation.

## Setup and Dependencies
## 1. Libraries Used
   The project uses several libraries:
   - `PIL` for image manipulation
   - `pandas` for data handling
   - `matplotlib` for visualization
   - `fastai` for building and training the model

3. Environment
   This project was built on Google Colab, with data stored in Google Drive for easy access. It also requires `fastai` and `fastcore` libraries, which can be installed as follows:
   ```bash
   -!pip install --upgrade -q fastcore fastai
   
## 4. Mounting google drive to colab
   from google.colab import drive
drive.mount('/content/drive')

## 5. Data Preparation
 Image Unzipping and Folder Setup:
Copy and unzip the image dataset into the working directory.
Set image_folder as the path to the unzipped image folder.


-`!cp 'drive/My Drive/Turtle_screening/IMAGES_512.zip'`
`!unzip -q IMAGES_512.zip`
`image_folder = 'IMAGES_512'`

 Loading Train Data
Load Train.csv containing bounding box coordinates (x, y, w, h) for each image.
python
`train = pd.read_csv('drive/My Drive/Turtle_screening/Train.csv')`

## 6. Visualizing Sample Image with Bounding Box
Open an image using PIL and draw a bounding box around the turtle face.

 Data Loading:
We define a DataBlock with ImageBlock as input and RegressionBlock(n_out=4) as target to predict the bounding box coordinates.

`dblock = DataBlock(
    blocks=(ImageBlock(), RegressionBlock(n_out=4)),
    getters=[ColReader('Image_ID', pref=f'{image_folder}/', suff='.JPG'),
             ColReader(['x', 'y', 'w', 'h'])],
    splitter=RandomSplitter(),
    item_tfms=[Resize(256)],
    n_inp=1
)`

## 7. Creating the Model
The model is a CNN with ResNet34 as the architecture, with a custom loss function for bounding box regression.
We specify n_out=4 and y_range=(0, 1) to keep predictions in a normalized range.

`learn = cnn_learner(dls, resnet34, n_out=4, y_range=(0, 1),
                    loss_func=MSELossFlat(reduction='mean'),
                    metrics=iou)`
## 8. Training
Fine-tune the model for several epochs.
`learn.fine_tune(25)`

## 9. Evaluation and Prediction
 Intersection over Union (IoU)
IoU is used as a metric to evaluate bounding box predictions against ground truth.

 Prediction Visualization
For a sample image, we visualize both actual and predicted bounding boxes.

## 10. Submission Preparation
Load Sample.csv, generate predictions, and save the results for submission.
`Sample.to_csv('Submission2.csv', index=False)`

## 11. Results and Submission
The model achieves a public leaderboard score of approximately 0.12 on the test data. Predictions are saved in Submission2.csv, with absolute variance calculated to measure the prediction accuracy.

#For the second modle, we used the YOLOv5 Neutral network model for identifying tutle faces from pictures by drawing a bounding box on the indentified face turtle face in a given Image

###Working

1.Added a new column called class to create labels that is to say; Turtle face and Background .

2.Trained a yolov5 model to indentify the turtle face from the image test data set.

3.At first used one epoch and the model was not perfoming well then Used 10 epoches in building the final model. Note the more the epoches the better the performance of the model

Performance Metrics[P (Precision)] 0.994 (99.4%) 
Precision measures how accurate the model’s positive predictions are. High precision here indicates that most detected turtle faces were indeed true positives.

R (Recall): 0.994 (99.4%) 
Recall shows how well the model identifies all relevant objects. A recall close to 1 means the model detects nearly every instance.

mAP50 (Mean Average Precision at 50% IoU): 0.995 (99.5%) 
mAP50 is a primary metric for object detection, evaluating how accurately the bounding boxes match ground-truth boxes at 50% Intersection over Union (IoU). High mAP50 indicates precise bounding boxes.

mAP50-95: 0.872 (87.2%) 
This averages the mAP over IoU thresholds from 50% to 95%. It’s a more stringent measure of accuracy. The model’s 87.2% here suggests high-quality detections at various IoU thresholds.
   
