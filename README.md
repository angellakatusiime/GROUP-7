# GROUP-7
#JONAH MUTYABA
#KUGUMA VICTOR
#KATUSIIME ANGELLA
# Automated Detection of Sea Turtle Facial Patterns Using Bounding Box Frameworks

This project uses a convolutional neural network (CNN) to detect and localize facial patterns of sea turtles in images by predicting bounding boxes. We frame the object detection task as a regression problem where we predict the `x`, `y`, `width`, and `height` of bounding boxes, providing an end-to-end solution from data preparation to model training and evaluation.

#Setup and Dependencies
1. Libraries Used
   The project uses several libraries:
   - `PIL` for image manipulation
   - `pandas` for data handling
   - `matplotlib` for visualization
   - `fastai` for building and training the model

3. Environment
   This project was built on Google Colab, with data stored in Google Drive for easy access. It also requires `fastai` and `fastcore` libraries, which can be installed as follows:
   ```bash
   -!pip install --upgrade -q fastcore fastai
   
4. Mounting google drive to colab
   from google.colab import drive
drive.mount('/content/drive')

5. Data Preparation
# Image Unzipping and Folder Setup:
Copy and unzip the image dataset into the working directory.
Set image_folder as the path to the unzipped image folder.


-`!cp 'drive/My Drive/Turtle_screening/IMAGES_512.zip'`
`!unzip -q IMAGES_512.zip`
`image_folder = 'IMAGES_512'`

# Loading Train Data
Load Train.csv containing bounding box coordinates (x, y, w, h) for each image.
python
`train = pd.read_csv('drive/My Drive/Turtle_screening/Train.csv')`

6. Visualizing Sample Image with Bounding Box
Open an image using PIL and draw a bounding box around the turtle face.
Modeling
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

7. Creating the Model
The model is a CNN with ResNet34 as the architecture, with a custom loss function for bounding box regression.
We specify n_out=4 and y_range=(0, 1) to keep predictions in a normalized range.

`learn = cnn_learner(dls, resnet34, n_out=4, y_range=(0, 1),
                    loss_func=MSELossFlat(reduction='mean'),
                    metrics=iou)`
8. Training
Fine-tune the model for several epochs.
`learn.fine_tune(25)`

9. Evaluation and Prediction
# Intersection over Union (IoU)
IoU is used as a metric to evaluate bounding box predictions against ground truth.

# Prediction Visualization
For a sample image, we visualize both actual and predicted bounding boxes.

10. Submission Preparation
Load Sample.csv, generate predictions, and save the results for submission.
`Sample.to_csv('Submission2.csv', index=False)`

11. Results and Submission
The model achieves a public leaderboard score of approximately 0.12 on the test data. Predictions are saved in Submission2.csv, with absolute variance calculated to measure the prediction accuracy.
   
