# Intel Image Classification

## The Neural Network
This convolutional neural network predicts whether an image is of a building, glacier, forest, sea, street, or mountain . The model will predict a list of 6 elements (indices 0 - 5), where each value in the list represents the probability that the image represents one of the classes. In other words, given an input image, the model will output a list [*probability image is of a building*, *probability image is of a forest*, *probability image is of a glacier*, *probability image is of a mountain*, *probability image is of a sea*, *probability image is of a street*]. The element with the highest probability is the model's prediction. Since the model is a multiclass classification algorithm that predicts categorical values, it uses a categorical crossentropy loss function, has 6 output neurons (one for each class), and uses a standard softmax activation function. It uses a standard SGD optimizer with a learning rate of 0.001 and has a dropout layer to prevent overfitting. The model has an architecture consisting of:
- 1 Input layer (with an input shape of (256, 256, 1))
    * The images only have one color channel because they are considered grayscale
- 1 Conv2D layer (with 32 filters, a kernel size of (3, 3), and a ReLU activation function)
- 1 Conv2D layer (with 30 filters, a kernel size of (3, 3), and a ReLU activation function)
- 1 Max pooling 2D layer (with a pooling size of (2, 2) and strides of (2, 2) and "valid" padding)
- 1 Conv2D layer (with 30 filters, a kernel size of (3, 3), and a ReLU activation function)
- 1 Max pooling 2D layer (with a pooling size of (2, 2) and strides of (2, 2) and "valid" padding)
- 1 Conv2D layer (with 30 filters, a kernel size of (3, 3), and a ReLU activation function)
- 1 Flatten layer
- 3 Hidden layers (each with either 128, 64, or 32 neurons and a ReLU activation function)
- 1 Dropout layer (in between the two hidden layers and with a dropout rate of 0.2)
- 1 Output neuron (with 4 neurons and a softmax activation function)

When running the **chest_xray_cnn.py** file, you will need to input the paths of the training, testing, and validation datasets (three paths total) as strings. These paths need to be inputted where the file reads:
- " < PATH TO TRAIN SET IMAGES > " 
- " < PATH TO TEST SET IMAGES > "
- " < PATH TO VALIDATION SET IMAGES > " 

Feel free to further tune the hyperparameters or build upon the model!

## The Dataset
The dataset used here can be found at this link: https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis. Credit for the dataset collection goes to **Hubert Serowski**, **Khizar Khan**, **Adryn H.**, and others on *Kaggle*. The dataset contains approximately 6566 training images, 801 testing images. and 48 validation images (7135 images total). Note that the images from the original dataset are considered grayscale. The dataset is not included in the repository because it is too large to stabley upload to Github, so just use the link above to find and download the dataset.

## Libraries
This neural network was created with the help of the Tensorflow library.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install

## Disclaimer
Please note that I do not recommend, endorse, or encourage the use of any of my work here in actual medical use or application in any way. 
