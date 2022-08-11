# Intel Image Classification

## The Neural Network
This convolutional neural network predicts whether an image is of a building, glacier, forest, sea, street, or mountain . The model will predict a list of 6 elements (indices 0 - 5), where each value in the list represents the probability that the image represents one of the classes. In other words, given an input image, the model will output a list [*probability image is of a building*, *probability image is of a forest*, *probability image is of a glacier*, *probability image is of a mountain*, *probability image is of a sea*, *probability image is of a street*]. The element with the highest probability is the model's prediction. Since the model is a multiclass classification algorithm that predicts categorical values, it uses a categorical crossentropy loss function, has 6 output neurons (one for each class), and uses a standard softmax activation function. It uses a SGD optimizer with a learning rate of 0.001 and has a dropout layer to prevent overfitting. The model has an architecture consisting of:
- 1 Horizontal random flip layer (for image preprocessing)
- 1 VGG16 base model (with an input shape of (128, 128, 3))
- 1 Flatten layer
- 1 Dropout layer (with a dropout rate of 0.3)
- 1 Hidden layer (with 256 neurons and a ReLU activation function
- 1 Output layer (with 6 output neurons and a softmax activation function)

Note that when running the **intel_cnn.py** file, you will need to input the paths of the training and testing sets as strings — the location for where to put the paths are signified in the file with the words "< PATH TO TRAINING DATA >" and "< PATH TO TESTING DATA >." Note that when you input these paths, they should be such that — when they are concatenated with the individual elements listed in the **path_list** variable — they are complete paths. For example:
> The dataset is stored in a folder called *intel-data*, under which are the respective *train* and *test* directories that can be downloaded from the source (the link to the download site is below)
> - Thus, your file structure is something like:

>     ↓ folder1
>       ↓ folder2
>         ↓ food-data
>           ↓ train
>             ↓ buildings
>                 < Images >
>             ↓ forest
>                 < Images >
>             ↓ glacier
>                 < Images >
>             ↓ mountain
>                 < Images >
>             ↓ sea
>                 < Images >
>             ↓ street
>                 < Images >
>           ↓ test
>             ↓ buildings
>                 < Images >
>             ↓ forest
>                 < Images >
>             ↓ glacier
>                 < Images >
>             ↓ mountain
>                 < Images >
>             ↓ sea
>                 < Images >
>             ↓ street
>                 < Images >

> The paths you input should be something along the lines of: *~/folder1/folder2/intel-data/train/* and *~/folder1/folder2/intel-data/test/*, and the **path_list** variable should be set to ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'], so that when the **create_dataset()** function is running it concatenates the paths with the elements of **path_list** to produce fully coherent paths, such as *~/folder1/folder2/food-data/train/buildings*, *~/folder1/folder2/food-data/train/forest*, *~/folder1/folder2/food-data/train/glacier*, etc.

Feel free to further tune the hyperparameters or build upon the model!

## The Dataset
The dataset used here can be found at this link: https://www.kaggle.com/datasets/puneet6060/intel-image-classification. Credit for the dataset collection goes to **Vincent Liu**, **Uzair Khan**, **Puru Behl** and others on *Kaggle*. The dataset contains approximately 14064 training images and 3000 testing images (the prediction images are not used here). Note that the images from the original dataset are resized to 128 x 128 x 3 images so that they are more manageable for the model. They are considered RGB by the model since the VGG16 model only accepts images with three color channels. The dataset is not included in the repository because it is too large to stabley upload to Github, so just use the link above to find and download the dataset.

## Libraries
This neural network was created with the help of the Tensorflow library.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
