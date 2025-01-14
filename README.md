# Intro to Machine Learning - TensorFlow Project

Project code for Udacity's Intro to Machine Learning with TensorFlow Nanodegree program. In this project, you will first develop code for an image classifier built with TensorFlow, then you will convert it into a command line application.

In order to complete this project, you will need to use the GPU enabled workspaces within the classroom.  The files are all available here for your convenience, but running on your local CPU will likely not work well.

You should also only enable the GPU when you need it. If you are not using the GPU, please disable it so you do not run out of time!

### Data

The data for this project is quite large - in fact, it is so large you cannot upload it onto Github.  If you would like the data for this project, you will want download it from the workspace in the classroom.  Though actually completing the project is likely not possible on your local unless you have a GPU.  You will be training using 102 different types of flowers, where there ~20 images per flower to train on.  Then you will use your trained classifier to see if you can predict the type for new images of the flowers.

### Information
This project is part of Udacity's Intro to Machine Learning with TensorFlow Nanodegree program. The project is split into two main parts:

Part 1: Build and train an image classifier using TensorFlow.

Part 2: Convert the trained model into a command-line application that takes an input image and predicts the image's class.

In Part 1, you will develop code for an image classifier using TensorFlow and train it on a dataset of flower images. The classifier will learn to distinguish between 102 different flower species based on their visual features.
In Part 2, you will create a command-line interface to load the trained model, process a new image, and output the top predicted flower classes along with their probabilities. This allows for easy testing of new images.
### Basic usage:
'python predict.py ./test_images/orange_dahlia.jpg my_model.h5'

# Return the top 3 most likely classes:
'python predict.py ./test_images/orange_dahlia.jpg my_model.h5 --top_k 3'

# Use a label_map.json file to map labels to flower names:
'python predict.py ./test_images/orange_dahlia.jpg my_model.h5 --category_names label_map.json'

Data
The dataset for this project contains images of 102 flower species, with around 20 images per species. Due to the large size, the dataset is not included here. 
It is available in the Udacity classroom workspace, where you can download it if needed.

The classifier will use these images for training, and the command-line application will allow you to load and test new images for classification.