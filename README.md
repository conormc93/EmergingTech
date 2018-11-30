## Emerging Technologies Project
###### Written by Conor McGrath

For my emerging technologies module in college I had to write various notebooks and a python script regarding machine learning topics.

    - NumPy.random package
    - Iris Dataset
    - MNIST Dataset
    - digitrec.py
    - Digit Recognition 

This repository holds all the solutions to my notebooks and my script to recognise hand written digits using the MNIST Dataset. 

This README will explain: how to launch each notebook and run the script also, the technologies I used, the architecture of my files and some notes on how I found completing the project.

### Thechnologies Used
The technologies I'm using in this project are:

    1. Python (3.6)
    2. Tensorflow with Keras


### How to use this repository

1. Ensure you have Python 3.x, Jupyter, Tensorflow, Keras, Numpy and Git installed locally. (I'd recommend using Anaconda to install the python packages required to run this.)
2. Enter the following commands into your command line.

```
# Change directory to anywhere you desire
cd anywhere..

# Clone this repository using git
git clone https://github.com/conormc93/EmergingTech.git
cd EmergingTech-master

# Launch the notebooks
jupyter notebook

```

### How to use digit recognition script

```
# Launch the script
python digitrec.py

```

The specification of the project was to be to train a Neural Network(NN) using a popular library such as Tensorflow and use that trained NN to predict the what number is present on a file passed in from a users input.

### Conclusion
I enjoyed working with neural networks. After having done the notebooks and script, I have gained a good understanding of machine learning techniques. I really like working with Python. The learning curve is not steep and there is plenty help and documentation if you get stuck.  

My personal aim of this project was to get to a good understanding and practice on neural networks. I wanted to challenge myself to be able to develop my own out of college projects using neural networks.

If I were to do this project again, I'd consider adding some extra features such as training another model with different datasets. I would also have liked to create a web application that allows users to alter between hand-drawn images to images from the dataset. I would also play around with the layers in my neural network and play around with some dropouts to take care of any overfittings that may happen on my current NN.

