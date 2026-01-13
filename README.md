# PredikcijaBrzineUsvajanjaDubokoUcenje

Pet Adoption Speed Prediction (Multimodal Deep Learning)
This project implements a Multimodal Deep Learning system to predict the speed at which pets are adopted from the PetFinder platform. The core innovation lies in a hybrid architecture that simultaneously processes visual data (images of pets) and structured tabular data (age, breed, color, health status) to classify the AdoptionSpeed.

📖 Project Overview
The objective is to predict the adoption speed category (0-4) by leveraging both the physical appearance of the animal and its metadata. The project explores and compares three distinct neural network architectures:

Custom CNN (LeNet-5 Variation)

Transfer Learning (ResNet50)

Depthwise Separable Convolutions (Efficient Architecture)

Each model uses a Late Fusion approach, where features extracted from images are merged (Concatenate) with features from tabular data before the final classification layers.

📂 File Structure
main.py: A comprehensive script for Exploratory Data Analysis (EDA) and Image Analysis. It includes functions for:

Analyzing breeds, colors, and naming trends by state.

Image metrics: Color histograms, dominant color detection, contrast analysis, and Shannon entropy (image complexity).

lenet.py: Implementation of a custom CNN based on the LeNet-5 architecture using Tanh activation and Average Pooling.

resnet50.py: Implementation of Transfer Learning using ResNet50 pre-trained on ImageNet. It utilizes a frozen base model with a custom classification head.

separable.py: An optimized architecture using SeparableConv2D layers, significantly reducing the parameter count while maintaining performance.

sample_subset.py: A utility script to generate a smaller, balanced subset of the data (e.g., 2000 training samples) for faster iteration and debugging.

🛠️ Tech Stack
Deep Learning: TensorFlow / Keras

Data Analysis: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Image Processing: PIL (Pillow)

Dataset from: https://github.com/MilanBojic1999/RAF-PetFinder-dataset.git

🧠 Model Architecture
All implemented models follow a dual-input "Two-Stream" design:

Image Branch: Takes a 64x64x3 image as input, processes it through convolutional blocks (or a pre-trained ResNet), and outputs a feature vector.

Tabular Branch: Processes 12 standard features (Type, Age, Breed, etc.) through a series of Dense layers.

Fusion: The outputs of both branches are concatenated to create a joint representation.

Classification Head: Final Dense layers with Dropout for regularization, leading to a Softmax output layer for the 5 adoption speed classes.
