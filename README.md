# mitbih-arrhythmia-classification

Deep Learning Model: MIT-BIH Arrhythmia Dataset
This repository contains a Jupyter Notebook that demonstrates the setup, training, and evaluation of a deep learning model using the MIT-BIH Arrhythmia dataset.

 Project Purpose
The main goal of this project is to train a deep learning model on the MIT-BIH Arrhythmia dataset, which contains heart rhythm data, to classify arrhythmia types (5 classes).
The project involves building a Sequential neural network model with TensorFlow and tracking key metrics such as accuracy, loss, and epochs during the training process.

 Workflow and Steps
 1. Data Preparation

Load the MIT-BIH Arrhythmia dataset and process it using pandas

Perform data cleaning, check for missing values, and apply normalization if needed

Split the dataset into training, validation and test sets using scikit-learn (approximately 70% train, 15% validation, 15% test)

 2. Exploratory Data Analysis (EDA)

Analyze data distribution and class balance

Visualize insights using matplotlib and seaborn

 3. Deep Learning Model Setup

- Build a TensorFlow Sequential model
- Add Conv1D (convolutional layers), MaxPooling1D (pooling layers), Dropout (regularization), Flatten, and Dense layers
- Use activation functions such as ReLU and softmax
- Compile the model with an optimizer and loss function suitable for classification

 4. Model Training

Train the model on the training set over multiple epochs

Track metrics:

-Accuracy: Prediction success rate

-Loss: Deviation from true values

-Epochs: Number of full passes over the training data

 5. Model Evaluation

Measure accuracy and loss on the test set

Visualize accuracy and loss curves from the training process

(Optional) Analyze the confusion matrix or class-based performance



Test accuracy: ~96.22% 

Training loss: ~0.1898

Epochs: 20

Number of classes: 5 (different arrhythmia types)

📊 Dataset Used
Name: ECG Heartbeat Categorization Dataset (based on MIT-BIH Arrhythmia Database)

Source: https://www.kaggle.com/datasets/shayanfazeli/heartbeat


🧩 Libraries Used
numpy

pandas

matplotlib & seaborn

scikit-learn

tensorflow
