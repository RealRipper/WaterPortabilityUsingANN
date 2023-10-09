# Water Potability Prediction Using Artificial Neural Networks

This project uses Artificial Neural Networks (ANNs) to predict water potability based on various water quality parameters. ANNs are implemented both using the neuralnet package in R and manually.

## Dataset

The project uses the "water_potability.csv" dataset, which contains information about water quality attributes and whether the water is potable or not. The dataset is loaded and preprocessed to handle missing values and standardize the data.

## Neural Network Models

### Using neuralnet Package

An ANN model is created using the neuralnet package in R. The architecture of the neural network includes two hidden layers with 5 and 3 neurons, respectively. The model is trained and evaluated, and predictions are made on a test dataset.

### Manual Implementation

An ANN model is manually implemented in R. The neural network architecture includes an input layer, a hidden layer with 5 neurons, and an output layer. The model is trained using backpropagation, and predictions are made on a test dataset.

## Evaluation

Both the neuralnet package model and the manually implemented model are evaluated in terms of Mean Squared Error (MSE) and Root Mean Squared Error (RMSE). Binary predictions are made based on a threshold of 0.5.

## Visualizations

- A histogram of predicted probabilities is created to visualize the distribution of predicted values.
- The error vs. epochs plot shows how the mean squared error changes during training.
- Weight histograms show the distribution of weights in the neural network layers.

## Heatmap

A heatmap of the entire dataset is generated to visualize the relationships between variables.

## Instructions

To run the code and reproduce the results, follow these steps:

1. Clone this repository to your local machine.
2. Ensure you have R and the required packages installed (neuralnet, dplyr, tidyr).
3. Run the R script provided to train the models and create visualizations.
4. Adjust hyperparameters and thresholds as needed.
5. Explore the results and use the models for water potability prediction.

## Files

- `water_potability.csv`: The dataset used for prediction.
- `water_potability_prediction.R`: The R script containing the code for data preprocessing, model training, and visualization.
- `README.md`: This readme file.

## Author

- Ankit Singh
