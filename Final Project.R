#import file
data=read.csv("water_potability.csv")
data
summary(data)

library(dplyr)
library(tidyr)

#changed all NAN values with its mean
data <- data %>%
  mutate_all(~ ifelse(is.na(.), mean(., na.rm = TRUE), .))
summary(data)
#standardization data
standardized_data <- data %>%
  select(-Potability) %>%  # Exclude the target variable if present
  mutate_all(scale)
data <- cbind(standardized_data, Potability = data$Potability)

summary(data)
#splitting in train_test
set.seed(12345)
data_rows <- floor(0.70 * nrow(data))
train_indices <- sample(c(1:nrow(data)), data_rows)
train_data <- data[train_indices,]
test_data <- data[-train_indices,]

library(neuralnet)

nn_model <- neuralnet(
  formula<-Potability ~ ph+ Hardness+ Solids+Chloramines+Sulfate+Conductivity+Organic_carbon+Trihalomethanes+Turbidity,
  data = train_data,
  threshold=0.5,
  hidden = c(5,3),
  err.fct = "sse",
  act.fct="logistic",
  lifesign = "full", 
  linear.output = FALSE,
  rep=10
)

plot(nn_model,rep = "best")
predictions <- predict(nn_model, newdata = test_data, type = "response")

# Convert predictions to binary outcomes (0 or 1) based on a threshold
threshold <- 0.5
binary_predictions <- ifelse(predictions > threshold, 1, 0)

# Optionally, combine predictions with the original test data
predicted_results <- data.frame(test_data, Predictions = binary_predictions)

# Print or further analyze the predictions
print(predicted_results)

print(test_data$Potability)
conf_matrix <- table(binary_predictions, test_data$Potability)
print(conf_matrix)

mse <- mean((test_data[,c("Potability")] - predictions)^2)
rmse <- sqrt(mse)

# Print or use the values as needed
print(paste("Mean Squared Error (MSE):", mse))
print(paste("Root Mean Squared Error (RMSE):", rmse))

#-------------Manually ANN-----------

# Define the sigmoid activation function and its derivative
sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

sigmoid_derivative <- function(x) {
  x * (1 - x)
}

# Define the neural network architecture
input_neurons <- ncol(train_data) - 1  # Number of input features
hidden_neurons <- 5
output_neurons <- 1

# Initialize weights with random values
set.seed(123)
w_input_hidden <- matrix(runif(input_neurons * hidden_neurons, -1, 1), nrow = input_neurons)
w_hidden_output <- matrix(runif(hidden_neurons * output_neurons, -1, 1), nrow = hidden_neurons)

# Set learning rate and number of epochs
learning_rate <- 0.01
epochs <- 1000

# Ensure train_data and test_data are numeric matrices
train_data <- as.matrix(train_data)
test_data <- as.matrix(test_data)

# Training loop (backpropagation)
for (epoch in 1:epochs) {
  # Forward propagation
  hidden_layer_input <- sigmoid(train_data[, -ncol(train_data)] %*% w_input_hidden)
  output_layer_input <- sigmoid(hidden_layer_input %*% w_hidden_output)
  
  # Calculate error
  error <- train_data[, ncol(train_data)] - output_layer_input
  
  # Backpropagation
  d_output <- error * sigmoid_derivative(output_layer_input)
  error_hidden <- d_output %*% t(w_hidden_output)
  d_hidden <- error_hidden * sigmoid_derivative(hidden_layer_input)
  
  # Update weights
  w_hidden_output <- w_hidden_output + t(hidden_layer_input) %*% d_output * learning_rate
  w_input_hidden <- w_input_hidden + t(train_data[, -ncol(train_data)]) %*% d_hidden * learning_rate
}

# Make predictions
hidden_layer_input <- sigmoid(test_data[, -ncol(test_data)] %*% w_input_hidden)
final_output <- sigmoid(hidden_layer_input %*% w_hidden_output)

# Calculate Mean Squared Error (MSE)
mse <- mean((test_data[, ncol(test_data)] - final_output)^2)
rmse <- sqrt(mse)

# Print or use the values as needed
print(paste("Mean Squared Error (MSE):", mse))
print(paste("Root Mean Squared Error (RMSE):", rmse))

# Create a histogram of predicted probabilities
predicted_probabilities <- final_output
hist(predicted_probabilities, main = "Histogram of Predicted Probabilities",
     xlab = "Predicted Probabilities", ylab = "Frequency")

# Calculate the accuracy
threshold <- 0.5  # Adjust the threshold as needed
predicted_classes <- ifelse(predicted_probabilities >= threshold, 1, 0)
actual_classes <- test_data[, ncol(test_data)]
accuracy <- sum(predicted_classes == actual_classes) / length(actual_classes)
cat("Accuracy:", accuracy, "\n")

# Plot for weights (w_input_hidden)
par(mfrow = c(1, 2))
hist(w_input_hidden, main = "Histogram of Input to Hidden Weights",
     xlab = "Weight Values", ylab = "Frequency")
hist(w_hidden_output, main = "Histogram of Hidden to Output Weights",
     xlab = "Weight Values", ylab = "Frequency")
par(mfrow = c(1, 1))

