# Load necessary libraries
library(readxl)
library(neuralnet)
library(Metrics)
library(ggplot2)

# Load and prepare data
usd_eur_data <- read_excel("path/ExchangeUSD (2).xlsx")
usd_to_eur_rates <- usd_eur_data[, 3]
daily_rates_ts <- ts(usd_to_eur_rates, frequency = 365)

# Normalize data
normalized_data <- scale(daily_rates_ts)

# Split data into training and evaluation sets
training_set <- normalized_data[1:400]
evaluation_set <- normalized_data[401:500]

lags <- 4  # Adjust if needed for different numbers of input lags

# Function to prepare data matrix
prepare_data <- function(data, lags) {
  matrix <- matrix(0, nrow = length(data) - lags, ncol = lags + 1)
  for (j in 1:lags) {
    matrix[, j] <- data[(lags - j + 1):(length(data) - j)]
  }
  matrix[, lags + 1] <- data[(lags + 1):length(data)]
  return(matrix)
}

# Define experiment configurations
configurations <- list(
  list(inputs=1, layers=c(4), activation="tanh", linear=TRUE),
  list(inputs=2, layers=c(4, 4), activation="tanh", linear=TRUE),
  list(inputs=3, layers=c(4, 4, 4), activation="tanh", linear=TRUE),
  list(inputs=2, layers=c(3), activation="logistic", linear=FALSE),
  list(inputs=2, layers=c(4, 4), activation="logistic", linear=TRUE),
  list(inputs=3, layers=c(3, 3, 3), activation="tanh", linear=FALSE),
  list(inputs=1, layers=c(4, 4), activation="logistic", linear=FALSE),
  list(inputs=3, layers=c(4), activation="logistic", linear=TRUE),
  list(inputs=4, layers=c(4, 4, 4), activation="tanh", linear=FALSE),
  list(inputs=5, layers=c(5, 5), activation="logistic", linear=TRUE),
  list(inputs=1, layers=c(3, 3, 3, 3), activation="tanh", linear=TRUE),
  list(inputs=2, layers=c(6, 6, 6), activation="logistic", linear=FALSE),
  list(inputs=3, layers=c(4, 4), activation="tanh", linear=TRUE),
  list(inputs=2, layers=c(5, 5, 5), activation="logistic", linear=FALSE),
  list(inputs=1, layers=c(8), activation="tanh", linear=TRUE)
)



# Initialize results dataframe
results <- data.frame(
  Inputs=integer(),
  HiddenLayers=character(),
  Neurons=character(),
  ActivationFunction=character(),
  LinearOutput=logical(),
  RMSE=numeric(),
  MAE=numeric(),
  MAPE=numeric(),
  sMAPE=numeric(),
  stringsAsFactors=FALSE
)

# Loop through each configuration and train neural networks
for (config in configurations) {
  if (config$inputs == lags) {
    cat("Lag and input are the same for this configuration.\n")
  }
  
  # Prepare training and evaluation data
  training_matrix <- prepare_data(training_set, config$inputs)
  evaluation_matrix <- prepare_data(evaluation_set, config$inputs)
  
  # Scale data
  normalized_train <- scale(training_matrix)
  normalized_eval <- scale(evaluation_matrix)
  
  # Neural network formula
  input_vars <- paste0("V", 1:config$inputs)
  target_var <- paste0("V", config$inputs + 1)
  formula <- as.formula(paste(target_var, "~", paste(input_vars, collapse=" + ")))
  
  # Train the neural network
  nn <- neuralnet(formula,
                  data = normalized_train,
                  hidden = config$layers,
                  linear.output = config$linear,
                  act.fct = config$activation,
                  algorithm = "rprop+",
                  stepmax = 1e6,  # Increase or adjust as necessary
                  rep = 1,        # Can increase to average over multiple runs
                  threshold = 0.3)  # Adjust threshold for convergence
  
  # Check if the neural network object is created correctly
  if (!inherits(nn, "nn")) {
    print("Error: The neural network object was not created successfully.")
  } else {
    # Visualization of the neural network
    plot(nn)
    plot(nn, rep = "best")
    
    # Apply the model to make predictions
    predictions <- compute(nn, normalized_eval[, 1:config$inputs, drop=FALSE])$net.result
    predictions_rescaled <- predictions * attr(normalized_eval, "scaled:scale")[1] + attr(normalized_eval, "scaled:center")[1]
    
    # Calculate metrics
    rmse_value <- rmse(predictions_rescaled, evaluation_set[(config$inputs + 1):length(evaluation_set)])
    mae_value <- mae(predictions_rescaled, evaluation_set[(config$inputs + 1):length(evaluation_set)])
    mape_value <- mape(predictions_rescaled, evaluation_set[(config$inputs + 1):length(evaluation_set)]) * 100
    smape_value <- smape(predictions_rescaled, evaluation_set[(config$inputs + 1):length(evaluation_set)]) * 100
    
    # Append results
    results <- rbind(results, data.frame(
      Inputs=config$inputs,
      HiddenLayers=toString(config$layers),
      Neurons=toString(unlist(config$layers)),
      ActivationFunction=config$activation,
      LinearOutput=config$linear,
      RMSE=rmse_value,
      MAE=mae_value,
      MAPE=mape_value,
      sMAPE=smape_value
    ))
  }
}

# Print results
print(results)

# Find the best-performing model based on RMSE
best_model <- results[which.min(results$RMSE), ]

# Print the best model
print("Best Model:")
print(best_model)

# Define denormalization function
denormalize <- function(data, center, scale) {
  return(data * scale + center)
}

# Denormalize the predicted values
predictions_denormalized <- denormalize(predictions_rescaled, attr(normalized_eval, "scaled:center")[1], attr(normalized_eval, "scaled:scale")[1])

# Plot the predictions against the actual data
plot_data <- data.frame(
  Time = 1:nrow(normalized_eval),
  Actual = denormalize(evaluation_set[(config$inputs + 1):length(evaluation_set)], attr(normalized_eval, "scaled:center")[1], attr(normalized_eval, "scaled:scale")[1]),
  Predicted = predictions_denormalized
)

# Plot
ggplot(plot_data, aes(Time)) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "Actual vs Predicted Exchange Rates",
       x = "Time",
       y = "Exchange Rate",
       color = "Line") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal()
