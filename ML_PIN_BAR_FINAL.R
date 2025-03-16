# Load necessary libraries
library(dplyr)
library(lubridate)
library(class)  # For KNN
library(randomForest)  # For Random Forest
library(caret)  # For confusion matrix and other metrics

# Step 1: Data Preparation
hsi_data <- read.csv("C:/Data Science/trading/hist_hsi_data.csv", stringsAsFactors = FALSE)
hsi_data$Date <- as.Date(hsi_data$Date, format="%m/%d/%Y")

# Filter data for the specified range
hsi_data <- hsi_data %>%
  filter(Date >= as.Date("2000-11-01") & Date <= as.Date("2022-02-28")) %>%
  filter(!is.na(Open) & !is.na(High) & !is.na(Low) & !is.na(Close))

# Separate training and test data
train_data <- hsi_data %>%
  filter(Date >= as.Date("2001-01-01") & Date <= as.Date("2015-12-31"))

test_data <- hsi_data %>%
  filter(Date >= as.Date("2016-01-01") & Date <= as.Date("2021-12-31"))

# Step 2: Feature Extraction for Training Data
train_data <- train_data %>%
  mutate(
    body_size = abs(Close - Open),
    upper_wick_size = High - pmax(Open, Close),
    lower_wick_size = pmin(Open, Close) - Low,
    PinBar = (lower_wick_size > body_size),
    UpsideDownPinBar = (upper_wick_size > body_size)
  )

# Step 3: Backtesting Logic for Buy and Sell Signals
trades <- data.frame(Date = character(), EntryPrice = numeric(),
                     ExitPrice = numeric(), Success = logical(),
                     TradeType = character(), stringsAsFactors = FALSE)

# Backtesting logic for buy signals from pin bars
for (i in 1:nrow(train_data)) {
  if (train_data$PinBar[i]) {
    entry_price <- train_data$Close[i]
    exit_price <- NA
    trade_success <- FALSE
    
    for (j in (i + 1):(i + 20)) {
      if (j > nrow(train_data)) break
      
      if (train_data$Low[j] < (entry_price - 100)) {
        break
      }
      
      if (train_data$High[j] >= (entry_price + 100)) {
        exit_price <- train_data$Close[j]
        trade_success <- TRUE
        break
      }
    }
    
    trades <- rbind(trades, data.frame(Date = train_data$Date[i], EntryPrice = entry_price, 
                                       ExitPrice = exit_price, Success = trade_success, 
                                       TradeType = "Buy"))
  }
}

# Backtesting logic for sell signals from upside-down pin bars
for (i in 1:nrow(train_data)) {
  if (train_data$UpsideDownPinBar[i]) {
    entry_price <- train_data$Close[i]
    exit_price <- NA
    trade_success <- FALSE
    
    for (j in (i + 1):(i + 20)) {
      if (j > nrow(train_data)) break
      
      if (train_data$High[j] > (entry_price + 100)) {
        break
      }
      
      if (train_data$Low[j] <= (entry_price - 100)) {
        exit_price <- train_data$Close[j]
        trade_success <- TRUE
        break
      }
    }
    
    trades <- rbind(trades, data.frame(Date = train_data$Date[i], EntryPrice = entry_price, 
                                       ExitPrice = exit_price, Success = trade_success, 
                                       TradeType = "Sell"))
  }
}

# Step 4: Prepare Data for Machine Learning
ml_data <- train_data %>%
  filter(!is.na(Close)) %>%
  select(Close, body_size, upper_wick_size, lower_wick_size)

# Add success outcomes based on trades
ml_data$Success <- ifelse(1:nrow(ml_data) %in% match(trades$Date, train_data$Date) & trades$Success[match(train_data$Date, trades$Date)], 1, 0)

# Ensure Success is a factor for classification
ml_data$Success <- as.factor(ml_data$Success)

# Step 5: Time Series Cross-Validation for KNN
# Using a single split instead of multiple folds for faster execution
train_index <- 1:floor(0.8 * nrow(ml_data))
test_index <- (floor(0.8 * nrow(ml_data)) + 1):nrow(ml_data)

# Train the kNN model
knn_model <- train(Success ~ body_size + upper_wick_size + lower_wick_size,
                   data = ml_data[train_index, ],
                   method = "knn",
                   tuneLength = 3)  # Further reduce tuning length

# Save the kNN model
saveRDS(knn_model, "knn_model.rds")

# Step 6: Time Series Cross-Validation for Random Forest
# Train the Random Forest model
rf_model <- train(Success ~ body_size + upper_wick_size + lower_wick_size,
                  data = ml_data[train_index, ],
                  method = "rf",
                  tuneLength = 3)  # Further reduce tuning length

# Save the Random Forest model
saveRDS(rf_model, "rf_model.rds")

# Step 7: Feature Extraction for Test Data
test_data <- test_data %>%
  mutate(
    body_size = abs(Close - Open),
    upper_wick_size = High - pmax(Open, Close),
    lower_wick_size = pmin(Open, Close) - Low,
    PinBar = (lower_wick_size > body_size),
    UpsideDownPinBar = (upper_wick_size > body_size)
  )

# Add success outcomes to test data based on backtesting logic
test_data$Success <- ifelse((test_data$PinBar & lead(test_data$Close, default = 0) > test_data$Close) | 
                              (test_data$UpsideDownPinBar & lead(test_data$Close, default = 0) < test_data$Close), 1, 0)

# Ensure Success is a factor for classification
test_data$Success <- as.factor(test_data$Success)

# Prepare the test data for prediction
test_ml_data <- test_data %>%
  select(body_size, upper_wick_size, lower_wick_size)

# Step 8: Load the Trained Models for Prediction
knn_model <- readRDS("knn_model.rds")
rf_model <- readRDS("rf_model.rds")

# Step 9: Make Predictions on the Test Data Using KNN
knn_predictions <- predict(knn_model, test_ml_data)

# Ensure levels of knn_predictions match test_data$Success
knn_predictions <- factor(knn_predictions, levels = levels(test_data$Success))

# Step 10: Evaluate KNN Model
test_data$Predictions <- knn_predictions
confusion_matrix_knn <- confusionMatrix(test_data$Predictions, test_data$Success)
print("KNN Confusion Matrix:")
print(confusion_matrix_knn)

# Step 11: Make Predictions on the Test Data Using Random Forest
rf_predictions <- predict(rf_model, test_ml_data)

# Ensure levels of rf_predictions match test_data$Success
rf_predictions <- factor(rf_predictions, levels = levels(test_data$Success))

# Step 12: Evaluate Random Forest Model
test_data$Predictions_RF <- rf_predictions
confusion_matrix_rf <- confusionMatrix(test_data$Predictions_RF, test_data$Success)
print("Random Forest Confusion Matrix:")
print(confusion_matrix_rf)

# Extract metrics for both models
metrics <- function(conf_matrix) {
  accuracy <- conf_matrix$overall['Accuracy']
  sensitivity <- conf_matrix$byClass['Sensitivity']
  specificity <- conf_matrix$byClass['Specificity']
  prevalence <- sum(test_data$Success == "1") / nrow(test_data)
  
  cat("Overall Accuracy (%):", accuracy * 100, "\n")
  cat("Sensitivity (%):", sensitivity * 100, "\n")
  cat("Specificity (%):", specificity * 100, "\n")
  cat("Prevalence (%):", prevalence * 100, "\n")
}

cat("\nKNN Model Metrics:\n")
metrics(confusion_matrix_knn)

cat("\nRandom Forest Model Metrics:\n")
metrics(confusion_matrix_rf)

# Step 13: Create an Ensemble Prediction
# Combine predictions using majority voting
ensemble_predictions <- ifelse(knn_predictions == "1" | rf_predictions == "1", "1", "0")

# Convert to factor
ensemble_predictions <- factor(ensemble_predictions, levels = levels(test_data$Success))

# Evaluate Ensemble Model
test_data$Predictions_Ensemble <- ensemble_predictions
confusion_matrix_ensemble <- confusionMatrix(test_data$Predictions_Ensemble, test_data$Success)
print("Ensemble Confusion Matrix:")
print(confusion_matrix_ensemble)

# Extract metrics for the Ensemble Model
cat("\nEnsemble Model Metrics:\n")
metrics(confusion_matrix_ensemble)