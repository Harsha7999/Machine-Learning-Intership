# TASK 1 : Customer Churn Prediction

This project aims to predict customer churn for a telecom company using customer usage data, service complaints, and other features. Accurately predicting churn helps the company to take proactive measures to retain customers, thereby reducing revenue loss.

## Dataset Information

The dataset includes customer demographics, account information, service usage, and complaints. Key features include:

- `customerID`: Unique identifier for each customer
- `tenure`: Number of months the customer has been with the company
- `MonthlyCharges`: The amount charged to the customer monthly
- `TotalCharges`: The total amount charged to the customer
- `Churn`: Whether the customer has churned (Yes/No)
- Other features: Gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod

**Preprocessing Steps:**
- Handled missing values by imputing or removing incomplete entries
- Encoded categorical variables using techniques like One-Hot Encoding
- Normalized numerical features using Min-Max scaling

## Methodology

### Data Preprocessing
1. Handled missing values through imputation and removal.
2. Encoded categorical variables using One-Hot Encoding.
3. Normalized numerical features to ensure all features contribute equally to the model.

### Feature Engineering
1. Created new features such as average call duration, complaint frequency.
2. Used feature selection techniques like Recursive Feature Elimination (RFE) to identify the most important features.

### Handling Imbalanced Data
1. Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.

### Model Selection
1. Trained various models including Logistic Regression, Random Forest, and XGBoost.
2. Used Grid Search with cross-validation to tune hyperparameters and select the best model.

## Model Training and Evaluation

### Models Trained
- Logistic Regression
- Random Forest
- XGBoost

### Hyperparameter Tuning
- Performed Grid Search with cross-validation to find optimal parameters.

### Evaluation Metrics
- Optimized for Accuracy and F1-score.
- Evaluated models using Confusion Matrix, ROC-AUC, Precision, Recall, and F1-score.

![task1](https://github.com/Harsha7999/Machine-Learning-Intership/assets/138028961/cfaaf184-db20-4268-a344-f25f7cb56f7d)

## Results (example)

The best-performing model was the XGBoost classifier with the following metrics:
- Accuracy: 0.85
- F1-score: 0.82
- ROC-AUC: 0.88

Visualizations:
- Confusion Matrix
- ROC Curve
- Feature Importance Plot

## Conclusion and Future Work

The model successfully predicts customer churn with high accuracy and F1-score. Future improvements could include:
- Incorporating additional features such as social media activity or customer feedback.
- Exploring deep learning models for potentially higher accuracy.
- Implementing real-time churn prediction in the telecom's CRM system.

# TASK 2 : Rainfall Prediction

This project aims to predict whether it will rain tomorrow based on today's weather data in major Australian cities. Accurate rainfall prediction can help mitigate human and financial losses.

## Dataset Information

The dataset includes daily weather observations from multiple Australian weather stations. Key features include:

- `Date`: The date of observation
- `Location`: The location of the weather station
- `MinTemp`: Minimum temperature of the day
- `MaxTemp`: Maximum temperature of the day
- `Rainfall`: Amount of rainfall recorded
- `WindSpeed9am`, `WindSpeed3pm`: Wind speed recorded at 9 AM and 3 PM
- `Humidity9am`, `Humidity3pm`: Humidity recorded at 9 AM and 3 PM
- `Pressure9am`, `Pressure3pm`: Atmospheric pressure at 9 AM and 3 PM
- `RainToday`: Whether it rained today (Yes/No)
- `RainTomorrow`: Target variable indicating if it will rain tomorrow (Yes/No)

![task2](https://github.com/Harsha7999/Machine-Learning-Intership/assets/138028961/0a471ef4-f316-4d01-bbb6-0b388215338e)

**Preprocessing Steps:**
- Handled missing values by imputing or removing incomplete entries.
- Encoded categorical variables using techniques like Label Encoding.
- Normalized numerical features using Min-Max scaling.

## Methodology

### Data Preprocessing
1. Handled missing values by imputing or removing incomplete entries.
2. Encoded categorical variables using Label Encoding.
3. Normalized numerical features using Min-Max scaling.

### Feature Engineering
1. Created new features such as humidity difference and temperature change.
2. Selected features using correlation analysis and feature importance scores.

### Model Selection
1. Trained models such as Logistic Regression, Decision Trees, Random Forest, and SVM.
2. Used Grid Search with cross-validation to tune hyperparameters.

## Model Training and Evaluation

### Models Trained
- Logistic Regression
- Decision Trees
- Random Forest
- SVM

### Hyperparameter Tuning
- Performed Grid Search with cross-validation to find optimal parameters.

### Evaluation Metrics
- Optimized for Accuracy and F1-score.
- Evaluated models using Confusion Matrix, ROC-AUC, Precision, Recall, and F1-score.

## Results (example)

The best-performing model was the Random Forest classifier with the following metrics:
- Accuracy: 0.83
- F1-score: 0.78
- ROC-AUC: 0.85

Visualizations:
- Confusion Matrix
- ROC Curve
- Feature Importance Plot

## Conclusion and Future Work

The model successfully predicts rainfall with good accuracy and F1-score. Future improvements could include:
- Incorporating additional weather features such as cloud cover or solar radiation.
- Exploring advanced models like Gradient Boosting Machines or neural networks.
- Implementing a real-time weather prediction system.

# TASK 3 : Disaster or Not?

This project aims to classify tweets to determine whether they indicate a disaster. Accurate classification helps disaster relief agencies and news outlets track emergencies in real time.

## Dataset Information

The dataset includes tweets labeled as indicating a disaster or not. Key features include:

- `id`: Unique identifier for each tweet
- `text`: The text of the tweet
- `target`: Target variable indicating if the tweet is about a disaster (1) or not (0)

![task3](https://github.com/Harsha7999/Machine-Learning-Intership/assets/138028961/7c46232b-edec-4599-a157-b533e486b9cc)

**Preprocessing Steps:**
- Cleaned text data by removing stopwords, punctuation, and URLs.
- Tokenized and vectorized text data using TF-IDF or word embeddings.

## Methodology

### Data Preprocessing
1. Cleaned text data by removing stopwords, punctuation, and URLs.
2. Tokenized and vectorized text data using TF-IDF or word embeddings.

### Feature Engineering
1. Extracted additional features such as tweet length and presence of certain keywords.
2. Used techniques like LDA for topic modeling.

### Model Selection
1. Trained models such as Naive Bayes, Logistic Regression, LSTM, and BERT.
2. Used Grid Search with cross-validation to tune hyperparameters.

## Model Training and Evaluation

### Models Trained
- Naive Bayes
- Logistic Regression
- LSTM
- BERT

### Hyperparameter Tuning
- Performed Grid Search with cross-validation to find optimal parameters.

### Evaluation Metrics
- Optimized for Accuracy and F1-score.
- Evaluated models using Confusion Matrix, ROC-AUC, Precision, Recall, and F1-score.

## Results (example)

The best-performing model was the BERT classifier with the following metrics:
- Accuracy: 0.90
- F1-score: 0.87
- ROC-AUC: 0.92

Visualizations:
- Confusion Matrix
- ROC Curve
- Word Cloud of disaster-related tweets

## Conclusion and Future Work

The model successfully classifies disaster-related tweets with high accuracy and F1-score. Future improvements could include:
- Incorporating additional sources of data such as images or videos from tweets.
- Exploring ensemble methods to combine predictions from multiple models.
- Implementing a real-time tweet classification system for disaster monitoring.

# TASK 4 : Predictive Maintenance: Jet Engine Sensor Readings

## Project Title and Description
**Predictive Maintenance Using Jet Engine Sensor Data**

This project aims to predict the Remaining Useful Life (RUL) of jet engines based on sensor data collected from multiple engines during their operational cycles. Predicting the RUL helps in performing timely maintenance and preventing unexpected engine failures, which is crucial for ensuring safety and operational efficiency in the aerospace industry.

## Dataset Information
**Datasets Used:**
1. **Training Data (`t4train.txt`):** Contains sensor readings from multiple engines over time. The last cycle for each engine indicates the failure point.
2. **Test Data (`t4test.txt`):** Contains sensor readings from multiple engines without the failure point.
3. **Truth Data (`t4truth.txt`):** Contains the actual RUL for each engine in the test data.

**Features:**
- **Operational Settings:** `setting1`, `setting2`, `setting3`
- **Sensor Measurements:** `s1` to `s21`
- **Other Columns:** `id` (engine identifier), `cycle` (operational cycle number)

**Preprocessing Steps:**
1. Dropped columns with NaN values.
2. Normalized sensor data using MinMaxScaler.
3. Calculated RUL by subtracting the current cycle from the maximum cycle for each engine in the training data.

## Methodology
**Data Preprocessing:**
1. **Normalization:** Applied MinMax scaling to the sensor readings.
2. **RUL Calculation:** For the training data, calculated the RUL by subtracting the current cycle from the maximum cycle for each engine.
3. **Feature Engineering:** Added a binary feature indicating if the engine will fail within a specified window (`failure_within_w1`).

**Exploratory Data Analysis:**
1. Scatter plots of sensor readings against cycles.
2. Time series plots for specific engines.
3. Visualized normalized sensor data for better understanding and patterns recognition.

**Modeling:**
1. **Initial Model:** Simple Neural Network for predicting cycles.
2. **Advanced Model:** LSTM (Long Short-Term Memory) neural network for capturing temporal dependencies in the sensor data.

**Handling Imbalanced Data:**
- Focused on RUL prediction which inherently deals with imbalanced data by treating each time step individually.

**Model Selection:**
1. **Neural Networks:** Dense layers with dropout for regularization.
2. **Recurrent Neural Networks:** LSTM layers to capture temporal patterns.

## Model Training and Evaluation
**Training Process:**
1. **Dense Neural Network:**
   - Layers: Dense with ReLU activation and Dropout for regularization.
   - Loss Function: Mean Squared Error (MSE).
   - Optimization: Adam optimizer.
   - Early Stopping: Used to prevent overfitting.

2. **LSTM Network:**
   - Layers: Stacked LSTM with Dropout for regularization.
   - Loss Function: Mean Squared Error (MSE).
   - Optimization: Adam optimizer.
   - Early Stopping: Used to prevent overfitting.

**Evaluation Metrics:**
- **Root Mean Squared Error (RMSE):** Primary metric for model evaluation.
- **Mean Absolute Error (MAE):** Secondary metric for evaluation.

**Model Performance:**
- Tracked training and validation loss over epochs.
- Visualized actual vs. predicted RUL for the test set.

![task4](https://github.com/Harsha7999/Machine-Learning-Intership/assets/138028961/89116a60-eaa0-4b51-9f85-22181fe4b7de)

## Results
**Final Model Performance:**
- **Dense Network:** Provided initial baseline performance.
- **LSTM Network:** Improved performance by capturing temporal dependencies in the data.

**Key Metrics:**
- **RMSE:** [Insert RMSE value]
- **MAE:** [Insert MAE value]

**Visualizations:**
- Plots of actual vs. predicted RUL.
- Training and validation accuracy plots.

## Conclusion and Future Work
**Findings:**
- The LSTM model effectively captures the temporal dependencies in the sensor data, leading to improved RUL predictions.
- Feature normalization and proper preprocessing are crucial for enhancing model performance.

**Future Directions:**
1. **Data Augmentation:** Generate synthetic data to further improve model robustness.
2. **Feature Engineering:** Explore additional features or transformations to improve model accuracy.
3. **Model Optimization:** Experiment with different neural network architectures and hyperparameters.
4. **Deployment:** Develop a real-time prediction system for predictive maintenance applications.

This README provides a comprehensive overview of the project, detailing each step from data preprocessing to model training and evaluation.
