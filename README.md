# Customer Churn Prediction Pipeline
![Customer Churn](https://cdn.prod.website-files.com/666fdec617b81d256ca8f1e5/666fdec617b81d256ca8f7ad_6-Ways-CRM-Stop-Customer-Churn.png)


This project is an end-to-end pipeline for predicting customer churn using machine learning. It includes loading, cleaning, and balancing data, training multiple models, hyperparameter tuning, and deploying the model for inference through a Streamlit app. Additionally, the model generates personalized emails to retain customers based on their churn probability.

## Project Overview
Customer churn prediction helps identify customers likely to leave a service, enabling targeted retention efforts. This project specifically considers that retaining a customer is more cost-effective than acquiring a new one, and the prediction model focuses on accuracy in identifying potential churners. The pipeline is built with a focus on high recall for reliable identification.

## Features
1. Data Preparation: Loading and cleaning the dataset, with handling of skewed data using SMOTE (Synthetic Minority Oversampling Technique) to achieve balanced classes.
2. Model Training: Training five different machine learning models:<br>
        Random Forest<br>
        XGBoost<br>
        K-Nearest Neighbors (KNN)<br>
        Support Vector Machines (SVM)<br>
        Decision Tree<br>
4. Ensemble Voting: A majority voting mechanism is used to combine the results from three models and determine an average churn probability.
5. Key Factor Analysis: Trend analysis for customer churn and identification of influential factors to improve interpretability.
6. Evaluation Metrics: Recall is used as the primary evaluation metric, given that capturing potential churn is critical to this business application.
7. Visualization: Churn probabilities and contributing factors are presented graphically in the Streamlit web app for user-friendly insights.
8. Customer Retention Strategy: Llama-3.1 7B model generates custom emails offering incentives to high-risk customers to improve retention rates.

## Project Pipeline
1. Data Loading: Import the customer dataset and perform initial exploration.
2. Data Cleaning and Preprocessing: Handle missing values, outliers, and normalize/standardize features as needed.
3. Data Balancing: Apply SMOTE to balance the classes due to the high imbalance in churn and non-churn classes.
4. Feature Selection: Identify key features that most influence customer churn for focused insights and interpretability.
5. Model Training: Train and optimize five different machine learning models.
6. Hyperparameter Tuning: Tune model parameters using techniques like grid search to enhance model performance.
7. Model Evaluation: Evaluate the models based on recall to prioritize capturing potential churners accurately.
8. Ensemble Voting: Calculate an average churn probability using an ensemble voting approach from three selected models.
9. Deployment: Serve the model through a Streamlit app, allowing for real-time churn probability predictions and visualizations.
10. Customer Retention Email: Generate a personalized email for customers at high churn risk, offering tailored incentives to encourage retention.
