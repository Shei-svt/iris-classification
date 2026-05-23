# 🌸 Iris Species Classification Dashboard

**Universidad de la Costa – Data Mining Final Project**  
**Professor:** José Escorcia-Gutierrez, Ph.D.

## Author
- Sheila Daniela Hernandez Carrillo - 18038


## Project Overview

This project presents an end-to-end Data Mining workflow to classify Iris flowers (Iris setosa, Iris versicolor, Iris virginica) using a Random Forest Classifier, deployed through an interactive Streamlit dashboard. The objective is to predict the species of a flower based on four numerical features and to visually communicate the results through an intuitive and interactive interface.

### Dataset
The project uses the classic Iris Dataset, which contains:

- 150 samples
- 3 species
- 4 features

Features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

## Methodology

1. Data Understanding
   Descriptive statistics
   Feature distributions per species
   Correlation analysis

2. Data Preprocessing
   Clean dataset (no missing values)
   Feature scaling using StandardScaler
   Stratified train-test split (75% / 25%)

3. Modeling
   Algorithm: Random Forest Classifier
   n_estimators = 200
   class_weight = 'balanced'
   random_state = 42

4. Evaluation
   Accuracy, Precision, Recall, F1-score
   Confusion Matrix
   Classification Report per class
   Stratified 5-Fold Cross-Validation

5. Dashboard Development
   Interactive prediction panel
   3D visualization of predictions
   Statistical visualizations (histograms, boxplots)
   Scatter matrix and PCA projection
   Feature importance and correlation heatmap


### Why Random Forest?
The Random Forest Classifier was selected because:

- It reduces overfitting through ensemble learning
- It handles multi-class classification efficiently
- It captures non-linear relationships in the data
- It provides feature importance for interpretability
- It performs strongly on small, structured datasets like Iris


## Deployment

Deployed on **Streamlit Community Cloud**:  
https://iris-classification-acsappvej9udlgcdgvb5bkg.streamlit.app/

## Video Presentation

[Add your video link here]


*Universidad de la Costa · Data Mining · 2025*
