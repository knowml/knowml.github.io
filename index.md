# Machine Learning and the rest

Description: A cheatsheet, describe alogithms and methods used in Machine Learning
Written By: Ewelina Kaminska
Contact Information: 
Last Updated: 29.05.2022
Machine learning Pipeline and it's R&D nature


# Define problem
Define needs, possibilities, requirements, acceptance criteria (min, mid, max), benefits of project, risks if it won't work, priorities, possible data, inputs and outputs, what to monitor, what is crutial to monitor in the future, how to implment model within application functions
product owner, business analyst, data scientist to evaluate what is possible to model

# Gather data
Gather data and their description from different data sources and put it to the analytical ML environement.
Define data scope and history
data engineer, business analyst

# Analyse data

## Analysis
### Duplication in data
### Distribution
#### Histograms
#### Scatter plots
### Correlation
### Missing values
### Outliers
#### Isolation forest

## Preprocessing
### Encoding categorical data
### Feature engineering
### Missing data
### Normalisation
### Scaling
### Dimentionality reduction
#### PCA
#### t-sne

# Modelling
## Tools
### Experimantation:
- MLFlow
- Neptune.ai

### Model registry
### Environement:
- spark, jupyter,

## Supervised learning
### Regression
#### Linear regression
#### Decision trees for regression

## Classification
### SVM
### Logistic regression
### Tress based:
#### Decision tree
#### Random forest
#### XGBoost
#### Ada Boost
### Neural Network based:
#### Percpetron
#### CNN
#### RNN

## Unsupervised learning
### Clustering
#### K-means
#### DBSCAN
#### Hierarchical clustering

### Anomaly detection
### Recommendation engines

## Optimisation
### Gradient descent
### Metaheuristics
#### Genetic algorithm
#### PSO
#### Memetic algorithm
#### CMA-ES
### Hyperparameters optimisation
- Optuna
- Hyperopt

## Other algorithms
### Text mining
### Image mining


# Preparing for production
- data pipeline
- retraining pipeline
- monitoring metrics and alters

# Production
## Batch 
## Real-time
