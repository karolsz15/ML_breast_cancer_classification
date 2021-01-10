# MACHINE LEARNING CASE STUDY: BREAST CANCER CLASSIFICATION

## 1. PROBLEM STATEMENT

- Predicting if the cancer diagnosis is benign or malignant based on several observations/features 
- 30 features are used, examples:
        - radius (mean of distances from center to points on the perimeter)
        - texture (standard deviation of gray-scale values)
        - perimeter
        - area
        - smoothness (local variation in radius lengths)
        - compactness (perimeter^2 / area - 1.0)
        - concavity (severity of concave portions of the contour)
        - concave points (number of concave portions of the contour)
        - symmetry 
        - fractal dimension ("coastline approximation" - 1)

- Datasets are linearly separable using all 30 input features
- Number of Instances: 569
- Class Distribution: 212 Malignant, 357 Benign
- Target class:
         - Malignant
         - Benign
- Data source: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

## 2. IMPORTING DATA

### 2.1. IMPORTING LIBRARIES
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
```

### 2.2. IMPORTING THE DATA FROM THE SKLEARN LIBRARY (could have been also imported from the link provided in 'problem statement'))
`
from sklearn.datasets import load_breast_cancer\
cancer = load_breast_cancer()\
`

## 3. VISUALIZING THE DATA

### 3.1. Data visualisations for different data pairs - 'mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'.
![GitHub Logo](/images/datavisualisation1.png)

### 3.2. Count visualisation for class distribution (0 - Malignant, 1 - Benign)
![GitHub Logo](/images/datavisualisation2.png)

### 3.3. Scatterplot for x = 'mean area', y = 'mean smoothness', (hue = 'target').
![GitHub Logo](/images/datavisualisation3.png)

### 3.4. Checking the correlation between the variables
![GitHub Logo](/images/datavisualisation4.png)

## 4. MODEL TRAINING AND FINDING A PROBLEM SOLUTION
`
...
from sklearn.svm import SVC\ 
from sklearn.metrics import classification_report, confusion_matrix\
\
svc_model = SVC()\
svc_model.fit(X_train, y_train)\
`
`
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,\
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',\
  max_iter=-1, probability=False, random_state=None, shrinking=True,\
  tol=0.001, verbose=False)\
`

## 5. EVALUATING THE MODEL

`
y_predict = svc_model.predict(X_test)\
cm = confusion_matrix(y_test, y_predict)\
sns.heatmap(cm, annot=True)\
`
![GitHub Logo](/images/confusionmatrix1.png)

`print(classification_report(y_test, y_predict))`
`precision    recall  f1-score   support`
`
        0.0       0.00      0.00      0.00        48
        1.0       0.58      1.00      0.73        66
`
`avg / total       0.34      0.58      0.42       114`

## 6. IMPROVING THE MODEL

### 6.1. Feature scaling
![GitHub Logo](/images/featurescaling.png)

### 6.2. Training the model
`
...\
from sklearn.svm import SVC\
from sklearn.metrics import classification_report, confusion_matrix\
\
svc_model = SVC()\
svc_model.fit(X_train_scaled, y_train)\
`
`
y_predict = svc_model.predict(X_test_scaled)\
cm = confusion_matrix(y_test, y_predict)\
\
sns.heatmap(cm,annot=True,fmt="d")\
`
![GitHub Logo](/images/confusionmatrix2.png)

`print(classification_report(y_test,y_predict))`
`
print(classification_report(y_test,y_predict))\
print(classification_report(y_test,y_predict))\
`
`precision    recall  f1-score   support`
`
        0.0       1.00      0.90      0.95        48
        1.0       0.93      1.00      0.96        66
`
`avg / total       0.96      0.96      0.96       114`

## 7. IMPROVING THE MODEL - PART 2

`param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} `
`from sklearn.model_selection import GridSearchCV`
`grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)`
`grid.fit(X_train_scaled,y_train)`
`grid.best_params_`
`*{'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}*`
`grid.best_estimator_`
`grid_predictions = grid.predict(X_test_scaled)`
`cm = confusion_matrix(y_test, grid_predictions)`

`sns.heatmap(cm, annot=True)`
![GitHub Logo](/images/confusionmatrix3.png)

`print(classification_report(y_test,grid_predictions))`
`precision    recall  f1-score   support`
`
        0.0       1.00      0.94      0.97        48
        1.0       0.96      1.00      0.98        66
`
`avg / total       0.97      0.97      0.97       114`
