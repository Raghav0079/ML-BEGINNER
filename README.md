# Machine Learning for Beginners

## Algorithms Covered

### Linear Regression
- Supervised learning algorithm for continuous value prediction
- Equation: y = mx + b
- Implementation includes:
  - Feature selection
  - Model training using least squares method
  - R-squared and MSE evaluation

### Classification
#### Logistic Regression
- Binary classification using sigmoid function
- Metrics: accuracy, precision, recall, F1-score

#### Decision Trees
- Tree-based supervised learning
- Uses information gain and Gini impurity
- Includes tree pruning and depth control

### Clustering
#### K-means
- Unsupervised learning for data partitioning
- Process:
  1. Initialize K centroids
  2. Assign points
  3. Update centroids
  4. Repeat until convergence

#### Hierarchical Clustering
- Creates cluster dendrograms
- Agglomerative and Divisive approaches
- Multiple distance metrics support

## Getting Started

```python
# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
```

## Datasets
- homeprices.csv
- mallCustomerData.txt
- salaries.csv
- User_Data.csv
- 50_Startups.csv

## License
Educational purpose only
