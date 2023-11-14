# FCALC
 A library to perform lazy classification with FCA

# OSDA 2023 Lazy FCA

This repository contains tools for a "**Lazy FCA**" big homework assignment for the course "Ordered Sets in Data Analysis" taught at HSE in the "Data Science" master's program in Fall 2023.

## Task description

### Problem statement
This homework focuses on the task of binary classification. 
That is, we are given the data $X = \\{x_1, x_2, x_3, ...\\}$
and the corresponding labels $Y$ where each label $y \in Y$ is either True of False.
The task is to make a "prediction" $\hat{y}$ of a label $y \in Y$ for each datum $x \in X$ as if $y$ is unknown.

To estimate the quality of predictions we split the data $X$ into two non-overlapping sets $X_{train}$ and $X_{test}$.
Then we make make a prediction $\hat{y}$ for each test datum $x \in X_{test}$
based on the information obtained from the training data $X_{train}$. 
Finally, we measure how well predictions $\hat{y}$ represent the true test labels $y$.  

The original data $X$ often comes in various forms: numbers, categories, texts, graphs, etc.
Throughout the course of this assignment, you are going to work with two types of data: scaled (binarized) data, which you will obtain from preprocessing your datasets, and non-binarized data.

### Baseline algorithm

Assume that we want to make a prediction for description $x \subseteq M$ given
the set of training examples $X_{train} \subseteq 2^M$ and the labels $y_x \in \\{False, True\\}$
corresponding to each $x \in X_{train}$.

First, we split all examples $X_{train}$ to positive $X_{pos}$ and negative $X_{neg}$ examples:
$$X_{pos} = \\{x \in X_{train} \mid y_x \textrm{ is True} \\}, \quad X_{neg} = X \setminus X_{pos}.$$

To classify the descriptions $x$ we follow the procedure:
1) For each positive example $x_{pos} \in X_{pos}$ we compute the intersection $x \cap x_{pos}$.
Then, we count the support in positive and negative classes for this intersection, that is the number of respectively positive and negative examples $x_{train} \in X_{train}$ containing intersection $x \cap x_{pos}$;

2) Dually, for each negative example $x_{neg} \in X_{neg}$ we compute the intersection $x \cap x_{neg}$.
Then, we count the support in negative and positive classes for this intersection, that is the number of respectively negative and positive examples $x_{train} \in X_{train}$ containing intersection $x \cap x_{neg}$;.

Finally, we plug the obtained values of support into decision function and classify $x$ based on them.  
There are three parameterized methods for your choice:
1) "standard" we consider number of all alpha-weak hypothes and classify based on it: $$y = \dfrac{\sum\limits_{x_{pos} \in X_{pos}} [b_{x_{pos}} \leq \alpha * |X_{neg}|]}{|X_{pos}|} > \dfrac{\sum\limits_{x_{neg} \in X_{neg}} [b_{x_{neg}} \leq \alpha * |X_{pos}|]}{|X_{neg}|}$$ 

2) "standard-support" we consider number of all alpha-weak hypothes with their support and classify based on it: $$y = \dfrac{\sum\limits_{x_{pos} \in X_{pos}} a_{x_{pos}} [b_{x_{pos}} \leq \alpha * |X_{neg}|]}{|X_{pos}|^2} > \dfrac{\sum\limits_{x_{neg} \in X_{neg}} a_{x_{neg}} [b_{x_{neg}} \leq \alpha * |X_{pos}|]}{|X_{neg}|^2}$$

3) "ratio-support" we consider the ratio of support in target class and in opposite one for hypotheses which support in target class is $\alpha$ times higher than in other: $$y = \underset{k}{argmax} \left(\dfrac{|X_{train} \backslash X_{c_k}|\cdot \sum_{x_i \in X_{c_k}} a_i \cdot [\frac{a_i}{|X_{c_k}|} \geq \alpha \frac{b_i}{|X_{train} \backslash X_{c_k}|}]}{|X_{c_k}|\cdot \sum_{x_i \in X_{c_k}} b_i\cdot[\frac{a_i}{|X_{c_k}|} \geq \alpha \frac{b_i}{|X_{train} \backslash X_{c_k}|}]}  \right)$$

where $a_{x_k}$ is support in class $k$, and $b_{x_k}$ is support in the opposite class, of the intersection $x\cap x_k$.
### To-do list
1. Choose at least 3 datasets, define the target attribute, binarize data if necessary.\
Useful resources:
* [Kaggle](https://www.kaggle.com/)
* [UCI repository](https://archive.ics.uci.edu/datasets)
2. Perform classification using standard ML tools:
* [Decision tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) 
* [Random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)
* [xGboost](https://xgboost.readthedocs.io/en/latest/)
* [Catboost](https://catboost.ai/)
* [k-NN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
* [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
* [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)
3. Tune lazy-FCA classification with binary attributes to find the best classification. Select most contributing intersections responsible for classification
4. Tune  lazy-FCA classification with pattern structures to find the best classification.Select most contributing intersections responsible for classification
5.  Submit a report with comparison of all models, both standard and developed by you.

## Example
Let's start with importing the libraries
```python
import fcalc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
```
### Binarized data
I will use tic-tac-toe dataset as an example for the binarized data. Firstly we need to read our data and binarize it:
```python
column_names = [
        'top-left-square', 'top-middle-square', 'top-right-square',
        'middle-left-square', 'middle-middle-square', 'middle-right-square',
        'bottom-left-square', 'bottom-middle-square', 'bottom-right-square',
        'Class'
    ]
df = pd.read_csv('data_sets/tic-tac-toe.data', names = column_names)
df['Class'] = [x == 'positive' for x in df['Class']]
X = pd.get_dummies(df[column_names[:-1]], prefix=column_names[:-1]).astype(bool)
y = df['Class']
```
Then we need to split our data into train and test:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
Then we can initialize our classifier, to do so, you need to provide training data and training labels, both in numpy format:
```python
bin_cls = fcalc.classifier.BinarizedBinaryClassifier(X_train.values, y_train.to_numpy())
```
You can also specify the method and parameter for that method. Now you can predict the classes for your test data and evaluate the models:
```python
bin_cls.predict(X_test.values)
print(accuracy_score(y_test, bin_cls.predictions))
print(f1_score(y_test, bin_cls.predictions))
```
>0.9965277777777778

>0.9974160206718347
