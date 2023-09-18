# FCALC
 A library to perform lazy classification with FCA

# OSDA 2023 Lazy FCA

_**!THE CODE IS NOT READY YET!\
!YOU WILL BE NOTIFIED THROUGH TELEGRAM WHEN IT'S DONE!**_

_**NB: More updates to come**_

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
Baseline decision function is:
$$y = \dfrac{\sum\limits_{x_{pos} \in X_{pos}} a_{x_{pos}} [b_{x_{pos}} \leq \alpha * |X_{neg}|]}{|X_{pos}|^2} > \dfrac{\sum\limits_{x_{neg} \in X_{neg}} a_{x_{neg}} [b_{x_{neg}} \leq \alpha * |X_{pos}|]}{|X_{neg}|^2}$$
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
