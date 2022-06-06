# Machine Learning Algorithms

## Supervised learning
### Regression
#### Linear regression
#### Decision trees for regression

## Classification
### SVM
### Logistic regression

### Tress based:
#### Decision tree
Decision Trees is a type of model which breaks down the given input data through decision, based on the features we train the model and their values. 
Training part is respnsible for finding questions (features and split values) that best separate training data into target classes. 

#### Algorithm
1. This algorithm will begin at the root node (the topmost node) and split the data on the feature which results in the largest information gain or the lower Gini Impurity.
2. We repeat the splitting process iteratively at each child node until it is impossible to split further (or the stop criteria are met).

As you can imagine this process is prone to overfitting as it can result in a very deep tree with many nodes (one leaf per observation).
The typical approach to remedy this is by pruning the tree to prevent overfitting. The pruning process is quite simple in most scenarios, the easiest way would be to set a limit on the maximum depth of the tree.

##### Hyperparameters:
for sklearn.tree.DecisionTreeClassifier
- criterion 
    - Gini impurity = 1 - sum of (squared probability of samples belonging to class i at a given node - p^2) -> 0 for pure node, ~0.5 for super impure
    - entropy/log_loss - Shannon information gain = sum of (- probability of samples belonging to class i at a given node x log2 of that probability) -> ~0 for invormative split, 1 for no information gain  
    - they are very similar, gini faster 
    - https://quantdare.com/decision-trees-gini-vs-entropy/ 
- splitter
    -best or random
    - Strategies are 'best' to choose the best split and 'random' to choose the best random split
- max tree depth
    - prunning parameter, nodes are expanded until max depth or without limit
- min number of samples required to split an internal node
    - constant value or a fraction of trianing set
- min number of samples required to be at a leaf node
    - split point at any depth will only be considered if it leaves at least this training samples in each of the left and right branches
    - may have the effect of smoothing the model, especially in regression
    - constant value or a fraction of trianing set
- number of features to consider when looking for the best split
    - constant, fraction of features, square root of # of features, log of # of feautres, all features
- max numer of leaf nodes
    - constant or no limit
- min impurity decrease
    - A node will be split if this split induces a decrease of the impurity greater than or equal to this value
- class weight
    - assign weights to classes
    - with dict or auto, based on frequencies
- ccp_alpha
    - Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed 
    - https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning


 ##### PROS
- easy to interpret
- clear decision path
- no need for normalisation, scaling, deleting outliers
- fit to nonlinear problems
- we can see which features are most important for decision making process

##### CONS
- overfitting
- not do well on unbalanced data
- high variance, small change in data can result in a large change in the structure of the tree and decisions being made
- calculation time increases drastically with the number of classes and size of the data set


#### Random forest
Random forest is an ensemble of decision trees. Ensemble learning is a method which uses multiple learning algorithms to boost predictive performance. This method will allow a better generalization of the model and is less prone to overfitting.

#### Algorithm
1. Draw a random bootstrap sample of size n (randomly choose n samples from the training set with replacement - training size also n).
2. Grow a decision tree from the bootstrap sample. At each node:
3. Randomly select d features without replacement.
4. Split the node using the feature that provides the best split according to the objective function (gini impurity, information gain)
5. Repeat the steps 1 to 2 k times to grow k tress in the forest.
6. Prediction are made based on voting from all trees in the forest - Bagging or Bootstrap Aggregation 

##### Hyperparameters:
for sklearn.ensemble.RandomForestClassifier
- number of trees in the forest
- bootstrap
    - True: bootstrap samples are used when building trees, False: the whole dataset is used to build each tree
- oob_score
    - True: use out-of-bag samples to estimate the generalization score
    - Only available if bootstrap=True
    - when we have small dataset and don't want to have separate validation set
    - https://towardsdatascience.com/what-is-out-of-bag-oob-score-in-random-forest-a7fa23d710
- max number of samples to draw for bootstrap
    - same size as training set, contant, fraction of training set
- criterion, max depth, min samples split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease, class_weight, ccp_alpha
    - like in decision trees

 ##### PROS
- unlikely to overfit data
- no need for normalisation, scaling, deleting outliers
- fit to nonlinear problems
- we can see which features are most important for decision making process

##### CONS
- not do well on unbalanced data
- calculation time increases drastically with the number of classes and size of the data set
- harder to interpret


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


