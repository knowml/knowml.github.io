# Supervised Learning

## Regression
### Linear regression
### Decision trees for regression

## Classification
### SVM 
Support Vector Machine wants the smallest distance between data points and the decision boundary to be as large as possible. In other words, if you imagine the decision boundary as the central line of a street, SVM prefers an 8-line highway rather than a country road. The width of the street is called the margin.

#### Algorithm
1. Prediction:
    - 1 if parameters * values are > 1
    - 0 if parameters * values are < -1 
2. min cost function 
    - min(C * sum (y * cost + (1-y) * cost) for observations + 1/2 * sum of squared parameters)
    - C - regularization parameter - 1/lambda
3. why it choses the wider margin?
    - it't the regularisation part - sum of squared parameters
    - we have parameters * values -> for it to be > 1 or < -1:
        - we say that parameters * values is a projection of parameters matrix transposed on x vector - dot product
        - we can write it as projection * ||parameters||, so if projections (margins) are small, then parameters need to be high and cost funtion will be high

#### Hyperparameters
for sklearn.svm.LinearSVC, SVC
intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
- penalty 
    - l1 - lasso - absolute values of parameters, which can bring some parameters to 0 
    - l2 - ridge - squared values of parameters, all parameters are shrinked by the same value
- loss
    - hinge - normal cost
    - squared-hinge - squared cost 
-dual
    - true - use dual, with kernel svm
    - false - use primal, when n of samples > n of features
- C
    - Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
- multi_class - one vs rest
- fit_intercept - True/False
- intercept_scaling
    - True: instance vector x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equals to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic feature weight, Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased
- class weight
    - assign weights to classes
    - with dict or auto, based on frequencies
- kernel:
    - linear
    - polynomial - add polynomial features to have polynomial boundry
    - Gaussian RBF (Radial Basis Function)
        - Adds more features to the data by computing similarity features. A similarity feature measures how far a value of an existing feature is from a landmark. 
        Let’s make it practical: we have a data set with a single feature, x1. We want to create two similarity features, so we select two landmarks, i.e. reference values from our single feature. 

#### PROS
- effective in high dimensional spaces, or in cases where number of dimensions is greater than the number of samples
- uses a subset of training points in the decision function (called support vectors), so it is also memory efficient
- Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

#### CONS
- If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial
- SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation 

### Logistic regression

### Tress based:
#### Decision tree
Decision Trees is a type of model which breaks down the given input data through decision, based on the features we train the model and their values. 
Training part is respnsible for finding questions (features and split values) that best separate training data into target classes. 

##### Algorithm
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

##### Algorithm
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

#### Gradient Boosting Decision Trees
(tbc)
Grow trees that are not complex. All the trees are connected in series and each tree tries to minimize the error of the previous tree. Due to this sequential connection, boosting algorithms are usually slow to learn (controllable by the developer using the learning rate parameter), but also highly accurate.

##### Algorithm
1. Calculate log(odds) for target = 1 as a initial prediction for all  - Logistic Regression equivalent of average
    - odds - number of 1 / number of 0
2. Convert it to probability by applying to logistic function - e ^ log(odds) / 1 + e^ log(odds) 
3. Calculate Pseudo-Residuals
    - (1 - probability) for observation with true target 1 
    - (0 - probability) for observation with true target 0
4. Build a small, simple tree using features and as a target value use Pseudo-Residuals
5. In leaves we have observations and they Pseudo-Residuals
6. Calucate a weight of a leaf - its log(odds)
    - we refer to probabilities up to this point by using formula -> sum (residuals in leaf) / sum (previous probabilities x (1 - previous probabilities)) for each residual
7. To calcualte new values Pseudo-Residuals we need to go all the way from first leaf with first log(odd) up to newest leaf
8. For each observation:
    - calculate: first log(odd) X (learning rate x first tree leaf log(odds)) 
    - convert to probability - logistic function
    - calculate pseudo - residual 


#### Hyperparameters
- number of leaves (usually between 8 and 32) 
- learning reate - common 0.1


#### XGBoost
Extreme Gradient Boosting 
Designed to enhance the performance and speed for small-to-medium structured/tabular data.
XGBoost uses pre-sorted algorithm & histogram-based algorithm for computing the best split. 
The histogram-based algorithm splits all the data points for a feature into discrete bins and uses these bins to find the split value of the histogram. 
Also, in XGBoost, the trees can have a varying number of terminal nodes and left weights of the trees that are calculated with less evidence is shrunk more heavily.


#### LighGBM

#### AdaBoost
(tbc)
AdaBoosting uses decision stumps (decision trees with only one single split). 
It also attached weights to observations, adding more weight to ‘difficult-to-classify’ observations and less weight to those that are easy to classify.
The aim is to put stress on the difficult to classify instances for every new weak learner. So, for the next subsequent model, the misclassified observations will receive more weight, as a result, in the new dataset these observations are sampled more number of times according to their new weights, giving the model a chance to learn more of such records and classify them correctly.

### Neural Network based:
#### Percpetron
#### CNN
#### RNN
### Ensamble learning
Ensemble learning, in general, is a model that makes predictions based on a number of different models. By combining a number of different models, an ensemble learning tends to be more flexible (less bias) and less data sensitive (less variance). The two most popular ensemble learning methods are bagging and boosting.
#### Bagging 
Boosted aggregation - training a bunch of models in parallel way. 
Each model learns from a random subset of the data, where the dataset is same size as original but is randomly sampled with replacement (bootstrapped).
Example: Random Forest

#### Boosting
Training a bunch of models sequentially. 
Each model learns from the mistakes of the previous model. 
That is, the subsequent models tries to explain and predict the error left over by the previous model. 
In boosting, weak learners (ex: decision trees with only the stump) are used which perform only slightly better than a random chance. Boosting focuses on sequentially adding up these weak learners and filtering out the observations that a learner gets correct at every step.
Basically, the stress is on developing new weak learners to handle the remaining difficult observations at each step. 
Example: AdaBoost, XGBost