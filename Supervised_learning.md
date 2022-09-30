# Supervised Learning

## Regression
### Evaluation
- MAE: The mean absolute error (MAE) is the average absolute difference between the target values and the predicted values. This metric ranges from zero to infinity; a lower value indicates a higher quality model.
- RMSE: The root-mean-squared error is the square root of the average squared difference between the target and predicted values. RMSE is more sensitive to outliers than MAE,so if you're concerned about large errors, then RMSE can be a more useful metric to evaluate. Similar to MAE, a smaller value indicates a higher quality model (0 represents a perfect predictor).
- RMSLE: The root-mean-squared logarithmic error metric is similar to RMSE, except that it uses the natural logarithm of the predicted and actual values plus 1. RMSLE penalizes under-prediction more heavily than over-prediction. It can also be a good metric when you don't want to penalize differences for large prediction values more heavily than for small prediction values. This metric ranges from zero to infinity; a lower value indicates a higher quality model. The RMSLE evaluation metric is returned only if all label and predicted values are non-negative.
- r^2: r squared (r^2) is the square of the Pearson correlation coefficient between the labels and predicted values. This metric ranges between zero and one; a higher value indicates a higher quality model.
- MAPE: Mean absolute percentage error (MAPE) is the average absolute percentage difference between the labels and the predicted values. This metric ranges between zero and infinity; a lower value indicates a higher quality model.
- MAPE is not shown if the target column contains any 0 values. In this case, MAPE is undefined.
- Model feature attributions: Vertex AI shows you how much each feature impacts a model. The values are provided as a percentage for each feature: the higher the percentage, the more strongly that feature impacted model training. Review this information to ensure that all of the most important features make sense for your data and business problem.

### Linear regression
### Decision trees for regression

## Classification
### Evaluation
- Log loss: The cross-entropy between the model predictions and the target values. This ranges from zero to infinity, where a lower value indicates a higher-quality model.
- Confidence threshold: A confidence score that determines which predictions to return. A model returns predictions that are at this value or higher. A higher confidence threshold increases precision but lowers recall. Vertex AI returns confidence metrics at different threshold values to show how the threshold affects precision and recall.
- Recall: The fraction of predictions with this class that the model correctly predicted. Also called true positive rate.
- Recall at 1: The recall (true positive rate) when only considering the label that has the highest prediction score and not below the confidence threshold for each example.
- Precision: The fraction of classification predictions produced by the model that were correct.
- Precision at 1: The precision when only considering the label that has the highest prediction score and not below the confidence threshold for each example.
- F1 score: The harmonic mean of precision and recall. F1 is a useful metric if you're looking for a balance between precision and recall and there's an uneven class distribution.
- F1 score at 1: The harmonic mean of recall at 1 and precision at 1.
- True negative count: The number of times a model correctly predicted a negative class.
- True positive count: The number of times a model correctly predicted a positive class.
- False negative count: The number of times a model mistakenly predicted a negative class.
- False positive count: The number of times a model mistakenly predicted a positive class.
- False positive rate: The fraction of incorrectly predicted results out of all predicted results.
- False positive rate at 1: The false positive rate when only considering the label that has the highest prediction score and not below the confidence threshold for each example.
- Confusion matrix: A confusion matrix shows how often a model correctly predicted a result. For incorrectly predicted results, the matrix shows what the model predicted instead. The confusion matrix helps you understand where your model is "confusing" two results.
- Model feature attributions: Vertex AI shows you how much each feature impacts a model. The values are provided as a percentage for each feature: the higher the percentage, the more strongly that feature impacted model training. Review this information to ensure that all of the most important features make sense for your data and business problem.

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
        - we can write it as projection * lengh of parameters vector, so if projections (margins) are small, then parameters need to be high and cost funtion will be high

#### Hyperparameters
for sklearn.svm.LinearSVC, SVC
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

### K Nearest Neighbours
KNN can be used for both regression and classification. 
KNN tries to predict the correct class for the test data by calculating the distance between the test data and all the training points. Then select the K number of points which is closet to the test data. The KNN algorithm calculates the probability of the test data belonging to the classes of ‘K’ training data and class holds the highest probability will be selected. In the case of regression, the value is the mean of the ‘K’ selected training points.
#### Algorithm
1. Select the number K of the neighbors
2. Calculate the Euclidean distance of K number of neighbors
3. Take the K nearest neighbors as per the calculated Euclidean distance.
4. Among these k neighbors, count the number of the data points in each category.
5. Assign the new data points to that category for which the number of the neighbor is maximum.
6. Our model is ready.
#### Hyperparameters
- numer of neighbours
- distance metric
    - Euclidean Distance: Euclidean distance is calculated as the square root of the sum of the squared differences between a new point (x) and an existing point (y).
    - Manhattan Distance: This is the distance between real vectors using the sum of their absolute difference.
    - Hamming Distance: It is used for categorical variables. If the value (x) and the value (y) are the same, the distance D will be equal to 0 . Otherwise D=1.
- algorithm
    - auto - decide
    - Ball Tree - clusters of clusters, new points distance is calculated with centroids
    - KDTree - rearange dataset to tree to speed calculations
    - brute force -standard, calculate distance of point to all point in neibourhood
- leaf size 
    - Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.
    - deafult 30
- p 
    - Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
- weights
    - uniform : uniform weights. All points in each neighborhood are weighted equally
    - distance : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away

#### PROS
- K-NN is pretty intuitive and simple 
- K-NN has no assumptions: K-NN is a non-parametric algorithm which means there are assumptions to be met
 -No Training Step: K-NN does not explicitly build any model, it simply tags the new data entry based learning from historical data. New data entry would be tagged with majority class in the nearest neighbor.
 - It constantly evolves: Given it’s an instance-based learning; k-NN is a memory-based approach. The classifier immediately adapts as we collect new training data. It allows the algorithm to respond quickly to changes in the input during real-time use.
 - Very easy to implement for multi-class problem: Most of the classifier algorithms are easy to implement for binary problems and needs effort to implement for multi class whereas K-NN adjust to multi class without any extra efforts.
 - Can be used both for Classification and Regression: One of the biggest advantages of K-NN is that K-NN can be used both for classification and regression problems.

#### CONS
- K-NN slow algorithm
- Curse of Dimensionality: KNN works well with small number of input variables but as the numbers of variables grow K-NN algorithm struggles to predict the output of new data point.
- K-NN needs homogeneous features: If you decide to build k-NN using a common distance, like Euclidean or Manhattan distances, it is completely necessary that features have the same scale, since absolute differences in features weight the same
- Optimal number of neighbors: One of the biggest issues with K-NN is to choose the optimal number of neighbors to be consider while classifying the new data entry.
- Imbalanced data causes problems: k-NN doesn’t perform well on imbalanced data. If we consider two classes, A and B, and the majority of the training data is labeled as A, then the model will ultimately give a lot of preference to A. This might result in getting the less common class B wrongly classified.
- Outlier sensitivity: K-NN algorithm is very sensitive to outliers as it simply chose the neighbors based on distance criteria.
- Missing Value treatment: K-NN inherently has no capability of dealing with missing value problem.

### Tree based
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
    - entropy/log_loss - Shannon information gain = sum of (- probability of samples belonging to class i at a given node x log2 of that probability) -> ~0 for informative split, 1 for no information gain  
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

 #### PROS
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
3. Compare prediction with true value - calculate Pseudo-Residuals
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

Explained in [statquest](https://www.youtube.com/watch?v=jxuNLH5dXCs)

#### Hyperparameters
- number of leaves (usually between 8 and 32) 
- learning reate - common 0.1


#### XGBoost
Extreme Gradient Boosting 
Designed to enhance the performance and speed for small-to-medium structured/tabular data.
XGBoost uses pre-sorted algorithm & histogram-based algorithm for computing the best split. 
The histogram-based algorithm splits all the data points for a feature into discrete bins and uses these bins to find the split value of the histogram. 
Also, in XGBoost, the trees can have a varying number of terminal nodes and left weights of the trees that are calculated with less evidence is shrunk more heavily.

XGBoost advantages:
- Regularization
- Parallel Processing
- High Flexibility
- Handling Missing Values
- Tree Pruning
- Built-in Cross-Validation
- Continue on Existing Model

##### Algorithm

Gradient boosting is an approach where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction. It is called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models.

###### General Parameters
These define the overall functionality of XGBoost.

- booster [default=gbtree]
Select the type of model to run at each iteration. It has 2 options:
gbtree: tree-based models
gblinear: linear models
- silent [default=0]:
Silent mode is activated is set to 1, i.e. no running messages will be printed.
It’s generally good to keep it 0 as the messages might help in understanding the model.
- nthread [default to maximum number of threads available if not set]
This is used for parallel processing and number of cores in the system should be entered
If you wish to run on all cores, value should not be entered and algorithm will detect automatically
There are 2 more parameters which are set automatically by XGBoost and you need not worry about them. Lets move on to Booster parameters.

###### Booster Parameters
Though there are 2 types of boosters, I’ll consider only tree booster here because it always outperforms the linear booster and thus the later is rarely used.

- eta [default=0.3]
Analogous to learning rate in GBM
Makes the model more robust by shrinking the weights on each step
Typical final values to be used: 0.01-0.2
- min_child_weight [default=1]
Defines the minimum sum of weights of all observations required in a child.
This is similar to min_child_leaf in GBM but not exactly. This refers to min “sum of weights” of observations while GBM has min “number of observations”.
Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
Too high values can lead to under-fitting hence, it should be tuned using CV.
- max_depth [default=6]
The maximum depth of a tree, same as GBM.
Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
Should be tuned using CV.
Typical values: 3-10
- max_leaf_nodes
The maximum number of terminal nodes or leaves in a tree.
Can be defined in place of max_depth. Since binary trees are created, a depth of ‘n’ would produce a maximum of 2^n leaves.
If this is defined, GBM will ignore max_depth.
- gamma [default=0]
A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.
Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.
- max_delta_step [default=0]
In maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative.
Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced.
This is generally not used but you can explore further if you wish.
- subsample [default=1]
Same as the subsample of GBM. Denotes the fraction of observations to be randomly samples for each tree.
Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.
Typical values: 0.5-1
- colsample_bytree [default=1]
Similar to max_features in GBM. Denotes the fraction of columns to be randomly samples for each tree.
Typical values: 0.5-1
- colsample_bylevel [default=1]
Denotes the subsample ratio of columns for each split, in each level.
I don’t use this often because subsample and colsample_bytree will do the job for you. but you can explore further if you feel so.
- lambda [default=1]
L2 regularization term on weights (analogous to Ridge regression)
This used to handle the regularization part of XGBoost. Though many data scientists don’t use it often, it should be explored to reduce overfitting.
- alpha [default=0]
L1 regularization term on weight (analogous to Lasso regression)
Can be used in case of very high dimensionality so that the algorithm runs faster when implemented
- scale_pos_weight [default=1]
A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence.
 

###### Learning Task Parameters
These parameters are used to define the optimization objective the metric to be calculated at each step.

- objective [default=reg:linear]
This defines the loss function to be minimized. Mostly used values are:
binary:logistic –logistic regression for binary classification, returns predicted probability (not class)
multi:softmax –multiclass classification using the softmax objective, returns predicted class (not probabilities)
you also need to set an additional num_class (number of classes) parameter defining the number of unique classes
multi:softprob –same as softmax, but returns predicted probability of each data point belonging to each class.
- eval_metric [ default according to objective ]
The metric to be used for validation data.
The default values are rmse for regression and error for classification.
Typical values are:
rmse – root mean square error
mae – mean absolute error
logloss – negative log-likelihood
error – Binary classification error rate (0.5 threshold)
merror – Multiclass classification error rate
mlogloss – Multiclass logloss
auc: Area under the curve
- seed [default=0]
The random number seed.
Can be used for generating reproducible results and also for parameter tuning.
If you’ve been using Scikit-Learn till now, these parameter names might not look familiar. A good news is that xgboost module in python has an sklearn wrapper called XGBClassifier. It uses sklearn style naming convention. The parameters names which will change are:



##### Feature importance
Most informative is Gain. 
The meaning of the importance data table is as follows:

- The Gain implies the relative contribution of the corresponding feature to the model calculated by taking each feature's contribution for each tree in the model. A higher value of this metric when compared to another feature implies it is more important for generating a prediction.
- The Cover metric means the relative number of observations related to this feature. For example, if you have 100 observations, 4 features and 3 trees, and suppose feature1 is used to decide the leaf node for 10, 5, and 2 observations in tree1, tree2 and tree3 respectively; then the metric will count cover for this feature as 10+5+2 = 17 observations. This will be calculated for all the 4 features and the cover will be 17 expressed as a percentage for all features' cover metrics.
- The Frequency (/'Frequence') is the percentage representing the relative number of times a particular feature occurs in the trees of the model. In the above example, if feature1 occurred in 2 splits, 1 split and 3 splits in each of tree1, tree2 and tree3; then the weightage for feature1 will be 2+1+3 = 6. The frequency for feature1 is calculated as its percentage weight over weights of all features.

Why is it important to understand your feature importance results?

Suppose that you have a binary feature, say gender, which is highly correlated with your target variable. Furthermore, you observed that the inclusion/ removal of this feature form your training set highly affects the final results. If you investigate the importance given to such feature by different metrics, you might see some contradictions:

Most likely, the variable gender has much smaller number of possible values (often only two: male/female) compared to other predictors in your data. So this binary feature can be used at most once in each tree, while, let say, age (with a higher number of possible values) might appear much more often on different levels of the trees. Therefore, such binary feature will get a very low importance based on the frequency/weight metric, but a very high importance based on both the gain, and coverage metrics!



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
