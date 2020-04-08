Project 5 - Identifying The Enron Email Fraud
====================
##### Author: Nikolas Thorun

In 2001 Enron Corporation went bankrupt. It was one of the world leaders in the supply of energy and services. For years, company directors made up balance sheets, wiped out losses and inflated profits. Indictments and rumors promoted what would be the biggest financial scandal in the United States.
The objective of this project is to analyze the characteristics of the company's employees in order to be able to predict which of them are People of Interest (POI), that is, employees who participated in the fraud. The use of machine learning is advantageous in this case, as it makes the data processing process much faster and more efficient than the human brain.

#### Data overview

The data set consists of 146 records with 20 features and 1 label (POI). Some features are financial, others relate to the use of e-mails. 18 records are labeled as POI's, people who were demonstrably involved in the fraud.
During the initial investigations, 2 of the 146 records did not represent people, so they were removed.


* `'TOTAL'`
* `'THE TRAVEL AGENCY IN THE PARK'`





#### Feature Selection

As a default, SelectKBest calculates the F-values of ANOVA (Analysis of Variance), that is, we can use it to select the characteristics of greatest variance.
Through this algorithm, the 12 initial characteristics with the highest variance were selected. The characteristics and F-values are shown in the table below.
        
|Features | Weight | 
|:-------|:------:|
|  exercised_stock_options | 24.81  |
|  total_stock_value | 24.18  |
|  bonus | 20.79  | 
|  salary | 18.28  |
|  deferred_income | 11.45 |
|  long_term_incentive | 9.92  |
|  restricted_stock | 9.21  |
|  total_payments | 8.77  |
|  shared_receipt_with_poi | 8.58  |
|  loan_advances | 7.18  |
|  expenses | 6.09  |
|  from_poi_to_this_person | 5.24  |

For each tested algorithm, a different number of characteristics produced the best results. The selection of characteristics for each algorithm took place through trial and error, considering the classifiers without adjustments. The number of characteristics used is shown in the next section.

No scaling process was used, since the tested algorithms do not use the Euclidean distance to calculate the distance between two points. In cases such as those of the SVM and kNN algorithms, for example, if the data set is not scaled, the variables with the highest values will have more influence on the classifier. Scaling aims to create non-dimensional variables, so that their greatness does not bias the result.

Two new variables were created and used together with the 12 original characteristics selected.
They are: `'messages_from_poi_ratio'` and `' messages_to_poi_ratio'`. The idea is to know the proportion in which a given employee receives and sends e-mails to some POI.


#### Selected Classifier
The Algorithm selected at the end of the tests was AdaBoost using Decision Trees as the basic estimator. In addition to it, Decision Tree, Random Forest and Gradient Boosting were tested. AdaBoost and Decision Trees had the highest values of the F1 score.
The best results for each algorithm are shown below rounded to the second decimal place.


* With the new features:

| | AdaBoost | DecisionTree | Gradient Boosting | Random Forest |
|:-------|:------:|:------:|:------:|:------:|
|  Accuracy | 0.85  | 0.88 | 0.77 | 0.86 |
|  Precision | 0.45  | 0.57 | 0.35 | 0.58 |
|  Recall | 0.62  | 0.44 | 0.60 | 0.23 |
|  F1 score | 0.52  | 0.50 | 0.44 | 0.33 |

* Without the new features:

| | AdaBoost | DecisionTree | Gradient Boosting | Random Forest |
|:-------|:------:|:------:|:------:|:------:|
|  Accuracy | 0.86  | 0.87 | 0.83 | 0.87 |
|  Precision | 0.48  | 0.54 | 0.46 | 0.65 |
|  Recall | 0.30  | 0.38 | 0.47 | 0.38 |
|  F1 score | 0.37  | 0.44 | 0.47 | 0.48 |


The number of characteristics used to obtain these results is shown in the table below:

| | AdaBoost | DecisionTree | Gradient Boosting | Random Forest |
|:-------|:------:|:------:|:------:|:------:|
|  Nº of features | 12  | 11 | 3 | 3 |


#### Settings

Fine adjustments in the algorithm perform the function of optimizing the classifier's performance. Classifiers, without defined parameters, tend to be generalists and in order to adjust them to a specific problem, adjustments are necessary. If this process is done poorly, the classifier is subject to a poor performance in the training set (_underfitting_) or to become too specific (_overfitting_).
At the beginning of the project, GridSearchCV was used to obtain the optimum parameters of the classifier. However, it returns the parameters that obtained the highest accuracy during cross-validation and, in this case, it would be interesting to know if there is a combination of parameters that has a slightly lower accuracy and a much higher recall rate, for example.
Therefore, the `product` function of the` itertools` module was used in order to generate all possible combinations between the parameters. For each combination, the `test_classifier` function of `tester.py` is called and prints the results on the console.
The parameters adjusted for each algorithm are shown below:

###### Adaboost
* `n_estimators` - from 1 to 5
* `learning_rate` - from 1 to 3 
* `criterion` - 'gini' and 'entropy'
* `max_depth` - from 1 to 3
* `min_samples_split` - from 2 to 5 
* `min_samples_leaf` - from 1 to 10, stepped 2 by 2 

The first two are parameters of AdaBoost and the rest belong to Decision Trees, which is its basic estimator.

###### Decision Trees
* `criterion` - 'gini' and 'entropy'
* `max_depth` - from 1 to 3
* `min_samples_split` - from 2 to 5 
* `min_samples_leaf` - from 1 to 10, stepped 2 by 2 

###### Gradient Boosting
* `n_estimators` - 1, 10 and 25
* `learning_rate` - 0.5, 1 and 3 

###### Gradient Boosting
* `n_estimators` - 10, 100 and 200

#### Validation

Validation is the process of testing the machine learning algorithm on data that was not used during the training stage. You can divide a data set in two, use one part to train the algorithm and another part to validate how good the results were predicted by the algorithm by comparing them with the real values of this subset. A classic way to make mistakes during this process is to use the entire data set for training and then test the algorithm on part of that set. The classifier is likely to be overfitted and will probably give poor results on new data.
The validation used in this project was that of the `test_classifier` function of `tester.py`. In this function, StratifiedShuffleSplit is used, which receives the data set and divides it into training and test sets a thousand times, in the proportion of 90% to 10%. The results obtained at the end are the averages of the results in each division. StratifiedShuffleSplit ensures that all divisions have the same proportion between classes, which is quite interesting in this project, given that the data set is small.

#### Evaluation metrics
For the final classifier, four evaluation metrics were used:

| | AdaBoost | 
|:-------|:------:|
|  Accuracy | 0.85  | 
|  Precision | 0.45  | 
|  Recall | 0.62  | 
|  F1 score | 0.52  | 

* Accuracy

It can be said that approximately 85% of the observations were correctly classified between POI and non-POI. However, knowing that only 18 of the 144 (12.5%) employees are POI's, if the model classified all employees as non-POI the classifier would be 87.5% accurate. That is, accuracy is an important metric, but it would not be wise to use it alone.
* Precision

It can be said that approximately 45% of the observations classified as POI's were actually POI's. That is, 55% of the observations classified as POI's are false positives. The greater the accuracy, the fewer employees will be falsely accused of being POI's.
* Recall

It can be said that approximately 62% of the observations that really are POI's were correctly identified. That is, 38% of POI's are false negatives. The greater the recall, the less likely a POI will escape its identification.
* F1 score

The F1 score is the combination of precision and recall so that with just one number, we know how well the classifier works. The higher the F1 score, the better.
That is why AdaBoost was chosen, because despite presenting less accuracy than Decision Tree, the F1 score was slightly higher. In addition, the recall of AdaBoost was far superior, reducing the possibility of a POI getting away with it.
In this project, seeking higher recall values at the expense of lower precision values is understandable, since mistakenly identifying a POI is not as bad as failing to identify a POI.

