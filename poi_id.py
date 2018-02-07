#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
import pandas as pd
import numpy as np
sys.path.append("../tools/")
from numpy import mean
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
from sklearn.feature_selection import SelectKBest

def select_k_best(data_dict, features_list, num_features):
    data = featureFormat(data_dict, features_list)
    target, features = targetFeatureSplit(data)

    clf = SelectKBest(k = num_features)
    clf = clf.fit(features, target)
    features_list.pop(0)
    scores = clf.scores_
    scores = scores.tolist()
    feature_weights = {}
    for i in range(len(features_list)):
        feature_weights[features_list[i]] = scores[i]
    #print feature_weights
    selected_features = sorted(feature_weights.items(), key = lambda k: k[1], 
                               reverse = True)[:num_features]
    new_features = []
    for i, j in selected_features:
        new_features.append(i)
    #print new_features
    return new_features

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus','deferral_payments','deferred_income',
                 'director_fees','exercised_stock_options','expenses',
                 'from_messages','from_poi_to_this_person',
                 'from_this_person_to_poi','loan_advances',
                 'long_term_incentive','other','restricted_stock',
                 'restricted_stock_deferred','shared_receipt_with_poi',
                 'to_messages','total_payments','total_stock_value']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

### Task 3: Create new feature(s)
dataframe = pd.DataFrame.from_dict(data_dict, orient='index')
dataframe = dataframe.replace('NaN',np.nan)
dataframe['messages_from_poi_ratio'] = (dataframe.from_poi_to_this_person/
         dataframe.to_messages)
dataframe['messages_to_poi_ratio'] = (dataframe.from_this_person_to_poi/
         dataframe.from_messages)
dataframe = dataframe.replace(np.nan,0)
dataframe = dataframe.drop('email_address', axis=1)

### Store to my_dataset for easy export below.
my_dataset = dataframe.to_dict('index')

features_list = select_k_best(data_dict, features_list, 12)
features_list.insert(0, 'poi')
features_list.insert(len(features_list), 'messages_from_poi_ratio')
features_list.insert(len(features_list), 'messages_to_poi_ratio')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
clf = tree.tree.DecisionTreeClassifier()
clf = AdaBoostClassifier(tree.tree.DecisionTreeClassifier())
clf = RandomForestClassifier()
clf = GradientBoostingClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
    
import itertools
### Parameters
criterion = ['entropy','gini']
depth = [1,2,3]
min_split = [2,3,4,5]
min_leaf = xrange(1,10,2)
n_estimators = [1,2,3,4,5]
learning_rate = [1,2,3]
"""
### Combination for DecisionTree
for combination in itertools.product(criterion, depth, min_split, min_leaf):
### Combination for AdaBoost and GradientBoosting
#for combination in itertools.product(criterion, depth, min_split, min_leaf, n_estimators, learning_rate):
### Combination for RandomForest    
#for combination in itertools.product(criterion, depth, min_split, min_leaf, n_estimators):
    
    clf = tree.tree.DecisionTreeClassifier(criterion=combination[0], max_depth=combination[1], min_samples_split=combination[2], min_samples_leaf=combination[3], random_state=42)
    #clf = AdaBoostClassifier(tree.tree.DecisionTreeClassifier(criterion=combination[0], max_depth=combination[1], min_samples_split=combination[2], min_samples_leaf=combination[3], random_state=42), n_estimators=combination[4], learning_rate=combination[5], random_state=42)
    #clf = GradientBoostingClassifier(max_depth=combination[1], min_samples_split=combination[2], min_samples_leaf=combination[3], n_estimators=combination[4], learning_rate=combination[5], random_state=42)
    #clf = RandomForestClassifier(criterion=combination[0], max_depth=combination[1], min_samples_split=combination[2], min_samples_leaf=combination[3], n_estimators=combination[4], random_state=42)
    print '*****************************'
    test_classifier(clf, my_dataset, features_list)
"""
clf = AdaBoostClassifier(tree.tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=2, max_depth=1, min_samples_leaf=5, random_state=42), n_estimators=2, learning_rate=2, random_state=42)
#   clf = tree.tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=5, max_depth=3, min_samples_leaf=7, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)