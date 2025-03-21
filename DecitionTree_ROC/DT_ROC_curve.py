# -*- coding: utf-8 -*-
"""This used to be a file
"""

#Load sklearn kits: source sklearn-env/bin/activate
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn import tree
from sklearn.metrics import auc, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


#Input week to cutoff for diabetes
week_cutoff = 28

#feature_sub will be the features use includeParentOnset to add the parent onset as inputs
#Do not include father and mother onset.
feature_sub = ['wt4', 'wt8', 'rbg4', 'rbg8'] 
includeParentOnset = False #Features to run Decision Tree on, All possible features are below:
#sex, gestational_diet, nursing_diet, wt4, wt8, wt12, rbg4, rbg8, rbg12, rbg200_fa, rbg200_mo'
#For copy-pasting, ['sex', 'gestational_diet', 'nursing_diet', 'wt4', 'wt8', 'wt12', 'rbg4', 'rbg8', 'rbg12'] 

#Read from file
R_measure = pd.read_csv('measurement.csv', header = 0, index_col = 0)
R_onset_all = pd.read_csv('onset.csv', header = 0, index_col = 0)
R_onset = deepcopy(R_onset_all[(R_onset_all.overall_diet == "Rod")])

#Remove unnecessary rows (measurements pas 12 weeks)
measure_sample = R_measure[(R_measure.week < 16) & (R_measure.week % 4 == 0)][['NR_Name', 'week', 'weight', 'weight percentile', 'rbg']]

#Pivot the dataframe and remove na values for 4 and 8 week weight.
wt = measure_sample.pivot(index = 'NR_Name', columns = 'week', values = 'weight')
wt_filter = wt[(wt[4] > 0) & (wt[8] > 0)]

#Similar thing to rbg values, but also remove any rats above 200 (cutoff for diabetes)
rbg = measure_sample.pivot(index = 'NR_Name', columns = 'week', values = 'rbg')
rbg_filter = rbg[(rbg[4] > 0) & (rbg[4] < 200) & (rbg[8] > 0) & (rbg[8] < 200)]

#Merge all dataframes and rename relevant columns.
recording_all = pd.merge(wt, rbg, on = 'NR_Name').rename(
    columns={'4_x':'wt4', '8_x':'wt8', '12_x':'wt12', '4_y':'rbg4', '8_y':'rbg8', '12_y':'rbg12'}
    )

#'sex', 'generation', 'gestational_diet', 'nursing_diet', 'weanling_diet',
#Make non-numerical values into a boolean, 1 are the ones that tends to get diabetic
R_onset[['sex']] = R_onset[['sex']] == 'M'
R_onset[['nursing_diet']] = R_onset[['nursing_diet']] == 'Rod'
R_onset[['gestational_diet']] = R_onset[['gestational_diet']] == 'Rod'

#Merges with mother and father onset time
allData = pd.merge(recording_all[['wt4', 'wt8', 'wt12', 'rbg4', 'rbg8', 'rbg12']],
                   R_onset[['NR_Name', 'mother', 'father', 'sex', 'gestational_diet', 'nursing_diet', 'rbg200']], 
                   on = "NR_Name")

allData_mother = pd.merge(allData, R_onset_all[['NR_Name', 'rbg200']], how = 'left', left_on='mother', right_on='NR_Name').rename(
    columns = {'rbg200_x':'rbg200', 'rbg200_y': 'rbg200_mo', 'NR_Name_x':'NR_Name'}).drop(columns=['NR_Name_y'])

allData_parent = pd.merge(allData_mother, R_onset_all[['NR_Name', 'rbg200']], how = 'left', left_on='father', right_on='NR_Name').rename(
    columns = {'rbg200_x':'rbg200', 'rbg200_y': 'rbg200_fa', 'NR_Name_x':'NR_Name'}).drop(columns=['NR_Name_y'])

#column_to_move will be the features, removed wt12 and rbg
column_to_keep = 'rbg200'
if (includeParentOnset):
    features = feature_sub + ['rbg200_fa', 'rbg200_mo']
else :
    features = feature_sub
noNA = allData_parent[features + [column_to_keep]].dropna(subset = feature_sub)

#Create x and y train and test set to use for 5-fold CV
x = noNA[features] 
x = x.fillna(80) #This is to fill parent onset weeks, it's set to 80 but maybe 60 is better
y = noNA[column_to_keep]
y = y.fillna(80) #This will just become a boolean so it just needs to be bigger than week_cutoff
y = y <= week_cutoff
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=888)

#5-fold CV for Deicision Tree, using Decision Tree because 5-fold CV tunes the hyperparameters.
sklearn_5fold_DT = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid={
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'max_depth': [3, 4, 5, 6]
    },
    cv=5,
    scoring='accuracy'
)
sklearn_5fold_DT.fit(x_train, y_train)
print(sklearn_5fold_DT.best_params_)

#Use the parameters from 5-fold CV for the sample
sklearn_DT = DecisionTreeClassifier(splitter = sklearn_5fold_DT.best_params_['splitter'], 
                                criterion = sklearn_5fold_DT.best_params_['criterion'],
                                max_depth = sklearn_5fold_DT.best_params_['max_depth']) #Insert best parameters
sklearn_DT.fit(x_train, y_train)

#Print out which features are important, as well as showing what the decision tree looks like.
print(sklearn_DT.feature_importances_)
y_pred = sklearn_DT.predict(x_test)
text_representation = tree.export_text(sklearn_DT)
print(text_representation)
print(classification_report(y_test, y_pred))

#Plots the ROC curve of the model.
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
print(f'model 1 AUC score: {roc_auc_score(y_test, y_pred)}')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show() #Plot shown will need to be closed before the next can be opened

#Plots the ROC curve for each feature
plt.figure(figsize=(10, 8))
for i in range(x.shape[1]):
    # Get the probabilities for class 1
    y_probs = sklearn_DT.predict_proba(x_test)[:, 1]
    
    # Calculate the ROC curve and AUC for each feature
    fpr, tpr, thresholds = roc_curve(y_test, x_test.iloc[:, i])
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'{features[i]} (AUC = {roc_auc:.2f})')

# Plot random chance (diagonal line)
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')

# Customize the plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Feature')
plt.legend(loc='lower right')
plt.show()
