# -*- coding: utf-8 -*-
"""This used to be a file
"""

#Load sklearn kits: source sklearn-env/bin/activate
import numpy as np
import copy
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pandas as pd

def week_class(week):
    if week <= 24:
        return 0
    elif week <= 40:
        return 1
    else:
        return 2
    
R_measure = pd.read_csv('measurement.csv', header = 0, index_col = 0)
R_onset_all = pd.read_csv('onset.csv', header = 0, index_col = 0)
R_onset = copy.deepcopy(R_onset_all[(R_onset_all.overall_diet == "Rod")])

measure_sample = R_measure[(R_measure.week < 16) & (R_measure.week % 4 == 0)][['NR_Name', 'week', 'weight', 'weight percentile', 'rbg']]
wt = measure_sample.pivot(index = 'NR_Name', columns = 'week', values = 'weight')
wt_filter = wt[(wt[4] > 0) & (wt[8] > 0) & (wt[12] > 0)]

rbg = measure_sample.pivot(index = 'NR_Name', columns = 'week', values = 'rbg')
rbg_filter = rbg[(rbg[4] > 0) & (rbg[8] > 0) & (rbg[12] > 0)]

recording_all = pd.merge(wt, rbg, on = 'NR_Name').rename(columns={'4_x':'wt4', '8_x':'wt8', '12_x':'wt12',
                                                               '4_y':'rbg4', '8_y':'rbg8', '12_y':'rbg12'})

#'sex', 'generation', 'gestational_diet', 'nursing_diet', 'weanling_diet',
R_onset[['sex']] = R_onset[['sex']] == 'M'
R_onset[['nursing_diet']] = R_onset[['nursing_diet']] == 'Rab'
R_onset[['gestational_diet']] = R_onset[['gestational_diet']] == 'Rab'

allData = pd.merge(recording_all[['wt4', 'wt8', 'wt12', 'rbg4', 'rbg8', 'rbg12']],
                   R_onset[['NR_Name', 'mother', 'father', 'sex', 'gestational_diet', 'nursing_diet', 'rbg200']], on = "NR_Name")

allData_mother = pd.merge(allData, R_onset_all[['NR_Name', 'rbg200']], how = 'left', left_on='mother', right_on='NR_Name').rename(
    columns = {'rbg200_x':'rbg200', 'rbg200_y': 'rbg200_mo', 'NR_Name_x':'NR_Name'}).drop(columns=['NR_Name_y'])

allData_parent = pd.merge(allData_mother, R_onset_all[['NR_Name', 'rbg200']], how = 'left', left_on='father', right_on='NR_Name').rename(
    columns = {'rbg200_x':'rbg200', 'rbg200_y': 'rbg200_fa', 'NR_Name_x':'NR_Name'}).drop(columns=['NR_Name_y'])

column_to_keep = 'rbg200'
column_to_move = ['sex', 'nursing_diet', 'gestational_diet', 'wt4', 'wt8', 'rbg4', 'rbg8']
noNA = allData_parent[['sex', 'nursing_diet', 'gestational_diet', 'wt4', 'wt8', 'rbg4', 'rbg8', 'rbg200']].dropna(subset = column_to_move)
x = noNA[column_to_move]
x = x.fillna(80)
y = noNA['rbg200']
y = y.fillna(80)
y = y <= 20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=888)

sklearn_5fold_DT = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid={
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'max_depth': [3, 5, 7, 9]
    },
    cv=5,
    scoring='accuracy'
)
sklearn_5fold_DT.fit(x_train, y_train)
print(sklearn_5fold_DT.best_params_)

sklearn_DT = DecisionTreeClassifier(splitter = sklearn_5fold_DT.best_params_['splitter'], 
                                criterion = sklearn_5fold_DT.best_params_['criterion'],
                                max_depth = sklearn_5fold_DT.best_params_['max_depth']) #Insert best parameters
sklearn_DT.fit(x_train, y_train)
y_pred = sklearn_DT.predict(x_test)
#print(classification_report(y_test, y_pred))
#print()
#text_representation = tree.export_text(sklearn_DT)
#print(text_representation)

#fpr, tpr, thresholds = roc_curve(y_test, y_pred)
#print(f'model 1 AUC score: {roc_auc_score(y_test, y_pred)}')
#plt.plot(fpr, tpr)
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.show()

plt.figure(figsize=(10, 8))
for i in range(x.shape[1]):
    # Get the probabilities for class 1
    y_probs = sklearn_DT.predict_proba(x_test)[:, 1]
    
    # Calculate the ROC curve and AUC for each feature
    fpr, tpr, thresholds = roc_curve(y_test, x_test.iloc[:, i])
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'{column_to_move[i]} (AUC = {roc_auc:.2f})')

# Plot random chance (diagonal line)
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')

# Customize the plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Feature')
plt.legend(loc='lower right')
plt.show()
