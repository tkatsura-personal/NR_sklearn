# -*- coding: utf-8 -*-
"""This used to be a file
"""

#Load sklearn kits: source sklearn-env/bin/activate
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
R_measure = pd.read_csv('measurement.csv', header = 0, index_col = 0)
R_onset = pd.read_csv('onset.csv', header = 0, index_col = 0)

measure_sample = R_measure[(R_measure.week < 16) & (R_measure.week % 4 == 0)][['NR_Name', 'week', 'weight', 'weight percentile', 'rbg']]
wt = measure_sample.pivot(index = 'NR_Name', columns = 'week', values = 'weight')
wt_filter = wt[(wt[4] > 0) & (wt[8] > 0) & (wt[12] > 0)]

wtpt = measure_sample.pivot(index = 'NR_Name', columns = 'week', values = 'weight percentile')
wtpt_filter = wtpt[(wtpt[4] > 0) & (wtpt[8] > 0) & (wtpt[12] > 0)]

rbg = measure_sample.pivot(index = 'NR_Name', columns = 'week', values = 'rbg')
rbg_filter = rbg[(rbg[4] > 0) & (rbg[8] > 0) & (rbg[12] > 0)]

recordings = pd.merge(wt, rbg, on = 'NR_Name').rename(columns={'4_x':'wt4', '8_x':'wt8', '12_x':'wt12',
                                                               '4_y':'rbg4', '8_y':'rbg8', '12_y':'rbg12'})
recording_all = pd.merge(recordings, wtpt, on = 'NR_Name').rename(columns={4:'wtpt4', 8:'wtpt8', 12:'wtpt12'})

#'sex', 'generation', 'gestational_diet', 'nursing_diet', 'weanling_diet',
R_onset[['sex']] = R_onset[['sex']] == 'M'
R_onset[['nursing_diet']] = R_onset[['nursing_diet']] == 'Rab'
R_onset[['gestational_diet']] = R_onset[['gestational_diet']] == 'Rab'

allData = pd.merge(recording_all[['wt4', 'wt8', 'wt12', 'wtpt4', 'wtpt8', 'wtpt12', 'rbg4', 'rbg8', 'rbg12']],
                   R_onset[['NR_Name', 'sex', 'gestational_diet', 'nursing_diet', 'rbg200']], on = "NR_Name")
possible_ML_options = [
    ['sex', 'wt4', 'wt8', 'rbg4', 'rbg8', 'rbg200'],
    ['sex', 'wtpt4', 'wtpt8', 'rbg4', 'rbg8', 'rbg200'],
    ['sex', 'wt4', 'wt8', 'wt12', 'rbg4', 'rbg8', 'rbg12', 'rbg200'], 
    ['sex', 'wtpt4', 'wtpt8', 'wtpt12', 'rbg4', 'rbg8', 'rbg12', 'rbg200'],
    ['sex', 'gestational_diet', 'wt4', 'wt8', 'rbg4', 'rbg8', 'rbg200'],
    ['sex', 'gestational_diet', 'wtpt4', 'wtpt8', 'rbg4', 'rbg8', 'rbg200'],
    ['sex', 'gestational_diet', 'wt4', 'wt8', 'wt12', 'rbg4', 'rbg8', 'rbg12', 'rbg200'], 
    ['sex', 'gestational_diet', 'wtpt4', 'wtpt8', 'wtpt12', 'rbg4', 'rbg8', 'rbg12', 'rbg200'],
    ['sex', 'nursing_diet', 'wt4', 'wt8', 'rbg4', 'rbg8', 'rbg200'],
    ['sex', 'nursing_diet', 'wtpt4', 'wtpt8', 'rbg4', 'rbg8', 'rbg200'],
    ['sex', 'nursing_diet', 'wt4', 'wt8', 'wt12', 'rbg4', 'rbg8', 'rbg12', 'rbg200'], 
    ['sex', 'nursing_diet', 'wtpt4', 'wtpt8', 'wtpt12', 'rbg4', 'rbg8', 'rbg12', 'rbg200'],
    ['sex', 'nursing_diet', 'gestational_diet', 'wt4', 'wt8', 'rbg4', 'rbg8', 'rbg200'],
    ['sex', 'nursing_diet', 'gestational_diet', 'wtpt4', 'wtpt8', 'rbg4', 'rbg8', 'rbg200'],
    ['sex', 'nursing_diet', 'gestational_diet', 'wt4', 'wt8', 'wt12', 'rbg4', 'rbg8', 'rbg12', 'rbg200'], 
    ['sex', 'nursing_diet', 'gestational_diet', 'wtpt4', 'wtpt8', 'wtpt12', 'rbg4', 'rbg8', 'rbg12', 'rbg200']]

for option in possible_ML_options :
    column_to_keep = 'rbg200'
    column_to_check = [col for col in allData[option] if col != column_to_keep]
    print(column_to_check)
    noNA = allData[option].dropna(subset = column_to_check)
    x = noNA[column_to_check]
    y = noNA['rbg200']
    y = y.fillna(80)
    y_bool = (y <= 40)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=110)
    x_train_bool, x_test_bool, y_train_bool, y_test_bool = train_test_split(x, y_bool, test_size=0.2, random_state=888)

    sklearn_DT = DecisionTreeClassifier()
    sklearn_DT.fit(x_train_bool, y_train_bool)
    y_pred = sklearn_DT.predict(x_test_bool)
    #print(classification_report(y_test_bool, y_pred))

    sklearn_5fold_DT = GridSearchCV(
        DecisionTreeClassifier(),
        param_grid={
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7, 9],
            'splitter': ['best', 'random']
        },
        cv=5,
        scoring='accuracy'
    )
    sklearn_5fold_DT.fit(x_train_bool, y_train_bool)
    print(sklearn_5fold_DT.best_params_)

    sklearn_DT = DecisionTreeClassifier(criterion = sklearn_5fold_DT.best_params_['criterion'], 
                                    max_depth = sklearn_5fold_DT.best_params_['max_depth'],
                                    splitter = sklearn_5fold_DT.best_params_['splitter']) #Insert best parameters
    sklearn_DT.fit(x_train_bool, y_train_bool)
    y_pred = sklearn_DT.predict(x_test_bool)
    print(classification_report(y_test_bool, y_pred))
    print()
