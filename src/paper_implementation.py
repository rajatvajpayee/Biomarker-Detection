## Below lines of code is the implementation of the method used in paper.

# Importing Necessary Libraries
import os
import sys
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

## Scientific Computing
import numpy as np
import scipy.io
import pandas as pd

## Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.preprocessing import scale 

## Visualization
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

## Models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

## Model Evaluation
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,confusion_matrix,make_scorer
from sklearn.model_selection import cross_val_score,GridSearchCV,StratifiedKFold

class Solution:
    def __init__(self,path):
        self.data = scipy.io.loadmat(path)          ## Load Data             

    def preprocess(self):
        """
        Function to preprocess the data
        """
        intLabel = []
        for i in range(40):
            if self.data['label'][i][0][0] == 'gaucher':
                intLabel.append(0)
            else:
                intLabel.append(1)

        df = pd.DataFrame(self.data['X'])

        return df,intLabel

    def pca_reduce(self,df,intLabel,n_components):
        """
        Dimensionality Reduction
        """
        print('[PROCESS] Principal Components Discriminant Analysis')
        pca = PCA() ## min(#data_points, #features)
        X_reduced = pca.fit_transform(scale(df))
        print('Number of Components Used : ',n_components)

        # ###
        pca = PCA(n_components=n_components)
        principalComponents = pca.fit_transform(scale(df))

        principalDF = pd.DataFrame(principalComponents)
        principalDF.head()

        return principalDF,intLabel

    def evaluate(self):

        ## Data Preprocessing
        df,intLabel = self.preprocess()

        # for i in tqdm(range(7,21),desc = 'Number of Components'):
        ## After running the below code for the components ranging from 7 to 20. It was found that model trained with 13/14 components showed the maximum variance.

        X,y = self.pca_reduce(df,intLabel,n_components=13)

        rounds = 10

        ## Storing Results
        accuracy_outer_scores = np.zeros(rounds)
        accuracy_nested_scores = np.zeros(rounds)
        
        spec_outer_scores = np.zeros(rounds)
        spec_nested_scores = np.zeros(rounds)
        
        sens_outer_scores = np.zeros(rounds)
        sens_nested_scores = np.zeros(rounds)
        

        ## Model and its parameters
        model = LinearDiscriminantAnalysis(solver= 'svd')
        param_grid = {'solver' : ['svd']}


        for i in tqdm(range(rounds),desc = 'Outer Loop'):
            inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
            outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)

            # Non-nested parameter search and accuracy
            clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv)        ## Grid Search to find the best score
            clf.fit(X, y)                                                                  ## Fit Grid Search
            accuracy_outer_scores[i] = clf.best_score_                                     ## Store best score value

            # Nested CV accuracy
            nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
            accuracy_nested_scores[i] = nested_score.mean()


            ## Specificity Calculation
            specificity = make_scorer(recall_score,pos_label = 0)                          ## Custom function for calculating specificity
            # Non-nested parameter search and specificity scoring
            clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv,scoring = specificity)   ## Grid Search to find the best score
            clf.fit(X, y)       ## Fit Grid Search
            spec_outer_scores[i] = clf.best_score_      ## Store best score value

            # Nested CV specificity
            nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv,scoring = specificity)
            spec_nested_scores[i] = nested_score.mean()


            ## Sensitivity Calculation
            sensitivity = make_scorer(recall_score, pos_label=1)
            # Non-nested parameter search and Sensitivity Calculation
            clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv,scoring = sensitivity)
            clf.fit(X, y)
            sens_outer_scores[i] = clf.best_score_

            # Nested CV and sensitivity score
            nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv,scoring = sensitivity)
            sens_nested_scores[i] = nested_score.mean()
            
        
        print('[RESULTS]\n')
        print('Overall Accuracy : ', round(sum(accuracy_nested_scores)*100/len(accuracy_nested_scores),2))
        print('Overall Sensitivity : ',round(100*sum(sens_nested_scores)/len(sens_nested_scores),2))
        print('Overall Specificity : ',round(100*sum(spec_nested_scores)/len(spec_nested_scores),2))



if __name__ == "__main__":
    ## Extracting values from parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--data_path", type=str,help="Data File Path")
    args = parser.parse_args()

    if not args.data_path:
        DATA_PATH  = '../../../Gaucherdata.mat'
    else:
        DATA_PATH = args.data_path

    ## Solution
    solution = Solution(DATA_PATH)
    solution.evaluate()