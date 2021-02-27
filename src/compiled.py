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
import math

## Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.preprocessing import scale 

## Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

## Models
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression

## Model Evaluation
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,confusion_matrix,make_scorer
from sklearn.model_selection import cross_val_score,cross_validate,StratifiedKFold

class Solution:
    def __init__(self,path,n_iteration,n_size):
        self.data = scipy.io.loadmat(path)          ## Load Data
        self.n_iteration = n_iteration              ## Set number of iterations
        self.n_size = n_size                        ## Resampling Size 

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

    def pca_reduce(self,df,intLabel,clf):
        """
        Dimensionality Reduction
        """
        print('[PROCESS] Principal Components Analysis')
        pca = PCA(n_components = min(len(df.columns),len(df)))
        X_reduced = pca.fit_transform(scale(df))
        
        plt.figure(figsize = (10,6))
        plt.ylim(0,max(pca.explained_variance_))
        plt.title('Principal Components Selection')
        plt.plot(pca.explained_variance_)
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Eigenvalue')
        plt.savefig('results/plots/{}_PCA.png'.format(clf))
        # plt.show()


        n_components = 14
        print('[Int. Result] Maximum Variance is explained when {} components are used'.format(n_components))

        # ###
        pca = PCA(n_components=n_components)
        principalComponents = pca.fit_transform(scale(df))

        principalDF = pd.DataFrame(principalComponents)
        principalDF.head()
        principalDF['label'] = intLabel
        return principalDF

    def evaluate(self,clf):
        curr_time = datetime.now().strftime('%H-%M_%d-%m-%Y')
        df,intLabel = self.preprocess()
        values = self.pca_reduce(df,intLabel,clf)

        n_iteration = self.n_iteration              ## Total iterations for bootstrapping
        n_size = self.n_size                        ## Size of each sample

        ## To store scores
        stats = []          ## Store Overall Accuracies for 10-k fold
        sens = []           ## Stores Sensitivity for each sampled data
        spec = []           ## Stores Specificity for each sampled data

        ## Storing results in JSON
        results = {}

        for i in tqdm(range(n_iteration),desc = '[ Iterations ]'):  ## Bootstrapping
            train = resample(values,n_samples = n_size)
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)

            # Train Model
            if clf != 'SVC':
                model = LogisticRegression()
                clf = 'Logistic Regression'
            else:
                model = SVC(kernel='linear')

            # Perform 10-cross validation
            ## Calculate Accuracy
            accuracy = cross_val_score(model,train.iloc[:,:-1], train.iloc[:,-1],cv= cv)
            score = sum(accuracy)/len(accuracy)
            stats.append(score)

            ## Calculate Specificity
            specificity = make_scorer(recall_score,pos_label = 0)
            spec_score = cross_val_score(model,train.iloc[:,:-1], train.iloc[:,-1],cv= cv,scoring=specificity)
            spec_score = sum(spec_score)/len(spec_score)
            spec.append(spec_score)
            
            ## Calculate Sensitivity
            sensitivity = make_scorer(recall_score, pos_label=1)
            sens_score = cross_val_score(model,train.iloc[:,:-1], train.iloc[:,-1],cv= cv,scoring = sensitivity)
            sens_score = sum(sens_score)/len(sens_score)
            sens.append(sens_score)
            
        
        ## Results
        results['Classifier'] = clf
        results['Bootstrapping Iterations'] = n_iteration
        results['Bootstrapping Sample Size'] = n_size
        results['Overall_Accuracy'] = '{:.2f}%'.format(100*sum(stats)/len(  stats))
        results['Overall_Sensitivity'] = '{:.2f}%'.format(100*sum(sens)/len(sens))
        results['Overall_Specificity'] = '{:.2f}%'.format(100*sum(spec)/len(spec))
        
        
        plt.figure(figsize = (10,6))
        print('Accuracy : {:.2f}%'.format(100*sum(stats)/len(stats)))
        print('Sensitivity : {:.2f}% \nSpecificity : {:.2f}%'.format(100*sum(sens)/len(sens),100*sum(spec)/len(spec)))
        sns.distplot(stats)
        plt.title('Accuracy Distribution for {}'.format(clf))
        plt.xlabel('accuracy')
        plt.savefig('results/plots/{}_boot_{}_{}.png'.format(clf,n_iteration,curr_time))

        plt.figure(figsize=(15,3))
        plt.plot(score)
        plt.subplot(1,3,1)
        plt.title('Accuracy')
        plt.xlabel('Iteration')
        
        plt.subplot(1,3,2)
        plt.plot(sens)
        plt.title('Sensitivity')
        plt.xlabel('Iteration')
        
        plt.subplot(1,3,3)
        plt.plot(spec)
        plt.title('Specificity')
        plt.xlabel('Iteration')
        
        plt.savefig('results/plots/{}_accuracy_{}.png'.format(clf,n_iteration,curr_time))


        # confidence intervals
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower = max(0.0, np.percentile(stats, p))
        p = (alpha+((1.0-alpha)/2.0)) * 100
        upper = min(1.0, np.percentile(stats, p))
        print('[Accuracy] %.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

        lower = max(0.0, np.percentile(sens, p))
        p = (alpha+((1.0-alpha)/2.0)) * 100
        upper = min(1.0, np.percentile(sens, p))
        print('[Sensitivity] %.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

        lower = max(0.0, np.percentile(spec, p))
        p = (alpha+((1.0-alpha)/2.0)) * 100
        upper = min(1.0, np.percentile(spec, p))
        print('[Specificity] %.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

        results['Confidence Interval'] = '{}% : {}% - {}%'.format(alpha*100, lower*100, upper*100)
        results['Accuracy_1000'] = stats
        results['Sensitivity'] = sens
        results['Specificity'] = spec

        ## Saving results in JSON file
        with open('results/logs/{}_{}.txt'.format(clf,curr_time),'w') as outfile:
            json.dump(results,outfile,indent=3)

        return stats,sens,spec


if __name__ == "__main__":
    ## Initialization Parameters

    ## Extracting values from parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--data_path", type=str,help="Data File Path")
    parser.add_argument("-n","--iter", type=int,help="Iterations for bootstrapping")
    parser.add_argument("-s","--size", type=int,help="Fraction of overall data used for resampling [0,1]")
    parser.add_argument("-clf","--classifier", type=str,help="Select any classifier from (SVC / Logistic Regression) ")
    args = parser.parse_args()

    if not args.data_path:
        DATA_PATH  = '../../../Gaucherdata.mat'
        n_iteration = 1000
        n_size = int(40*0.90)
        # clf = 'Logistic_Regression'
        clf = 'SVC'

    else:
        DATA_PATH = args.data_path
        n_iteration = args.iter
        n_size = int(40*args.size)
        clf = args.classifier

    try:
        os.mkdir('results')
        os.mkdir('results/plots')
    except:
        pass

    try:
        os.mkdir('results/logs')
    except:
        pass

    ## Solution
    solution = Solution(DATA_PATH,n_iteration,n_size)
    solution.evaluate(clf)