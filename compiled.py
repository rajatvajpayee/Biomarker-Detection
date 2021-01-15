# Importing Necessary Libraries
import os
import sys
import argparse
import json
from datetime import datetime

## Scientific Computing
import numpy as np
import scipy.io
import pandas as pd

## Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample

## Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

## Models
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

## Model Evaluation
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,confusion_matrix,make_scorer
from sklearn.model_selection import cross_val_score,cross_validate



class Solution:
    def __init__(self,path,n_iteration,n_size):
        self.data = scipy.io.loadmat(path)          ## Loads Data
        self.n_iteration = n_iteration
        self.n_size = n_size

    def preprocess(self):
        intLabel = []
        for i in range(40):
            if self.data['label'][i][0][0] == 'gaucher':
                intLabel.append(0)
            else:
                intLabel.append(1)

        df = pd.DataFrame(self.data['X'])

        return df,intLabel

    def pca_reduce(self,df,intLabel):
        scaler = StandardScaler()
        scaler.fit(df)
        X = StandardScaler().fit_transform(df)
        pca = PCA(n_components=20)
        principalComponents = pca.fit_transform(X)

        principalDF = pd.DataFrame(principalComponents)
        principalDF.head()
        principalDF['label'] = intLabel

        return principalDF

    def evaluate(self,clf):
        curr_time = datetime.now().strftime('%H-%M_%d-%m-%Y')
        df,intLabel = self.preprocess()
        values = self.pca_reduce(df,intLabel)

        n_iteration = self.n_iteration              ## Total iterations for bootstrapping
        n_size = self.n_size                        ## Size of each sample

        ## To store scores
        stats = []          ## Store Overall Accuracies for 10-k fold
        sens = []           ## Stores Sensitivity for each sampled data
        spec = []           ## Stores Specificity for each sampled data

        ## Storing results in JSON
        results = {}

        for i in tqdm(range(n_iteration),desc = 'Iterations'):  ## Bootstrapping
            train = resample(values,n_samples = n_size)

            # Train Model
            if clf != 'SVC':
                model = LogisticRegression()
                clf = 'Logistic Regression'
            else:
                model = SVC()

            # Perform 10-cross validation
            ## Calculate Accuracy
            accuracy = cross_val_score(model,train.iloc[:,:-1], train.iloc[:,-1],cv= 10)
            score = sum(accuracy)/len(accuracy)
            stats.append(score)

            ## Calculate Specificity
            specificity = make_scorer(recall_score,pos_label = 0)
            spec_score = cross_val_score(model,train.iloc[:,:-1], train.iloc[:,-1],cv= 10,scoring=specificity)
            spec_score = sum(spec_score)/len(spec_score)
            spec.append(spec_score)
            
            ## Calculate Sensitivity
            sensitivity = make_scorer(recall_score, pos_label=1)
            sens_score = cross_val_score(model,train.iloc[:,:-1], train.iloc[:,-1],cv= 10,scoring = sensitivity)
            sens_score = sum(sens_score)/len(sens_score)
            sens.append(sens_score)
            
        
        ## Results
        results['Classifier'] = clf
        results['Bootstrapping Iterations'] = n_iteration
        results['Bootstrapping Sample Size'] = n_size
        results['Accuracy_1000'] = stats
        results['Sensitivity'] = sens
        results['Specificity'] = spec
        results['Overall_Accuracy'] = '{:.2f}%'.format(100*sum(stats)/len(stats))
        results['Overall_Sensitivity'] = '{:.2f}%'.format(100*sum(sens)/len(sens))
        results['Overall_Specificity'] = '{:.2f}%'.format(100*sum(spec)/len(spec))

        print('Accuracy : {:.2f}%'.format(100*sum(stats)/len(stats)))
        print('Sensitivity : {:.2f}% \nSpecificity : {:.2f}%'.format(100*sum(sens)/len(sens),100*sum(spec)/len(spec)))
        sns.distplot(stats)
        plt.title('Accuracy Distribution for {}'.format(clf))
        plt.xlabel('accuracy')
        plt.savefig('results/plots/{}_boot_{}_{}.png'.format(clf,n_iteration,curr_time))

        # confidence intervals
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower = max(0.0, np.percentile(stats, p))
        p = (alpha+((1.0-alpha)/2.0)) * 100
        upper = min(1.0, np.percentile(stats, p))
        print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

        ## Saving results in JSON file
        with open('results/logs/log_{}.txt'.format(curr_time),'w') as outfile:
            json.dump(results,outfile,indent=4)



if __name__ == "__main__":
    ## Initialization Parameters

    ## Extracting values from parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--data_path", type=str,help="Data File Path")
    parser.add_argument("-n","--iter", type=int,help="Iterations for bootstrapping")
    parser.add_argument("-s","--size", type=int,help="Fraction of overall data used for resampling [0,1]")
    parser.add_argument("-clf","--classifier", type=str,help="Classifier")
    args = parser.parse_args()

    if not args.data_path:
        DATA_PATH  = '../Gaucherdata.mat'
        n_iteration = 1000
        n_size = int(40*0.95)
        clf = 'SVC'

    else:
        DATA_PATH = args.data_path
        n_iteration = args.iter
        n_size = int(40*args.size)
        clf = args.classifier

    try:
        os.mkdir('results')
        os.mkdirs('results/plots')
    except:
        pass

    try:
        os.mkdir('results/logs')
    except:
        print('[LOG Directory] exist')
        pass

    ## Solution
    solution = Solution(DATA_PATH,n_iteration,n_size)
    solution.evaluate(clf)