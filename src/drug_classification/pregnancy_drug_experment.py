import operator
from copy import copy

import sklearn
from catboost import CatBoostClassifier
from keras import regularizers
from scipy.optimize import fmin
from scipy.sparse import csr_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier, \
    BaggingClassifier
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, RFE
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix, average_precision_score, \
    precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, StandardScaler, MinMaxScaler, FunctionTransformer, \
    PolynomialFeatures
from sklearn.svm import LinearSVC, SVC, NuSVC
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.export_utils import set_param_recursive
from sklearn.ensemble import StackingClassifier

t = 0.32
def senstivity(y_test, test_preds_proba):
    test_preds_class = [1 if x> t else 0 for x in test_preds_proba]
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds_class).ravel()
    sensitivity = tp / (tp + fn)
    return sensitivity
def specficity(y_test, test_preds_proba):
    test_preds_class = [1 if x > t else 0 for x in test_preds_proba]
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds_class).ravel()
    specificity = tn / (tn + fp)
    return specificity
def f1_threshold(y_test, test_preds_proba):
    test_preds_class = [1 if x > t else 0 for x in test_preds_proba]
    f1 = f1_score(y_test, test_preds_class)
    return f1
def precision_threshold(y_test, test_preds_proba):
    test_preds_class = [1 if x > t else 0 for x in test_preds_proba]
    p = precision_score(y_test, test_preds_class)
    return p
def recall_threshold(y_test, test_preds_proba):
    test_preds_class = [1 if x > t else 0 for x in test_preds_proba]
    r = recall_score(y_test, test_preds_class)
    return r

class preg_scorer():
    def __init__(self,threshold=0.36):

        self.results = {'Name':[],'Model':[],'AUPR':[], 'AUC':[],'Kappa':[],'Threshold':[],'Itertion':[],'Modalities':[],'CV':[],'Precision':[], 'Recall':[],'F1-score':[],'Specificity':[],'Sensitivity':[] }
        self.threshold = threshold

    def as_df(self):
        c = dict(self.results)
        return pd.DataFrame(c)[['Name','Model','Modalities','AUC','AUPR','F1-score','Sensitivity','Specificity','Kappa']]

    def get_best_mod(self,column_to_max='AUC'):
        base_mod = list(self.as_df().groupby('Name').mean().sort_values(by=column_to_max, ).index[-1].split('*')[0].split(';'))
        new_auc = self.as_df().groupby('Name').mean().sort_values(by=column_to_max,)[column_to_max][-1]
        return base_mod,new_auc

    def add_score(self,modalities,clfName,y_test,test_preds,rep,cv_i,best_threshold):
        auc = roc_auc_score(y_test, test_preds)
        self.results['Modalities'].append(';'.join(modalities))
        self.results['Name'].append(';'.join(modalities) + '*' + clfName)
        self.results['Model'].append(clfName)
        self.results['CV'].append(cv_i)
        self.results['Itertion'].append(rep)
        self.results['Threshold'].append(best_threshold)
        self.results['AUC'].append(auc)
        test_preds_class = [0 if x < self.threshold else 1 for x in test_preds]

        aupr = average_precision_score(y_test, test_preds)
        self.results['AUPR'].append(aupr)

        report = classification_report(y_test, test_preds_class, output_dict=True)

        self.results['Precision'].append(report['0']['precision'])
        self.results['Recall'].append(report['0']['recall'])  # ==sensitivity
        self.results['F1-score'].append(report['0']['f1-score'])

        specificity = specficity(y_test, test_preds_class)
        sensitivity = senstivity(y_test, test_preds_class)

        t = 0.35
        kappa = sklearn.metrics.cohen_kappa_score(y_test, [0 if x < t else 1 for x in test_preds])
        self.results['Kappa'].append(kappa)
        self.results['Specificity'].append(specificity)
        self.results['Sensitivity'].append(sensitivity)

    def find_best_threshold(self,y_test,test_preds):
        best_thr = None
        best_val = 0
        for x0 in range(10):
            currMinres = fmin(thr_to_accuracy, args=(y_test, test_preds), x0=x0 / 10, disp=False,
                              full_output=True)
            if best_val < -currMinres[1]:
                best_val = -currMinres[1]
                best_thr = currMinres[0]
        return best_thr

    def add_from_cv(self, modalities, clfName,  cv_score):
        for i in range(len(cv_score['fit_time'])):
            auc = cv_score['test_roc_auc'][i]
            self.results['Modalities'].append(';'.join(modalities))
            self.results['Name'].append(';'.join(modalities) + '*' + clfName)
            self.results['Model'].append(clfName)

            self.results['CV'].append(i)
            rep=0
            self.results['Itertion'].append(rep)
            best_threshold=0
            self.results['Threshold'].append(best_threshold)

            self.results['AUC'].append(auc)
            self.results['AUPR'].append(cv_score['test_average_precision'][i])


            self.results['Precision'].append(cv_score['test_precision'][i])
            self.results['Recall'].append(cv_score['test_recall'][i])  # ==sensitivity
            self.results['F1-score'].append(cv_score['test_f1'][i])

            specificity = cv_score['test_specificity'][i]
            sensitivity = cv_score['test_sensitivity'][i]
            kappa = cv_score['test_cohen_kappa_score'][i]
            self.results['Kappa'].append(kappa)
            self.results['Specificity'].append(specificity)
            self.results['Sensitivity'].append(sensitivity)
        print('done adding',modalities,clfName)

def thr_to_accuracy(thr, Y_test, predictions):
    return -f1_score(Y_test, np.array(predictions > thr, dtype=np.int))

labels_array=['Safe','Limited']
def transform_labels(data):
    return [labels_array.index(x) for x in data]

