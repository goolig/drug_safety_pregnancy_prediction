import random
import time
import xgboost
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL

import os
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import make_scorer, cohen_kappa_score, average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from src.data_readers.tagged_preg_reader import tagged_data_reader
from src.drug_classification.pregnancy_drug_experment import transform_labels, preg_scorer, \
    recall_threshold, precision_threshold, specficity, f1_threshold, senstivity

class optimize():
    def __init__(self,X,y,evals):
        self.evals=evals
        booster = ['gblinear', 'gbtree', 'dart']
        self.X=X
        self.y=y
        self.best_params = None
        self.space ={
                'n_estimators': hp.uniformint('n_estimators', 1, 100),  # hp.normal
                'eta': hp.quniform('eta', 0.001, 1.0, 0.001),
                'skip_drop': hp.quniform('skip_drop', 0.0, 1.0, 0.01),#only dart
                'rate_drop': hp.quniform('rate_drop', 0.0, 1.0, 0.01),#only dart
                'normalize_type': hp.choice('normalize_type', ['tree', 'forest']),  # only dart
                'sample_type': hp.choice('sample_type', ['uniform', 'weighted']),  # only dart
                'max_depth': hp.uniformint('max_depth', 1, 13),
                'min_child_weight': hp.uniformint('min_child_weight', 1, 50),
                'subsample': hp.quniform('subsample', 0.1, 1, 0.01),
                'gamma': hp.quniform('gamma', 0.1, 10, 0.01),
                'lambda': hp.quniform('lambda', 0.1, 10, 0.01),
                'alpha': hp.quniform('alpha', 0.1, 10, 0.01),
                'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.01),
                'colsample_bylevel': hp.quniform('colsample_bylevel', 0.1, 1, 0.01),
                'colsample_bynode': hp.quniform('colsample_bynode', 0.1, 1, 0.01),
                'booster': hp.choice('booster', booster),
                'objective': 'binary:logistic',
                #'nthread': 6,
                'silent': 1,
                }
    def score(self,params):
        print("Training with params : ")
        print(params)
        stat = STATUS_OK
        curr_score = 0
        try:
            model = xgb.XGBClassifier(**params)
            kf = RepeatedStratifiedKFold(n_splits=10, random_state=0, n_repeats=1)
            scores = cross_validate(estimator=model, X=self.X, y=self.y, cv=kf, scoring='roc_auc', n_jobs=-1)
            scores = scores['roc_auc']
            curr_score = np.array(scores).mean()
        except:
            stat = STATUS_FAIL
        print("\tAUC {0}\n\n".format(curr_score))
        return {'loss': 1 - curr_score, 'status': stat}

    def optimize(self,trials):
        best = fmin(self.score, self.space, algo=tpe.suggest, trials=trials, max_evals=self.evals,return_argmin =False)
        print('best params:')
        print(best)
        self.best_params = best

seed= 0
random.seed(seed)
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
metrics = {'cohen_kappa_score':make_scorer(cohen_kappa_score),'roc_auc':make_scorer(roc_auc_score,needs_proba=True), 'f1':make_scorer(f1_threshold,needs_proba=True), 'precision':make_scorer(precision_threshold,needs_proba=True), 'recall':make_scorer(recall_threshold,needs_proba=True),
           'average_precision':make_scorer(average_precision_score,needs_proba=True),'sensitivity':make_scorer(senstivity,needs_proba=True),'specificity':make_scorer(specficity,needs_proba=True)}

#Read data
#Drug were read from drugbank
X_unlabled = pd.read_csv(r'data\drugData.csv.gz',compression='gzip',index_col=0)
modalities_df = pd.read_csv(r'data\modalities.csv.gz',compression='gzip',index_col=0)
print('done reading from disk')

# morgan = pd.read_json(os.path.join('pickles','data','morgan_fp','morgan.json')).set_index('drugBank_id').drop('Smiles',axis=1)
# a = np.array(morgan.morgan_fp.tolist())
# mat = [[0 for x in range(1024)] for y in range(len(a))]
# for x in range(len(a)):
#     for y in range(1024):
#         if a[x] is not None:
#             mat[x][y]=a[x][y]
#         else:
#             mat[x][y] = np.nan
# morgan = pd.DataFrame(mat, index= morgan.index)
# morgan = morgan.dropna()
morgan = pd.read_csv(os.path.join('pickles','data','Challa','1,024-bit Morgan fingerprints for 611 eligible drugs.csv'),index_col=0)
morgan = morgan.set_index('V1')
morgan.columns = [str(x) for x in range(1024)]
rar = pd.read_csv(os.path.join('pickles','data','Challa','Drugs with RAR and HDAC Tox21 coverage.csv'),index_col=0)
rar['is_RAR_HDAC'] =1
MOE = pd.read_csv(os.path.join('pickles','data','Challa','MOE calculations for all of 611 drugs with coverage in NCATS compound library.csv'),index_col='DRUGBANK_ID')
MOE = MOE.drop(labels = ['mol','SECONDARY_ACCESSION_NUMBERS','COMMON_NAME','CAS_NUMBER','UNII','SYNONYMS'],axis=1)
#MOE, from comments in original code: the cutoff of HOMO energy at -9.570580 eV appears to optimize sensitivity (~52%) and specificity (~52%).
#MOW, from comments in original code: the cutoff of LUMO energy at -5.196165 eV appears to optimize sensitivity (~55%) and specificity (~53%).

# drug_smiles = pd.read_csv(os.path.join('pickles','data','Challa','MOE calculations for all of 611 drugs with coverage in NCATS compound library.csv'),index_col='mol')[['DRUGBANK_ID']]
# rar_data = pd.read_csv(os.path.join('pickles','data','Challa','Tox21','RAR antagonist Tox21.csv'),index_col='SMILES').drop(labels=['SAMPLE_NAME'],axis=1)
# hdac_data = pd.read_csv(os.path.join('pickles','data','Challa','Tox21','HDAC Tox21.csv'),index_col='SMILES').drop(labels=['SAMPLE_NAME'],axis=1)
# rar_data.pivot_table(index=rar_data.index, columns='ASSAY_OUTCOME')
# size= hdac_data.groupby(hdac_data.index).size()
# size.name='size'
# drug_smiles.join(size,how='left')

finalChallData = morgan.copy()
assert finalChallData.isna().sum().sum()==0
finalChallData = finalChallData.join(rar).fillna(0)
finalChallData = finalChallData.join(MOE)

random.seed(seed)
preg_tagged_data_reader = tagged_data_reader()
X_cv = finalChallData.join(
    preg_tagged_data_reader.read_all(remove_disputed=True, read_smc=True, read_Eltonsy_et_al=False,
                                     read_safeFetus=False), how='inner')
y_cv = X_cv['preg_class']
y_cv = pd.Series(transform_labels(y_cv), index=y_cv.index, name='tag')
del X_cv['preg_class']

results_collector = preg_scorer()
start_time = time.monotonic()
reps=30

y_cv_shtar = X_unlabled.join(y_cv,how='right')['tag']
X_cv_shtar = X_unlabled[X_unlabled.index.isin(y_cv_shtar.index)]


xgb = xgboost.XGBClassifier()
ExtraTrees = ExtraTreesClassifier()
kf = RepeatedStratifiedKFold(n_splits=10, random_state=0, n_repeats=reps)

results_collector.add_from_cv(['Challa'], 'xgb', cross_validate(estimator=xgb, X=X_cv, y=y_cv, cv=kf, scoring=metrics, n_jobs=-1))
results_collector.add_from_cv(['Category'], 'extraTrees', cross_validate(estimator=ExtraTrees, X=X_cv_shtar, y=y_cv_shtar, cv=kf, scoring=metrics, n_jobs=-1))

print('xgb','minutes: ', (time.monotonic() - start_time) / 60)
#print(results_collector.as_df().groupby(['Name', 'Model', 'Modalities']).mean().sort_values(by='AUC', ))
print('minutes: ', (time.monotonic() - start_time) / 60)
print(results_collector.as_df().groupby(['Name', 'Model', 'Modalities']).mean().sort_values(by='AUC', ))
#results_collector.as_df().to_csv(os.path.join('output', 'results', 'results_' + exp_name + '_cv.csv'))
results_collector.as_df().to_csv(os.path.join('output','data','chall_cv.csv'))



finalChallData = morgan.copy()
assert finalChallData.isna().sum().sum()==0
finalChallData = finalChallData.join(rar).fillna(0)
finalChallData = finalChallData.join(MOE)

preg_tagged_data_reader = tagged_data_reader()
train_set = finalChallData.join(preg_tagged_data_reader.read_all(remove_disputed=True, read_who=True,
                                                             read_smc=False, read_Eltonsy_et_al=False,
                                                             read_safeFetus=False), how='inner')
y_train = train_set['preg_class']
y_train = pd.Series(transform_labels(y_train), index=y_train.index, name='tag')
X_train = train_set
del X_train['preg_class']

test_set = finalChallData.join(preg_tagged_data_reader.read_all(remove_disputed=True, read_who=False,
                                                            read_smc=True, read_Eltonsy_et_al=False,
                                                            read_safeFetus=False), how='inner')
print('Removing drug appearing in train ans test both:', len(set(test_set.index) & set(X_train.index)))

test_set = test_set.loc[~test_set.index.isin(train_set.index)]
y_test = test_set['preg_class']
y_test = pd.Series(transform_labels(y_test), index=y_test.index, name='tag')
X_test = test_set
del X_test['preg_class']

y_train_shtar = X_unlabled.join(y_train,how='right')['tag']
X_train_shtar = X_unlabled[X_unlabled.index.isin(y_train_shtar.index)].loc[:,modalities_df.loc[modalities_df.modality=='Category','feature']]


y_test_shtar = X_unlabled.join(y_test,how='right')['tag']
X_test_shtar = X_unlabled[X_unlabled.index.isin(y_test_shtar.index)].loc[:,modalities_df.loc[modalities_df.modality=='Category','feature']]


results_collector_db = preg_scorer()
for rep in range(reps):
    xgb.fit(X_train, y_train)
    results_collector_db.add_score(['challa'], 'xgb', y_test, xgb.predict_proba(X_test)[:, 1], rep, 0, 0)
    ExtraTrees.fit(X_train_shtar, y_train_shtar)
    results_collector_db.add_score(['Category'], 'ExtraTrees', y_test_shtar, ExtraTrees.predict_proba(X_test_shtar)[:, 1], rep, 0, 0)
print(results_collector_db.as_df().groupby(['Name', 'Model', 'Modalities']).mean().sort_values(by='AUC'))


#
# dr = set()
# for field in['Category: Opioids','Category: Narcotics','Category: Opioid Agonist' ,'Category: NMDA Receptor Antagonists']:
#     data = X_unlabled.join(
#         preg_tagged_data_reader.read_all(remove_disputed=True, read_smc=False, read_Eltonsy_et_al=False,
#                                          read_safeFetus=False), how='inner')
#     data = data[~data.preg_class.isna()]
#     data = data[data[field]==True]
#     dr= dr | set(data.index)
#     print(data[[field,'preg_class']])