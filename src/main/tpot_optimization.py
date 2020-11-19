import random

import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from src.data_readers.tagged_preg_reader import tagged_data_reader
from src.drug_classification.pregnancy_drug_experment import transform_labels, preg_scorer

#Drug were read from drugbank
X_unlabled = pd.read_csv(r'data\drugData.csv.gz',compression='gzip',index_col=0)
modalities_df = pd.read_csv(r'data\modalities.csv.gz',compression='gzip',index_col=0)
print('done reading from disk')


print('done reading from disk')
preg_tagged_data_reader = tagged_data_reader()
X_cv = X_unlabled.join(preg_tagged_data_reader.read_all(remove_disputed=True,read_smc=True,read_Eltonsy_et_al=False,read_safeFetus=False),how='inner')
y_cv = X_cv['preg_class']
y_cv = pd.Series(transform_labels(y_cv),index=y_cv.index,name='tag')
del X_cv['preg_class']


kf = RepeatedStratifiedKFold(n_splits=10, random_state=0, n_repeats=2)
tpot = TPOTClassifier(generations=5, population_size=20,max_eval_time_mins=1,
                      cv=kf,scoring='roc_auc',max_time_mins=60*3,verbosity=2,n_jobs=-1,
                     early_stop=2,config_dict='TPOT light') #verbosity=1,
#config_dict='TPOT sparse' 'TPOT MDR' 'TPOT light'
print(tpot)
tpot.fit(X_cv, y_cv)
f_name = 'tpot_model'+str(random.uniform(1.5, 1.9))+'.py'
tpot.export(f_name)
with open(f_name, 'r') as fin:
    print(fin.read())


results_collector = preg_scorer()
metrics = ['roc_auc', 'f1', 'precision', 'recall','average_precision']
cv_score_all = cross_validate(estimator=tpot.fitted_pipeline_, X=X_cv, y=y_cv, cv=kf, scoring=metrics, n_jobs=-1)
results_collector.add_from_cv(['all'], 'tpot',  cv_score_all)
with open(f_name, 'r') as fin:
    print(fin.read())
print(results_collector.as_df().groupby('Name').mean().sort_values(by='AUC',))
