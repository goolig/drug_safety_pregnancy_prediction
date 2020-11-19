import random
from sklearn.pipeline import make_pipeline
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score,make_scorer,cohen_kappa_score
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
from src.drug_classification.supervised_dimensionality_reduction import  get_col_clusters
from group_lasso import LogisticGroupLasso
from src.data_readers.tagged_preg_reader import tagged_data_reader
from src.drug_classification.supervised_dimensionality_reduction import extract_text_features
from src.drug_classification.pregnancy_drug_experment import  transform_labels, preg_scorer, senstivity, specficity, f1_threshold, precision_threshold,recall_threshold
from src.multimodal_learning.multimodal_classifiers import multimodal_classifiers
import pandas as pd

def add_mods(X,X2,mods,modality_name):
    #function used to add new modality to the data
    for c in X2.columns:
        mods = mods.append({'modality': modality_name, 'feature': c}, ignore_index=True)
    X = X.join(X2, how='left')
    return X, mods


seed= 0
random.seed(seed)
LogisticGroupLasso.LOG_LOSSES = True
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#Drug were read from drugbank
X_unlabled = pd.read_csv(r'data\drugData.csv.gz',compression='gzip',index_col=0)
modalities_df = pd.read_csv(r'data\modalities.csv.gz',compression='gzip',index_col=0)
print('done reading from disk')

#Setting relevant modalities
base_mods = ['ATC_Level_1_description', 'ATC_Level_2_description', 'ATC_Level_3_description', 'ATC_Level_4_description',  'ATC_Level_5',  'Associated_condition',  'Carrier','Category',  'Enzyme', 'Group',  'Target', 'Taxonomy', 'Transporter', 'Type']
mods_domain_expert = ['Category', 'ATC_Level_3_description', 'ATC_Level_2_description', 'ATC_Level_4_description', 'Associated_condition', 'Taxonomy']

#Adding our features
X_unlabled_text = extract_text_features(X_unlabled[modalities_df.loc[modalities_df.modality.isin(mods_domain_expert), 'feature']])
X_unlabled, modalities_df= add_mods(X_unlabled, X_unlabled_text, modalities_df, 'Text')
print(X_unlabled_text.columns)
text_mods = ['Text']
print('done processing text')
num_clusters=3600 #Number of custers to create. 3600 was optimal in paper.
clustering_mods=[]
X_cluster = get_col_clusters(X_unlabled[modalities_df.loc[modalities_df.modality.isin(mods_domain_expert), 'feature']], num_clusters)
print('clusters head:')
print(X_cluster.head())
print(X_cluster.columns)
mod_name='Clusters'
clustering_mods.append(mod_name)
X_unlabled , modalities_df= add_mods(X_unlabled, X_cluster, modalities_df, mod_name)
print('done clustering')

#Writing new data to disk
X_unlabled.to_csv(r'data\drugData_w_text.csv.gz',compression='gzip')
modalities_df.to_csv(r'data\modalities_w_text.csv.gz',compression='gzip')

print("done preproc")
all_modalities = list(modalities_df.modality.unique())
print('mods:',all_modalities)

def cv_eval(X_unlabled, classifiers, exp_name):
    # ###### CV exp
    random.seed(seed)
    kf = RepeatedStratifiedKFold(n_splits=10, random_state=0, n_repeats=reps)
    preg_tagged_data_reader = tagged_data_reader()
    X_cv = X_unlabled.join(
        preg_tagged_data_reader.read_all(remove_disputed=True, read_smc=True, read_Eltonsy_et_al=False,
                                         read_safeFetus=False), how='inner')
    y_cv = X_cv['preg_class']
    y_cv = pd.Series(transform_labels(y_cv), index=y_cv.index, name='tag')
    del X_cv['preg_class']

    results_collector = preg_scorer()
    start_time = time.monotonic()
    for current_mods,current_algorithm,clf in classifiers:
        cv_score_all = cross_validate(estimator=clf, X=X_cv, y=y_cv, cv=kf, scoring=metrics, n_jobs=1)
        results_collector.add_from_cv(current_mods, current_algorithm,  cv_score_all)
    print('minutes: ',(time.monotonic() - start_time)/60)
    print(results_collector.as_df().groupby(['Name','Model','Modalities']).mean().sort_values(by='AUC',))
    results_collector.as_df().to_csv(os.path.join('output','results_'+exp_name+'_cv.csv'))


def cross_expert_eval(X_unlabled, classifiers, exp_name,remove_disputed):
    preg_tagged_data_reader = tagged_data_reader()
    train_set = X_unlabled.join(preg_tagged_data_reader.read_all(remove_disputed=remove_disputed,read_who=True,
                                                                 read_smc=False,read_Eltonsy_et_al=False,read_safeFetus=False),how='inner')
    y_train = train_set['preg_class']
    y_train = pd.Series(transform_labels(y_train),index=y_train.index,name='tag')
    X_train = train_set
    del X_train['preg_class']

    test_set = X_unlabled.join(preg_tagged_data_reader.read_all(remove_disputed=remove_disputed,read_who=False,
                                                                read_smc=True,read_Eltonsy_et_al=False,
                                                                read_safeFetus=False),how='inner')
    print('Removing drug appearing in train ans test both:',len(set(test_set.index) & set(X_train.index)))

    test_set = test_set.loc[~test_set.index.isin(train_set.index)]
    y_test = test_set['preg_class']
    y_test = pd.Series(transform_labels(y_test),index=y_test.index,name='tag')
    X_test = test_set
    del X_test['preg_class']

    results_collector_db = preg_scorer()

    for current_mods,current_algorithm,clf in classifiers:
        for rep in range(reps):
            try:
                clf.fit(X_train, y_train)
                ExtraTrees_preds = clf.predict_proba(X_test)[:, 1]
                results_collector_db.add_score(current_mods,current_algorithm, y_test, ExtraTrees_preds, rep, 0, 0)
            except:
                print('cannot run',current_algorithm)
    print(results_collector_db.as_df().groupby(['Name','Model','Modalities']).mean().sort_values(by='AUC'))
    results_collector_db.as_df().to_csv(os.path.join('output','results_'+exp_name+'_db.csv'))
    print('Number of drugs in train but not in test:',len(set(X_train.index) - set(X_test.index)))
    print('Number of drugs in test but not in train:',len(set(X_test.index) - set(X_train.index)))
    print('Number of drugs in test but not in both:',len(set(X_test.index) & set(X_train.index)))

metrics = {'cohen_kappa_score':make_scorer(cohen_kappa_score),'roc_auc':make_scorer(roc_auc_score,needs_proba=True), 'f1':make_scorer(f1_threshold,needs_proba=True), 'precision':make_scorer(precision_threshold,needs_proba=True), 'recall':make_scorer(recall_threshold,needs_proba=True),
           'average_precision':make_scorer(average_precision_score,needs_proba=True),'sensitivity':make_scorer(senstivity,needs_proba=True),'specificity':make_scorer(specficity,needs_proba=True)}
reps=1 #number of CV and cross db repeats. 30 was used in paper

classifiers_man = multimodal_classifiers(modalities_df,list(X_unlabled.columns))

#Base models we use
algos = [
        ('ExtraTreesClassifier',ExtraTreesClassifier(n_jobs=1)),
        ('RandomForestClassifier',RandomForestClassifier(n_jobs=1)),
        ('LogisticRegression',make_pipeline(StandardScaler(),LogisticRegression(n_jobs=1))),
        ('KNeighborsClassifier',KNeighborsClassifier(n_jobs=1,metric='jaccard')),
        ('MultinomialNB',make_pipeline(Normalizer(norm="max"),MultinomialNB(alpha=0.01, fit_prior=True)))
         ]

#Adding all modality selection and base model combinations
classifiers_exp_single_mods = []
for algo_name, algo in algos: #all mods
    classifiers_exp_single_mods.append((['Lasso'], algo_name, classifiers_man.get_lasso_classifier(base_mods, algo, 0.01, X_unlabled[ modalities_df.loc[modalities_df.modality.isin(base_mods), 'feature']])))
    classifiers_exp_single_mods.append((['Greedy selection'], algo_name, classifiers_man.get_greedy_addition_group_selection_predictions(base_mods, algo)))
    for mods_name,m in [
        ('All',base_mods),
        ('Domain expert',mods_domain_expert),
        ('All + Clusters',base_mods+clustering_mods),
        ('All + Text',base_mods+text_mods),
        ('Clusters',clustering_mods),
        ('Text',text_mods)
        ]:
        classifiers_exp_single_mods.append(([mods_name], algo_name, classifiers_man.get_modalities_base_clf(m, algo)))
        if len(m)>1: #only for more than 1 modality
            classifiers_exp_single_mods.append(([mods_name],'Voting + ' + algo_name,classifiers_man.get_mod_voting(m,algo)))
            voting_clf_ort, stacking_clf_ort = classifiers_man.get_orthogonal_stacking_voting(m,algo)
            classifiers_exp_single_mods.append(([mods_name],'Orthogonal small + '+algo_name,voting_clf_ort))

# large ortho, it uses the same base models, makes no sense to put it in the loop.
for mods_name, m in [
    ('All', base_mods),
    ('Domain expert', mods_domain_expert),
    ('All + Clusters', base_mods + clustering_mods),
    ('All + Text', base_mods + text_mods),
                     ]:
    voting_clf_ort_large, stacking_clf_ort_large = classifiers_man.get_orthogonal_large_voting_stacking(m)
    classifiers_exp_single_mods.append(([mods_name],'Voting orthogonal large',voting_clf_ort_large))

#Adding all single modalities with base models
for algo_name, algo in algos: #all algo
    for m in [[x] for x in base_mods]:
        print(m)
        classifiers_exp_single_mods.append((m, algo_name, classifiers_man.get_modalities_base_clf(m, algo)))

#Run experiments
results_collector = cv_eval(X_unlabled, classifiers_exp_single_mods, exp_name='all_exp')
results_collector_db = cross_expert_eval(X_unlabled, classifiers_exp_single_mods, exp_name='all_exp',remove_disputed=True)
results_collector_db_w_disagreed = cross_expert_eval(X_unlabled, classifiers_exp_single_mods, exp_name='all_exp_w_disputed',remove_disputed=False)

