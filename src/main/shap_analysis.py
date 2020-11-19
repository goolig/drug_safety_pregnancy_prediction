import os
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesClassifier

from src.data_readers.tagged_preg_reader import tagged_data_reader
from src.drug_classification.pregnancy_drug_experment import transform_labels

os.chdir(".."); os.chdir(".."); os.chdir("..")
def add_mods(X,X2,mods,modality_name):
    for c in X2.columns:
        mods = mods.append({'modality': modality_name, 'feature': c}, ignore_index=True)
    X = X.join(X2, how='left')
    return X, mods

def get_data_for_output_table():
    # Drug were read from drugbank
    X_unlabled = pd.read_csv(r'data\drugData.csv.gz', compression='gzip', index_col=0)
    modalities_df = pd.read_csv(r'data\modalities.csv.gz', compression='gzip', index_col=0)
    print('done reading from disk')
    drugs = X_unlabled[modalities_df.loc[(modalities_df.feature=='Group: approved')|(modalities_df.feature=='Group: withdrawn'),'feature']]
    drugs.columns = ['Approved','Withdrawn']
    drugs = drugs.replace(True,'Yes')
    drugs = drugs.replace(False,'No')

    return drugs

def get_data(remove_disputed=False, red_smc_train=False):
    # Read data
    # Drug were read from drugbank
    X_unlabled = pd.read_csv(r'data\drugData.csv.gz', compression='gzip', index_col=0)
    modalities_df = pd.read_csv(r'data\modalities.csv.gz', compression='gzip', index_col=0)
    print('done reading from disk')

    # mods_domain_expert = ['Category', 'ATC_Level_3_description', 'ATC_Level_2_description', 'ATC_Level_4_description','Associated_condition', 'Taxonomy']
    # X_cluster = get_col_clusters(X_unlabled[modalities_df.loc[modalities_df.modality.isin(mods_domain_expert), 'feature']], 3600)
    # print('clusters head:')
    # print(X_cluster.head())
    # print(X_cluster.columns)
    # X_unlabled, modalities_df = add_mods(X_unlabled, X_cluster, modalities_df, ['Clusters'])
    # print('done clustering')



    #X_unlabled.columns = [x.replace('[', '(').replace(']', ')').replace('<', '-') for x in X_unlabled.columns] #for XGBoost
    preg_tagged_data_reader = tagged_data_reader()
    train_set = X_unlabled.join(preg_tagged_data_reader.read_all(remove_disputed=remove_disputed, read_who=True,
                                                                 read_smc=red_smc_train, read_Eltonsy_et_al=False,
                                                                 read_safeFetus=False), how='inner')
    y_train = train_set['preg_class']
    y_train = pd.Series(transform_labels(y_train), index=y_train.index, name='tag')
    X_train = train_set
    del X_train['preg_class']

    if not read_smc_train:
        test_set = X_unlabled.join(preg_tagged_data_reader.read_all(remove_disputed=remove_disputed, read_who=False,
                                                                     read_smc=True, read_Eltonsy_et_al=False,
                                                                     read_safeFetus=False), how='inner')
        y_test = test_set['preg_class']
        y_test = pd.Series(transform_labels(y_test), index=y_test.index, name='tag')
        X_test = test_set
        del X_test['preg_class']
    else:
        X_test=pd.DataFrame(columns=X_unlabled.columns)
        y_test=pd.DataFrame(columns=X_unlabled.columns)
    return y_train, X_train, X_test,  X_unlabled, y_test, modalities_df


def get_drug_names():
    return pd.read_csv(r'c:\tmp\drug_names.csv.gz', compression='gzip',index_col=0)



def plot_shap_force(drug_id,expected_value,shap_values_test,data_for_shap,drug_names,X_train,force_plot_file_type,dpi,eval_label):
    curr_shap_value = shap_values_test.loc[drug_id, :]  # explainer.shap_values(curr_drug_features)
    curr_features = data_for_shap.loc[drug_id, :]
    drug_name = drug_names.loc[drug_id, 'Drug name']
    title = 'Probability higher risk (%s)' % (drug_name)

    curr_feature_vals = np.array(['Yes' if bool(x) else "No" if X_train.columns[i]!='Number of Category' else x for i,x in enumerate(curr_features.values)])

    p = shap.force_plot(base_value=expected_value,
                        shap_values=curr_shap_value.values,
                        #feature_names=[x.replace("Cluster: ",'').replace(';','\n') for x in X_train.columns], #x.split(': ')[1] if ': ' in x else
                        feature_names=[x.split(': ')[1] if ': ' in x else x for x in X_train.columns],
                        features=curr_feature_vals,#['Yes' if x else 'No' for x in curr_features.values],
                        out_names=[title], figsize=(20, 4)  #
                        , show=False, matplotlib=True, text_rotation=int(45 / 2)
                        )
    p.savefig(os.path.join('output', 'SHAP'+"_"+eval_label, drug_id + '.' + force_plot_file_type), dpi=dpi, bbox_inches='tight')
    # p.show()
    plt.close('all')

    shap.decision_plot(base_value=expected_value,  shap_values=curr_shap_value.values,
                           feature_names=[x.split(': ')[1] if ': ' in x else x for x in X_train.columns],
                           features=curr_feature_vals,
                           feature_display_range=slice(-1, -11, -1),
                           title=title,
                           show=False,
                            #link='logit',
                           #highlight=0
                           )
    p = plt.gcf()
    p.savefig(os.path.join('output', 'SHAP'+"_"+eval_label, drug_id + '_decision_plot.'+ force_plot_file_type), dpi=dpi, bbox_inches='tight')
    plt.close('all')


def run_shap(data_for_shap, eval_label='test', num_reps=30, force_plot_file_type='png', plots_dpi=100):
    shap_values_test = []
    predictions_cat_model = []
    #predictions_best_model = []
    expected_value = []
    print('Calculating shap values')


    for i in range(num_reps):
        clf = ExtraTreesClassifier(n_jobs=-1)
        clf.fit(X_train, y_train)
        clf_best = ExtraTreesClassifier(n_jobs=-1)
        #clf_best.fit(X_train_clustering, y_train)
        explainer = shap.TreeExplainer(clf, X_train)  # model_output='predict_proba'
        expected_value.append(explainer.expected_value[1])
        shap_values_test.append(explainer.shap_values(data_for_shap)[1])
        predictions_cat_model.append(clf.predict_proba(data_for_shap)[:, 1])
        if i % 5 == 0:
            print('Done shap rep:', i)
    expected_value = np.array(expected_value).mean()
    shap_values_test = pd.DataFrame(np.array(shap_values_test).mean(axis=0), index=data_for_shap.index)
    shap.summary_plot(shap_values_test.values, data_for_shap, show=False, max_display=10)
    img = plt.gcf()
    img.show()
    img.savefig(os.path.join('output', 'SHAP' +'_' + eval_label, 'summary_plot_test.pdf'), bbox_inches='tight',
                pad_inches=0.2)  # dpi=150,
    # Single drugs
    drug_names = get_drug_names()
    preds = pd.DataFrame(np.array(predictions_cat_model).mean(axis=0), columns=['Risk score'], index=data_for_shap.index)
    preds = preds.join(drug_names, how='left')
    preds = preds.sort_values(by=['Risk score'])
    for i, v in enumerate(list(preds.index)):
        plot_shap_force(v, expected_value, shap_values_test, data_for_shap, drug_names, X_train, force_plot_file_type,plots_dpi,eval_label)
        if i > 0 and i % 100 == 0:
            print(i)
            # break
    preg_tagged_data_reader = tagged_data_reader(read_details=True)
    who_tags = \
    preg_tagged_data_reader.read_all(remove_disputed=False, read_who=True, read_smc=False, read_Eltonsy_et_al=False,
                                     read_safeFetus=False)['preg_class'].rename('Polifka et al')
    smc_tags = \
    preg_tagged_data_reader.read_all(remove_disputed=False, read_who=False, read_smc=True, read_Eltonsy_et_al=False,
                                     read_safeFetus=False)['preg_class'].rename('Zerifin TIS')
    eltonsy_tags = \
    preg_tagged_data_reader.read_all(remove_disputed=False, read_who=False, read_smc=False, read_Eltonsy_et_al=True,
                                     read_safeFetus=False)['preg_class'].rename('Eltonsy et al')
    fda_code_tags = preg_tagged_data_reader.read_safeFetus(remove_disputed=False, convert_binary=False)[
        'preg_class'].rename('FDA category')  # safefetus.com
    for df in [who_tags, smc_tags, eltonsy_tags, fda_code_tags]:
        df.replace('Limited', 'Higher Risk', inplace=True)
        df.replace('Safe', 'Lower Risk', inplace=True)
    wiki = pd.read_excel(os.path.join('pickles', 'data', 'preg', 'wikipedia.xlsx'))
    wiki = wiki[~wiki.drugBank_id.isna()]
    wiki = wiki[['drugBank_id', 'Pregnancy category']]
    wiki_us = wiki[wiki['Pregnancy category'].str.contains('US ')]
    wiki_aus = wiki[wiki['Pregnancy category'].str.contains('Australian ')]
    wiki_us['Pregnancy category'] = wiki_us['Pregnancy category'].str.replace('US pregnancy category ', '')
    wiki_aus['Pregnancy category'] = wiki_aus['Pregnancy category'].str.replace('Australian pregnancy category ', '')
    wiki_us = wiki_us.groupby('drugBank_id')['Pregnancy category'].apply(''.join).rename('US pregnancy category')
    wiki_aus = wiki_aus.groupby('drugBank_id')['Pregnancy category'].apply(''.join).rename(
        'Australian pregnancy category')

    approved = get_data_for_output_table()

    preds = preds.join(who_tags, how='left').join(smc_tags, how='left').join(eltonsy_tags, how='left').join(
        fda_code_tags, how='left').join(wiki_us, how='left').join(wiki_aus, how='left').join(approved,how='left')
    preds.index.name = "DrugBank ID"
    preds = preds.reset_index()
    preds = preds[['DrugBank ID', 'Drug name', 'Polifka et al', 'Zerifin TIS', 'Eltonsy et al', 'FDA category',
                   'US pregnancy category', 'Australian pregnancy category', 'Risk score','Approved','Withdrawn']]
    # preds.to_csv(os.path.join('output','SHAP_original','preds.csv'),index=False)
    preds.to_json(os.path.join('output','SHAP' +'_' + eval_label, 'preds.json'), orient='records')


if __name__=="__main__":
    np.random.seed(0)
    remove_disputed = True

    eval_label = 'unlabled'#'unlabled' 'test' 'test_png'
    if eval_label=='test':
        read_smc_train = False
        plot_ext='pdf'
    elif eval_label =='unlabled':
        read_smc_train = True
        plot_ext = 'png'
    else:
        read_smc_train = False
        plot_ext='png'


    y_train, X_train, X_test, X_unlabled,  y_test, modalities_df = get_data(remove_disputed, red_smc_train=read_smc_train)

    y_train = y_train.astype(bool)
    y_test = y_test.astype(bool)
    X_train = X_train.astype(int)
    X_unlabled = X_unlabled.astype(int)
    X_test = X_test.astype(int)

    X_train = X_train.loc[:, modalities_df.loc[modalities_df.modality == 'Category', 'feature']]
    X_unlabled = X_unlabled.loc[:, modalities_df.loc[modalities_df.modality == 'Category', 'feature']]
    X_test = X_test.loc[:, modalities_df.loc[modalities_df.modality == 'Category', 'feature']]

    # X_train_clustering = X_train.loc[:, modalities_df.loc[modalities_df.modality == 'Clusters', 'feature']]
    # X_test_clustering = X_test.loc[:, modalities_df.loc[modalities_df.modality == 'Clusters', 'feature']]
    # X_unlabeled_clustering = X_unlabled.loc[:, modalities_df.loc[modalities_df.modality == 'Clusters', 'feature']]

    if eval_label=='test' or eval_label == 'test_png':
        data = X_test
    else:
        data = X_unlabled

    data.columns = [x.replace('Category: ', '') for x in data]
    run_shap(data_for_shap=data, eval_label=eval_label, force_plot_file_type=plot_ext,num_reps=30)

