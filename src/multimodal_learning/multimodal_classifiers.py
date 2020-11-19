import random
from pyDOE import pbdesign
import pandas
from group_lasso import GroupLasso
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, MaxAbsScaler

from src.multimodal_learning.pipelines import filter_cols, greedy_group_selection, greedy_group_elimination, \
    propgate_lables_predictor, propgate_lables_predictor_raw, lasso_transformer


class multimodal_classifiers():
    
    def __init__(self,modalities_df,columns):
        self.modalities_df = modalities_df
        self.columns = columns
    
    def get_mod_stacking(self,given_modalities,clf):
        estimators=[]
        for m in given_modalities:
            text_pipe = Pipeline(
                    [('select', filter_cols(self.modalities_df.loc[self.modalities_df.modality.isin([m]), 'feature'])),
                     ('pred', clf)])
            estimators.append((m,text_pipe))
        clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(),n_jobs=-1)
        return clf
    
    
    def get_mod_voting(self,given_modalities,clf):
        estimators=[]
        for m in given_modalities:
            text_pipe = Pipeline(
                    [('select', filter_cols(self.modalities_df.loc[self.modalities_df.modality.isin([m]), 'feature'])),
                     ('pred', clf)])
            estimators.append((m,text_pipe))
        clf = VotingClassifier(estimators=estimators, voting='soft',n_jobs=-1)
        return clf
    
    
    def get_orthogonal_stacking_voting(self,given_modalities,clf):
        estimators = []
        ort = pbdesign(len(given_modalities))
        #print(ort)
        for id,trial in enumerate(ort):
            mods = []
            for idx,val in enumerate(trial):
                if val==1:
                    mods.append(given_modalities[idx])
            if len(mods)>0:
                text_pipe = Pipeline(
                    [('select', filter_cols(self.modalities_df.loc[self.modalities_df.modality.isin(mods), 'feature'])),
                     ('pred', clf)])
                estimators.append((str(id)+str(trial), text_pipe))
        clf_stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), n_jobs=-1)
        clf_voting = VotingClassifier(estimators=estimators,voting='soft')
        return clf_voting,clf_stacking
    
    
    def get_orthogonal_large_voting_stacking(self,given_modalities):
        given_modalities = list(given_modalities)
        estimators = []
        for i in range(3):
            random.shuffle(given_modalities)
            ort = pbdesign(len(given_modalities))
            #print(ort)
            for id,trial in enumerate(ort):
                mods = []
                for idx,val in enumerate(trial):
                    if val==1:
                        mods.append(given_modalities[idx])
                if len(mods) > 0:
                    multinomal = make_pipeline(  # improves
                        Normalizer(norm="max"),
                        MultinomialNB(alpha=0.01, fit_prior=True)
                    )
                    logistic = make_pipeline(  # improve
                        MaxAbsScaler(),
                        LogisticRegression(C=10.0, dual=False, penalty="l2")
                    )
                    for cl in [ExtraTreesClassifier(n_jobs=-1), multinomal, logistic]:
                        text_pipe = Pipeline(
                            [('select',
                              filter_cols(self.modalities_df.loc[self.modalities_df.modality.isin(mods), 'feature'])),
                             ('pred', cl)])
                        estimators.append((str(id) + str(trial) + str(cl) + str(i), text_pipe))
        clf_stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), n_jobs=-1)
        clf_voting = VotingClassifier(estimators=estimators,voting='soft')
        return clf_voting,clf_stacking
    
    
    def get_modalities_base_clf(self, modalities_to_experemint, clf):
            return  Pipeline(
                [('select', filter_cols(self.modalities_df.loc[self.modalities_df.modality.isin(modalities_to_experemint), 'feature'])),
                 ('pred', clf)])

    def get_modalities_extra_trees_propagation(self,modalities_to_experemint,X_unlabled):
        return  Pipeline( [('select', filter_cols(self.modalities_df.loc[self.modalities_df.modality.isin(modalities_to_experemint), 'feature'])),
                           ('propagate',propgate_lables_predictor_raw(X_unlabled,ExtraTreesClassifier(n_jobs=-1))),
                           ])

    
    def get_lasso_classifier(self,modalities_to_experemint,clf,r,X):

        groups = pandas.DataFrame(X.columns, index=X.columns)
        mods = self.modalities_df[self.modalities_df.modality.isin(modalities_to_experemint)]
        groups_ids = groups.join(mods.set_index('feature')).modality.astype('category').cat.codes.values
        return Pipeline(
            [ ('select', filter_cols(self.modalities_df.loc[self.modalities_df.modality.isin(modalities_to_experemint), 'feature'])),
              ('scale',StandardScaler())
                ,('dummy_transform', lasso_transformer(r, groups_ids))
             ,('pred', clf)  ])
    
    
    def get_greedy_addition_group_selection_predictions(self,all_mods,clf):
        return Pipeline(
            [('select', greedy_group_selection(all_mods, self.modalities_df,clf)),
             ('pred', clf)])
    
    
    def add_greedy_group_elimination_predictions(self,all_mods,clf):
        return Pipeline(
            [('select', greedy_group_elimination(all_mods, self.modalities_df,clf)),
             ('pred', ExtraTreesClassifier(n_jobs=-1))])