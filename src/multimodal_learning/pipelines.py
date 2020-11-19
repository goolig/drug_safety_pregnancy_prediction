import math

import numpy as np
import scipy
from group_lasso import GroupLasso
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import pandas as pd

class filter_cols():
    def __init__(self,cols):
        self.cols=cols
        self.n_features_in_ = len(cols)
    def fit(self,X,y=None):
        pass
    def transform(self,X):
        ans = X[self.cols]
        return ans.values
    def fit_transform(self,X,y=None):
        self.fit(X,y)
        return self.transform(X)

class propgate_lables_predictor_raw():
    def __init__(self,X_unlabled,predictor):
        self.X_unlabled=X_unlabled
        self.predictor = predictor


    def fit(self,X,y):
        unlabled =self.X_unlabled.head(500)
        self.predictor.fit(X,y)


        self.predictor.fit(unlabled,self.predictor.predict(unlabled))

    def predict(self,X):
        return self.predictor.predict(X)

    def predict_proba(self,X):
        return self.predictor.predict_proba(X)


class propgate_lables_predictor():
    def __init__(self,X_unlabled,predictor):
        self.X_unlabled=X_unlabled
        self.prop_model = LabelSpreading(kernel='rbf',gamma=0.1,max_iter=1000,tol=0.001,n_jobs=-1,alpha=0.2)
        self.predictor = predictor


    def fit(self,X,y):
        unlabled =self.X_unlabled#.head(500)
        new_x, new_y_pre = pd.concat([pd.DataFrame(X), pd.DataFrame(unlabled.values)]), pd.concat([y, pd.DataFrame([-1] * len(unlabled))])
        scale = StandardScaler()
        self.prop_model.fit(scale.fit_transform(new_x), np.array(new_y_pre).ravel())

        new_y_post = self.prop_model.predict(scale.transform(new_x))
        pred_entropies = pd.Series(scipy.stats.entropy(self.prop_model.label_distributions_.T))


        X_final = new_x.reset_index(drop=True)
        pred_entropies.index = X_final.index


        #pred_entropies[new_y_pre==-1] = 0 #they are known, making sure they are in
        y_final = pd.concat([pd.Series(y), pd.Series(new_y_post[len(y):])])
        y_final.index = X_final.index

        cond = (~pred_entropies.isna()) & (pred_entropies < pred_entropies.iloc[len(y):].mean())

        X_final = X_final.loc[cond,:]
        y_final = y_final[cond]

        print(len(X),'final amount of instances:',len(X_final))
        self.predictor.fit(X_final,np.array(y_final).ravel())

    def predict(self,X):
        return self.predictor.predict(X)

    def predict_proba(self,X):
        return self.predictor.predict_proba(X)


class greedy_group_elimination():
    def __init__(self, possible_modalities, modalities_df,clf):
        self.possible_modalities=possible_modalities
        self.modalities=modalities_df
        self.clf=clf

    def fit(self,X,y):
        result = list(set(self.possible_modalities))
        pipeline = Pipeline(
            [('select', filter_cols(self.modalities.loc[self.modalities.modality.isin(result), 'feature'])),
             ('pred', self.clf)])
        scores = cross_val_score(pipeline, X, y, cv=10, n_jobs=-1, scoring='roc_auc')
        best_score = np.array(scores).mean()
        while True:
            best_mod = None
            for current_modlaity in set(result):

                pipeline = Pipeline([('select', filter_cols(
                    self.modalities.loc[self.modalities.modality.isin(list(set(result) - {current_modlaity})), 'feature'])),
                                     ('pred', self.clf)])
                auc = cross_val_score(pipeline, X, y, cv=10, n_jobs=-1, scoring='roc_auc')
                auc=np.array(auc).mean()
                if auc > best_score:
                    best_score = auc
                    best_mod = current_modlaity
            if best_mod != None:
                result.remove(best_mod)
            else:
                break
        self.result = result

    def transform(self,X):
        assert self.result is not None, 'Not fitted yet'
        return X[self.modalities.loc[self.modalities.modality.isin(self.result), 'feature']]

    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)

class lasso_transformer:
    def __init__(self,r,groups_ids):
        self.lasso = GroupLasso(    groups=groups_ids,    group_reg=r,     l1_reg=r,    n_iter = 1000,
                                       scale_reg="None",       supress_warning=True,    tol=1e-04,
    )
    def fit(self,X,y):
        self.lasso.fit(X,y.values.reshape(-1, 1))

    def transform(self,X,y=None):
        return self.lasso.transform(X)


    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X,y)


class greedy_group_selection():
    def __init__(self, possible_modalities, modalities_df,clf):
        self.possible_modalities=possible_modalities
        self.clf=clf
        self.modalities=modalities_df
    def fit(self,X,y):
        result = []
        best_score = 0.5
        while True:
            best_mod = None
            for current_modlaity in set(self.possible_modalities) - set(result):
                pipeline = Pipeline([('select', filter_cols(
                    self.modalities.loc[self.modalities.modality.isin(result + [current_modlaity]), 'feature'])),
                                     ('pred', self.clf)])
                scores = cross_val_score(pipeline,X,y,cv=10,n_jobs=-1,scoring='roc_auc')
                auc = np.array(scores).mean()
                if auc > best_score:
                    best_score = auc
                    best_mod = current_modlaity
            if best_mod != None:
                result.append(best_mod)
            else:
                break
        self.result=result

    def transform(self,X):
        assert self.result is not None, 'Not fitted yet'
        return X[self.modalities.loc[self.modalities.modality.isin(self.result), 'feature']]

    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)