import operator
import os
from collections import Counter

import pandas as pd
import keras
from keras import layers, regularizers
from keras.layers import multiply, Multiply, Concatenate
from pandas import HDFStore
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def getPCA(X_unlabled,features,modality,k=100):
    # import prince
    # pca = prince.MCA(
    #     n_components=1000,
    #  #n_iter=3,
    #  copy=True,
    #  check_input=True,
    #  engine='auto',
    #  random_state=42 )
    # X_unlabled_pca = pca.fit_transform(
    #     VarianceThreshold().fit_transform(
    #         X_unlabled[modalities.loc[modalities.modality.isin([modality]), 'feature']]
    #     )
    # )

    #pca = corex.Corex(n_hidden=50, dim_hidden=10, marginal_description='discrete', smooth_marginals=False,max_iter=100,n_repeat=1,n_cpu=None,verbose=True) # n_hidden = dim of output. dim_hiden=card of output#too slow
    # Define the number of hidden factors to use (n_hidden=2).
    # And each latent factor is binary (dim_hidden=2)
    # marginal_description can be 'discrete' or 'gaussian' if your data is continuous
    # smooth_marginals = True turns on Bayesian smoothing
    #layer1.fit(X)  # Fit on data.
    k = round(len(features) ** (0.5) )
    pca = PCA(n_components=k)
    X_unlabled_pca = pca.fit_transform(
       StandardScaler().fit_transform(X_unlabled[features]))

    X_unlabled_pca = pd.DataFrame(X_unlabled_pca)
    X_unlabled_pca.index = X_unlabled.index
    X_unlabled_pca.columns = [str(modality)+'_PCA' +"_" + str(x) for x in range(pca.n_components)]
    return X_unlabled_pca

def getselectK(X_unlabled, modality,t=0.01):
    select = VarianceThreshold(t)
    data = X_unlabled[modalities.loc[modalities.modality.isin([modality]), 'feature']]
    select.fit(data)
    data = data.iloc[:,select.get_support()]
    data.columns = [x+ ' select' for x in data.columns]
    return data



def extract_text_features(X_unlabled):
    output = X_unlabled.copy()
    output.columns = [c.split(': ')[1] if ': ' in c else c for c in output.columns]
    a = pd.melt(output.reset_index(), id_vars=['drugBank_id'])
    a.value = a.value.astype(bool)
    a = a[a.value == True]
    a = a.groupby('drugBank_id').variable.apply(lambda x: "%s " % ' '.join(x))

    count_vect = CountVectorizer(binary=True)
    a = a.fillna('')
    word_counts = count_vect.fit_transform(a)
    text_features = ['Mention: ' + x for x in count_vect.get_feature_names()]
    text_features = pd.DataFrame(word_counts.toarray(), columns=text_features, index=a.index)
    a = pd.DataFrame(index=X_unlabled.index).join(text_features, how='left').fillna('')
    # for c in text_features:
    #     modalities = modalities.append({'modality': 'text_processed', 'feature': str(c)}, ignore_index=True)
    return a#, modalities

def get_col_clusters(X_unlabled,number_of_clusters):
    #X_unlabled.columns = [x.replace(' description','').lower().replace('atc level ','ATC Level ') for x in X_unlabled.columns] #remove the description of the ATC codes
    s = pd.Series(X_unlabled.columns, index=X_unlabled.columns)
    s = s[~s.str.contains('Number of')]
    newvals = [c.split(': ')[1] if ': ' in c else c for c in s.values]
    s = pd.Series(newvals, index=s.index)
    count_vect = CountVectorizer(binary=True) #2-5 char_wb improves cat
    #count_vect = TfidfVectorizer(ngram_range=(2, 5), analyzer='char_wb',max_features=1000)  # 2-5 char_wb improves cat
    word_counts = count_vect.fit_transform(s)
    word_counts = word_counts
    text_features = count_vect.get_feature_names()
    text_features = pd.DataFrame(word_counts.toarray(), columns=text_features, index=s.index)
    ans = None
    print('features,words:',text_features.values.shape)
    print(list(text_features.columns)[:1000])
    dist_mat = pairwise_distances(text_features.values, metric='jaccard', n_jobs=1) #jaccard
    print('done sim matrix')
    #single; 9000: 0.8798, 7500:0.877
    #average selected mods: 2500 not good. 4000:0.882\0.896. 3500 0.88\0.893
    #average: 3500: best. 3750: small reduction. answer is in between
    kmeans = AgglomerativeClustering(n_clusters=number_of_clusters,linkage='average',affinity='precomputed')#affinity=sklearn.metrics.jaccard_score ,affinity='cosine'
    #kmeans.fit(text_features)
    clusters = pd.DataFrame(kmeans.fit_predict(dist_mat),columns=['cluster'],index=text_features.index)
    for g in clusters.groupby('cluster').groups:
        g= clusters.groupby('cluster').groups[g]
        col_name='Cluster '+str(number_of_clusters)+ ' :  ' + '; '.join([str(gr) for gr in g])
        g_col = X_unlabled[g].sum(axis=1).astype(bool)
        if ans is None:
            g_col.name = col_name
            ans = pd.DataFrame(g_col)
        else:
            ans[col_name] = g_col
    print('done clustering',number_of_clusters)

    print('done clustering')
    return ans


def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    c = Counter()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print(topic.argsort()[:-n_top_words - 1:-1])
        c.update(topic.argsort()[:-n_top_words - 1:1])
    print(c.most_common())

def get_lda_cluster(X_unlabled,num_clusters):
    #X_unlabled.columns = [x.replace(' description','').lower().replace('atc level ','ATC Level ') for x in X_unlabled.columns] #remove the description of the ATC codes
    columns_data = pd.Series(X_unlabled.columns, index=X_unlabled.columns)
    columns_data = columns_data[~columns_data.str.contains('Number of')]
    newvals = [c.split(': ')[1] if ': ' in c else c for c in columns_data.values]
    columns_data = pd.Series(newvals, index=columns_data.index)
     # Create a corpus from a list of texts

    count_vect = CountVectorizer(stop_words='english')
    word_counts = count_vect.fit_transform(columns_data)
    #text_features = count_vect.get_feature_names()
    #text_features = pd.DataFrame(word_counts.toarray(), columns=text_features, index=columns_data.index)
    # Create and fit the LDA model
    from sklearn.decomposition import LatentDirichletAllocation as LDA
    #import lda
    #lda = lda.LDA(n_topics=num_clusters,random_state=0,n_iter=2000,eta=0.001,alpha =0.01) #mean_change_tol=1e-8,max_iter=100, ,learning_method='online',batch_size=2500
    lda = LDA(n_components=num_clusters, random_state=0,n_jobs=1,learning_method='online',verbose=1,max_iter=50,evaluate_every=5)  # mean_change_tol=1e-8,max_iter=100, ,

    print(word_counts.shape)

    predicted_cluster = lda.fit_transform(word_counts).argmax(axis=1)
    #print_topics(lda, count_vect, 5)

    print(predicted_cluster.shape)
    predicted_cluster = pd.Series(predicted_cluster, index=columns_data.index)

    # predicted_cluster = lda.fit_transform(word_counts).argsort(axis=1)[:,:num_cluster_members]
    # print_topics(lda,count_vect,5)
    # print(predicted_cluster.shape)
    # predicted_cluster = pd.DataFrame(predicted_cluster, index=columns_data.index,columns=[str(x) for x in range(num_cluster_members)])
    # predicted_cluster = pd.concat([predicted_cluster[str(x)] for x in range(num_cluster_members)])
    # print(predicted_cluster.shape)
    ans=None
    print('predicted num clusters',len(predicted_cluster.unique()))
    for curr_cluster in range(num_clusters):
        curr_cols = predicted_cluster[predicted_cluster==curr_cluster]
        if len(curr_cols) >0:
            #print('curr cluster size',len(curr_cols))
            col_name = 'Cluster ('+str(num_clusters)+') '+str(curr_cluster)+': ' + '; '.join([str(col) for col in curr_cols.index])
            #print(col_name)
            g_col = X_unlabled[curr_cols.index].sum(axis=1).astype(bool)
            if ans is None:
                g_col.name = col_name
                ans = pd.DataFrame(g_col)
            else:
                ans[col_name] = g_col
    print('final number of clusters',ans.shape)

    print('done clustering')
    return ans

def convert_to_one_hot(X_unlabled, modalities, modality):
    cols = modalities.loc[modalities.modality.isin([modality]), 'feature']
    X_unlabled =  pd.get_dummies(X_unlabled,columns=cols,prefix=cols,prefix_sep=': ')
    modalities=modalities[modalities.modality!=modality]
    for c in X_unlabled.columns:
        if c.split(': ')[0] in cols.values:
            modalities = modalities.append({'modality': modality, 'feature': c}, ignore_index=True)
    return X_unlabled,modalities


def supervised_dim_reduction(X_train, y_train, X_test=None, validation=0.0):
    if X_test is None:
        X_test =X_train
    inputs = keras.Input(shape=(len(X_train.columns),))
    reg = None#regularizers.l1_l2(l1=1e-5, l2=1e-5)
    dense_out = layers.Dense(len(y_train.columns), activation="sigmoid", kernel_regularizer=reg)#for some weird reason softmax works better

    # mult = Multiply()([inputs, inputs])
    # layer = Concatenate()([mult,inputs])
    layer= inputs
    layers_list = []
    for i in range(10):
        layer = layers.Dropout(0.0)(layer)
        layer = layers.Dense(300, activation="relu",kernel_regularizer=reg)(layer)
        layers_list.append(layer)

    outputs = dense_out(layer)
    model = keras.Model(inputs=inputs, outputs=outputs, name="dim_reduction")
    model.compile(
        loss='binary_crossentropy',
        optimizer='Adam',
        metrics=["accuracy"],
    )
    model.fit(X_train, y_train.astype(int), shuffle=True, validation_split=validation, epochs=25, verbose=2)#epochs=25

    model_out = keras.Model(inputs=inputs, outputs=layers_list[0], name="dim_reduction")
    model_out.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    ans = model_out.predict(X_test)
    ans = pd.DataFrame(ans)
    return ans#,model.predict(X_test)

if __name__ =='__main__':
    os.chdir('..\\..')
    store = HDFStore('output\data\modalities_dict.h5')
    X_unlabled = store['df']
    modalities = store['modalities']
    cols_x = modalities[modalities.modality.isin(['mol2vec'])].feature
    cols_y = modalities[modalities.modality.isin(['Category'])].feature
    X = X_unlabled[cols_x]
    y = X_unlabled[cols_y].drop('Number of Category', axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.33, random_state=42)
    ans = supervised_dim_reduction(X,y,validation=0.2)
    # 1 - y.sum().sum() / y.count().sum()

    # print(np.sum(preds.round() == y_test, axis=0).sum() / y_test.count().sum())