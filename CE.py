# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import numpy as np


## use gridsearch to find best parameter ##
from sklearn.model_selection import GridSearchCV
def findBestEstimator (x_train,y_train,parameters,esm=RandomForestClassifier()):
    clf = GridSearchCV(esm,parameters,scoring='recall_macro')
    #clf = GridSearchCV(esm,parameters,scoring='recall_micro')
    clf.fit(x_train,y_train)
    return clf.best_estimator_

## use 10-fold cross validation to compare the performance between different algorithm ##
def individual_model(estimator,parameter):
    recall=np.zeros(2)
    precision=np.zeros(2)
    score=0
    f1=np.zeros(2)
    Auc=0
    for s in range(10):
        rdf=findBestEstimator (X_train[s],y_train[s],parameters,esm=estimator)
        rdf.fit(X_train[s].values,y_train[s].values)
        scr=rdf.score(X_test[s].values,y_test[s].values)
        score+=scr/10
        pred=rdf.predict(X_test[s])
        precision+=precision_score(y_test[s], pred, average=None,zero_division=0)/10
        recall+=recall_score(y_test[s], pred, average=None,zero_division=0)/10
        f1+=f1_score(y_test[s], pred, average=None)/10
        fpr, tpr, thresholds = metrics.roc_curve(y_test[s], pred, pos_label=1)
        Auc+=metrics.auc(fpr, tpr)/10
    print('score:',score,'\n','precision:',precision,'\n','recall:',recall,'\n','f1:',f1,'\n','Auc',Auc)
    
### bagging code ###
from sklearn.ensemble import VotingClassifier,StackingClassifier
from sklearn.decomposition import TruncatedSVD
def bagging(df_train,df_label,df_test,df_test_label,parameters,sample_n=50,base_estimator=RandomForestClassifier()):
    labdic={}
    res=[]
    arr=np.zeros(2)
    scor=0
    df_train_label=df_label.values
    '''
    ### Dimensionlity Reduction
    dimrd_TSVD=TruncatedSVD(n_components=30)
    dimrd_TSVD.fit(df_train)
    df_train=pd.DataFrame(dimrd_TSVD.transform(df_train))
    df_test=pd.DataFrame(dimrd_TSVD.transform(df_test))
    ###
    '''
    for lab in set(df_train_label):
        labdic[lab]=[]
    N=np.shape(df_train)[0]
    for i in range(N):
        labdic[df_train_label[i]].append(i)
    for turn in range(sample_n):
        X_resampled=[]
        for lab in set(df_train_label):
            X_resampled+=list(np.random.choice(labdic[lab],size=300)) ### replace =False
        #print(X_resampled)
        bbc=findBestEstimator (df_train.iloc[X_resampled],df_label.iloc[X_resampled],parameters,esm=base_estimator)
        #print(bbc)
        bbc.fit(df_train.iloc[X_resampled],df_label.iloc[X_resampled])
        #print(np.array(list(accEachClass(bbc.predict(df_train.iloc[X_resampled]),df_label.iloc[X_resampled]).values())))
        res.append(bbc.predict(df_test))
    pdt={}
    for i in range(np.shape(res)[1]):
        a=[]
        for m in range(sample_n):
            a.append(res[m][i])
        pdt[i]=pd.Series(a).value_counts().index[0]
    #vc=VotingClassifier(estimators=res,voting='hard')
    #scor+=vc.score(df_test,df_test_label)
    #return pdt,scor
    res_pred=list(pdt.values())
    return res_pred


### this function used to test performance by setting estimator and train/test data ###
def direct_estimator_i(rdf,target_x,target_y,s):
    Auc=0
    precision=np.zeros(2)
    recall=np.zeros(2)
    f1=np.zeros(2)
    score=0 
    rdf.fit(X_train[s].values,y_train[s].values)
    scr=rdf.score(target_x[s].values,target_y[s].values)
    pred=rdf.predict(target_x[s])
    score+=scr
    precision+=precision_score(target_y[s], pred, average=None,zero_division=0)
    recall+=recall_score(target_y[s], pred, average=None,zero_division=0)
    f1+=f1_score(target_y[s], pred, average=None)
    fpr, tpr, thresholds = metrics.roc_curve(target_y[s], pred, pos_label=1)
    Auc+=metrics.auc(fpr, tpr)
    print('score:',score,'\n','precision:',precision,'\n','recall:',recall,'\n','f1:',f1,'\n','auc:',Auc)


''' Data Process '''
### import data ###
import pandas as pd
df=pd.read_csv('deidentified_SLE_flare_medical_record_review_9-21-20.csv')

### preprocess data ###
df_copy=df.iloc[:,5:87]
df_copy.set_index(df['patient ID'],inplace=True)
for patient in set(df_copy.index):
    df_copy.loc[df_copy.index==patient]=df_copy[df_copy.index==patient].fillna(method='ffill')
df_copy.to_excel('prep_copy3.xlsx')
for col in df_copy.columns:
    df[col]=df_copy[col].values 
df.dropna(axis=0,subset=['visit #'],inplace=True)
df.set_index('patient ID',inplace=True)
df

### check the percentage of missing values ###
loss_percent=[]
N=len(df)
for col in df.columns:
    nn=df[col].count()
    misv=(N-nn)/N
    loss_percent.append(misv)
df.loc['missing_percentage']=loss_percent
df


### drop columns with missing value ###
df_clean=df.dropna(axis=1).drop('missing_percentage')
df.drop('missing_percentage',inplace=True)
df_clean['sle_immunosuppressant']=df['sle_immunosuppressant'].values
df_clean['iv_steroid']=df['iv_steroid'].values
df_clean['glucocorticoids']=df['glucocorticoids'].values
df_clean.fillna(value=0,inplace=True)
df_clean


### drop rssfi =5 ###
df1=df_clean[df_clean.rssfi!=5]
### drop date ###
date_feature=['visit_date','firstrheumvisit','firststudyrheumvisit','hcq_start_date','hcq_dose_change_date','redcap_repeat_instrument']
for f in date_feature:
    if f in df1.columns:
        df1=df1.drop(f,axis=1)

### seperate features and label ###
y=df1.rssfi
X=df1.drop(['rssfi','visit #','demographics_complete'],axis=1)
X

### Transfer to binary ###
X.index=range(1541)
y_b=y.replace([1.0,2.0,3.0],0)
y_b=y_b.replace(4.0,1)
y_b.index=range(1541)


### check imbalance ###
y_b.value_counts()


''' Individual Model '''
### This method is more focusing on the overall performance but not take imbalance into accounts.###

### Cross validation / StratifiedShuffleSplit ###
X_train,X_test,y_train,y_test=[],[],[],[]
from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=10,shuffle=False, random_state=None)
for train_index, test_index in skf.split(X, y_b):
    X_train.append(X.iloc[train_index]) 
    X_test.append(X.iloc[test_index]) 
    y_train.append(y_b.iloc[train_index])
    y_test.append(y_b.iloc[test_index]) 


### Randomforest ###
parameters={'max_depth':[3,5,8,10,20,60],'min_samples_leaf':[2,3,4],'n_estimators':[10,20,50,100]}
individual_model(RandomForestClassifier(),parameter=parameters)

### Decision Tree recall_macro ###
parameters={'max_depth':[3,5,8,10,20,60],'min_samples_leaf':[2,3,4]}
individual_model(DecisionTreeClassifier(),parameter=parameters)


### AdaBoost recall_macro ###
parameters={'n_estimators':[10,20,50,100]}
individual_model(AdaBoostClassifier(),parameter=parameters)


### Navie Bayes recall_macro ###
parameters={}
individual_model(BernoulliNB(fit_prior=0),parameter=parameters)

### Logistic Regression recall_macro ###
parameters={'C':[0.01,0.05,0.1,0.5,1,],'multi_class':['ovr','multinomial']}
individual_model(LogisticRegression(solver='saga',max_iter=5000),parameter=parameters)


''' Class weight '''
### This method solves the imbalance issue by manipulating the class weight assigned to different class ###

### Randomforest ###
parameters={'max_depth':[3,5,8,10,20,60],'min_samples_leaf':[2,3,4],'n_estimators':[10,20,50,100]}
individual_model(RandomForestClassifier(class_weight={0:6,1:1}),parameter=parameters)   

### Navie Bayes ###
parameters={}
individual_model(BernoulliNB(fit_prior=0,class_prior=[3,1]),parameter=parameters)

### Logistic Regression  ###
parameters={'C':[0.01,0.05,0.1,0.5,1,],'multi_class':['ovr','multinomial']}
individual_model(LogisticRegression(solver='saga',max_iter=1000,class_weight={0:7,1:1}),parameter=parameters)

### Decision Tree  ###
parameters={'max_depth':[3,5,8,10,20,60],'min_samples_leaf':[2,3,4]}
individual_model(DecisionTreeClassifier(class_weight={0:6,1:1}),parameter=parameters)


### AdaBoost  ###
### AdaBoost perform better using bagging ###

def Bagging_model(estimator,parameter):
    recall=np.zeros(2)
    precision=np.zeros(2)
    f1=np.zeros(2)
    Auc=0
    for s in range(10):
        pred=bagging(X_train[s],y_train[s],X_test[s],y_test[s],parameter,sample_n=50,base_estimator=estimator)
        precision+=precision_score(y_test[s], pred, average=None,zero_division=0)/10
        recall+=recall_score(y_test[s], pred, average=None,zero_division=0)/10
        f1+=f1_score(y_test[s], pred, average=None)/10
        fpr, tpr, thresholds = metrics.roc_curve(y_test[s], pred, pos_label=1)
        Auc+=metrics.auc(fpr, tpr)/10
    print('precision:',precision,'\n','recall:',recall,'\n','f1:',f1,'\n','Auc',Auc)

parameters={'n_estimators':[10,20,50,100]}
Bagging_model(AdaBoostClassifier(),parameter=parameters)


''' Calibrated Ensemble '''

### sample 10% data out ###
from random import sample
list1=sample(list(X.index),150)
X_sampled=X.iloc[list1]
y_sampled=y_b.iloc[list1]

### set X_t,y_t_b to be train data ###
X_t=X.drop(list1)
y_t_b=y_b.drop(list1)

### cross validation / StratifiedShuffleSplit ###
X_train,X_test,y_train,y_test=[],[],[],[]
from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=10,shuffle=False, random_state=None)
for train_index, test_index in skf.split(X_t, y_t_b):
    X_train.append(X_t.iloc[train_index]) 
    X_test.append(X_t.iloc[test_index]) 
    y_train.append(y_t_b.iloc[train_index])
    y_test.append(y_t_b.iloc[test_index]) 

### Iteration1 ###
rf,nb,lr=[],[],[]
chos={}

for s in range(10):
#Randomeforest#
    parameters={'max_depth':[3,5,8],'min_samples_leaf':[2,3,4],'n_estimators':[3,5,8,10]}
    rdf=findBestEstimator (X_train[s],y_train[s],parameters,esm=RandomForestClassifier(class_weight={0:6,1:1}))
    rdf.fit(X_train[s].values,y_train[s].values)
    prd_proba=rdf.predict_proba(X_test[s])
    res_rf0=[]
    for i in range(len(y_test[s].values)):
        k=int(y_test[s].values[i])
        res_rf0.append(prd_proba[i][k])

#Navie Bayes#
    parameters={}
    rdf=findBestEstimator (X_train[s],y_train[s],parameters,esm=BernoulliNB(fit_prior=0,class_prior=[3,1]))
    rdf.fit(X_train[s].values,y_train[s].values)
    prd_proba=rdf.predict_proba(X_test[s])
    res_nb0=[]
    for i in range(len(y_test[s].values)):
        k=int(y_test[s].values[i])
        res_nb0.append(prd_proba[i][k])

#Logistic Regression#
    parameters={'C':[0.001,0.01,0.05,0.1,]}
    rdf=findBestEstimator (X_train[s],y_train[s],parameters,esm=LogisticRegression(max_iter=5000,class_weight={0:7,1:1}))
    rdf.fit(X_train[s].values,y_train[s].values)
    prd_proba=rdf.predict_proba(X_test[s])
    res_lr0=[]
    for i in range(len(y_test[s].values)):
        k=int(y_test[s].values[i])
        res_lr0.append(prd_proba[i][k])

    res_rf0=dict(zip(y_test[s].index,res_rf0))
    res_nb0=dict(zip(y_test[s].index,res_nb0))
    res_lr0=dict(zip(y_test[s].index,res_lr0))

#Transfer instances 
    for i in y_test[s].index:
        k=0
        if res_rf0[i]<res_nb0[i]:
            k=1
        if res_nb0[i]<res_lr0[i]:
            if res_lr0[i]>res_rf0[i]:
                k=2
        if k==0:
            rf.append(i)
            chos[i]=res_rf0[i]                   
        if k==1:
            nb.append(i)
            chos[i]=(res_nb0[i])            
        if k==2:
            lr.append(i)
            chos[i]=(res_lr0[i])  
chos                          

### Generalized Interations ###
ssb=0
turn=0
rf_prob_final,nb_prob_final,lr_prob_final=0,0,0
for t in range(100):    
    rf,nb,lr=set(rf),set(nb),set(lr)
    rf_y=y_b.loc[rf]
    nb_y=y_b.loc[nb]
    lr_y=y_b.loc[lr]
    rf_cluster=X.loc[rf]
    nb_cluster=X.loc[nb]
    lr_cluster=X.loc[lr]
    
    X_train_1,X_test_1,y_train_1,y_test_1=[],[],[],[]
    skf=StratifiedKFold(n_splits=10,shuffle=False, random_state=None)
    for train_index, test_index in skf.split(rf_cluster, rf_y):
        X_train_1.append(rf_cluster.iloc[train_index]) 
        X_test_1.append(rf_cluster.iloc[test_index]) 
        y_train_1.append(rf_y.iloc[train_index])
        y_test_1.append(rf_y.iloc[test_index]) 
    
    rdf=RandomForestClassifier(warm_start=True,max_depth=5,class_weight={0:6,1:1},min_samples_leaf=3)
    Auc=0
    precision=np.zeros(2)
    recall=np.zeros(2)
    f1=np.zeros(2)
    score=0
    for s in range(10):
        rdf.fit(X_train_1[s].values,y_train_1[s].values)
        scr=rdf.score(X_test_1[s].values,y_test_1[s].values)
        pred=rdf.predict(X_test_1[s])
        score+=scr
        precision+=precision_score(y_test_1[s], pred, average=None,zero_division=0)/10
        recall+=recall_score(y_test_1[s], pred, average=None,zero_division=0)/10
        f1+=f1_score(y_test_1[s], pred, average=None)/10
        fpr, tpr, thresholds = metrics.roc_curve(y_test_1[s], pred, pos_label=1)
        Auc+=metrics.auc(fpr, tpr)/10
    print('score:',score/10,'\n','precision:',precision,'\n','recall:',recall,'\n','f1:',f1,'\n','auc:',Auc)

    rdf=RandomForestClassifier(warm_start=True,max_depth=5,class_weight={0:6,1:1},min_samples_leaf=3)
    rf_prob_rf={}
    for s in range(10):
        rdf.fit(X_train_1[s].values,y_train_1[s].values)
        p=rdf.predict_proba(X_test_1[s])
        for n in range(len(X_test_1[s])):
            k=int(y_test_1[s].values[n])
            rf_prob_rf[X_test_1[s].index[n]]=p[n][k]

    rdf=BernoulliNB(fit_prior=1,class_prior=[3,1])
    rf_prob_nb={}
    rdf.fit(nb_cluster,nb_y)
    for s in range(10):
        p=rdf.predict_proba(X_test_1[s])
        for n in range(len(X_test_1[s])):
            k=int(y_test_1[s].values[n])
            rf_prob_nb[X_test_1[s].index[n]]=p[n][k]

    rdf=LogisticRegression(warm_start=True,random_state=0,C=0.2,class_weight={0:4,1:1})
    rf_prob_lr={}
    rdf.fit(lr_cluster,lr_y)
    for s in range(10):
        p=rdf.predict_proba(X_test_1[s])
        for n in range(len(X_test_1[s])):
            k=int(y_test_1[s].values[n])
            rf_prob_lr[X_test_1[s].index[n]]=p[n][k]
            
    rf_prob=sum(rf_prob_rf.values())+sum(rf_prob_nb.values())+sum(rf_prob_lr.values())
    if rf_prob_final < rf_prob:
        rdf_rf=RandomForestClassifier(warm_start=True,max_depth=5,class_weight={0:6,1:1},min_samples_leaf=3).fit(rf_cluster,rf_y)
        rf_prob_final=rf_prob
        
    X_train_1,X_test_1,y_train_1,y_test_1=[],[],[],[]
    skf=StratifiedKFold(n_splits=10,shuffle=False, random_state=None)
    for train_index, test_index in skf.split(nb_cluster, nb_y):
        X_train_1.append(nb_cluster.iloc[train_index]) 
        X_test_1.append(nb_cluster.iloc[test_index]) 
        y_train_1.append(nb_y.iloc[train_index])
        y_test_1.append(nb_y.iloc[test_index]) 

    #NB without any technic class_weight=6:1###
    rdf=BernoulliNB(fit_prior=1,class_prior=[3,1])
    Auc=0
    precision=np.zeros(2)
    recall=np.zeros(2)
    f1=np.zeros(2)
    score=0
    for s in range(10):
        rdf.fit(X_train_1[s].values,y_train_1[s].values)
        scr=rdf.score(X_test_1[s].values,y_test_1[s].values)
        pred=rdf.predict(X_test_1[s])
        score+=scr
        precision+=precision_score(y_test_1[s], pred, average=None,zero_division=0)/10
        recall+=recall_score(y_test_1[s], pred, average=None,zero_division=0)/10
        f1+=f1_score(y_test_1[s], pred, average=None)/10
        fpr, tpr, thresholds = metrics.roc_curve(y_test_1[s], pred, pos_label=1)
        Auc+=metrics.auc(fpr, tpr)/10
    print('score:',score/10,'\n','precision:',precision,'\n','recall:',recall,'\n','f1:',f1,'\n','auc:',Auc)

    rdf=BernoulliNB(fit_prior=1,class_prior=[3,1])
    nb_prob_nb={}
    for s in range(10):
        rdf.fit(X_train_1[s],y_train_1[s])
        p=rdf.predict_proba(X_test_1[s])
        for n in range(len(X_test_1[s])):
            k=int(y_test_1[s].values[n])
            nb_prob_nb[X_test_1[s].index[n]]=p[n][k]

    rdf=LogisticRegression(warm_start=True,random_state=0,C=0.2,class_weight={0:4,1:1})
    nb_prob_lr={}
    rdf.fit(lr_cluster,lr_y)
    for s in range(10):
        p=rdf.predict_proba(X_test_1[s])
        for n in range(len(X_test_1[s])):
            k=int(y_test_1[s].values[n])
            nb_prob_lr[X_test_1[s].index[n]]=p[n][k]

    rdf=RandomForestClassifier(warm_start=True,max_depth=5,class_weight={0:6,1:1},min_samples_leaf=3)
    nb_prob_rf={}
    rdf.fit(rf_cluster,rf_y)
    for s in range(10):
        p=rdf.predict_proba(X_test_1[s])
        for n in range(len(X_test_1[s])):
            k=int(y_test_1[s].values[n])
            nb_prob_rf[X_test_1[s].index[n]]=p[n][k]
            
    nb_prob=sum(nb_prob_rf.values())+sum(nb_prob_nb.values())+sum(nb_prob_lr.values())
    if nb_prob_final < nb_prob:
        rdf_nb=BernoulliNB(fit_prior=1,class_prior=[3,1]).fit(nb_cluster,nb_y)
        nb_prob_final=nb_prob

    X_train_1,X_test_1,y_train_1,y_test_1=[],[],[],[]
    skf=StratifiedKFold(n_splits=10,shuffle=False, random_state=None)
    for train_index, test_index in skf.split(lr_cluster, lr_y):
        X_train_1.append(lr_cluster.iloc[train_index]) 
        X_test_1.append(lr_cluster.iloc[test_index]) 
        y_train_1.append(lr_y.iloc[train_index])
        y_test_1.append(lr_y.iloc[test_index]) 

    rdf=LogisticRegression(warm_start=True,random_state=0,C=0.2,class_weight={0:4,1:1})
    Auc=0
    precision=np.zeros(2)
    recall=np.zeros(2)
    f1=np.zeros(2)
    score=0    
    for s in range(10):
        rdf.fit(X_train_1[s].values,y_train_1[s].values)
        scr=rdf.score(X_test_1[s].values,y_test_1[s].values)
        pred=rdf.predict(X_test_1[s])
        score+=scr
        precision+=precision_score(y_test_1[s], pred, average=None,zero_division=0)/10
        recall+=recall_score(y_test_1[s], pred, average=None,zero_division=0)/10
        f1+=f1_score(y_test_1[s], pred, average=None)/10
        fpr, tpr, thresholds = metrics.roc_curve(y_test_1[s], pred, pos_label=1)
        Auc+=metrics.auc(fpr, tpr)/10
    print('score:',score/10,'\n','precision:',precision,'\n','recall:',recall,'\n','f1:',f1,'\n','auc:',Auc)

    rdf=LogisticRegression(warm_start=True,random_state=0,C=0.2,class_weight={0:4,1:1})
    lr_prob_lr={}
    for s in range(10):
        rdf.fit(X_train_1[s],y_train_1[s])
        p=rdf.predict_proba(X_test_1[s])
        for n in range(len(X_test_1[s])):
            k=int(y_test_1[s].values[n])
            lr_prob_lr[X_test_1[s].index[n]]=p[n][k]
        
    rdf=RandomForestClassifier(warm_start=True,max_depth=5,class_weight={0:6,1:1},min_samples_leaf=3)
    lr_prob_rf={}
    rdf.fit(rf_cluster,rf_y)
    for s in range(10):
        p=rdf.predict_proba(X_test_1[s])
        for n in range(len(X_test_1[s])):
            k=int(y_test_1[s].values[n])
            lr_prob_rf[X_test_1[s].index[n]]=p[n][k]

    rdf=BernoulliNB(fit_prior=1,class_prior=[3,1])
    lr_prob_nb={}
    rdf.fit(nb_cluster,nb_y)
    for s in range(10):
        p=rdf.predict_proba(X_test_1[s])
        for n in range(len(X_test_1[s])):
            k=int(y_test_1[s].values[n])
            lr_prob_nb[X_test_1[s].index[n]]=p[n][k]
            
    lr_prob=sum(lr_prob_rf.values())+sum(lr_prob_nb.values())+sum(lr_prob_lr.values())
    if lr_prob_final < lr_prob:
        rdf_lr=LogisticRegression(warm_start=True,random_state=0,C=0.2,class_weight={0:4,1:1}).fit(lr_cluster,lr_y)
        lr_prob_final=lr_prob
        
    rf=list(rf_cluster.index)
    nb=list(nb_cluster.index)
    lr=list(lr_cluster.index)

    pb_res=dict(zip(X_t.index,np.zeros(len(X_t.index))))
    for m1 in list(rf_cluster.index):
        if rf_prob_nb[m1] > rf_prob_rf[m1] or rf_prob_lr[m1] > rf_prob_rf[m1]:
            #print('number',len(rf_y[rf_y==rf_y[m1]]))
            if len(rf_y[rf_y==rf_y[m1]]) > 15: 
                rf.remove(m1)
                if rf_prob_lr[m1] > rf_prob_nb[m1]:
                    lr.append(m1)
                if rf_prob_lr[m1] <= rf_prob_nb[m1]:
                    nb.append(m1)
                rf_y=y_b.loc[set(rf)]
        if pb_res[m1]<(max([rf_prob_rf[m1],rf_prob_nb[m1],rf_prob_lr[m1]])):
            pb_res[m1]=(max([rf_prob_rf[m1],rf_prob_nb[m1],rf_prob_lr[m1]]))
                
    for m2 in list(lr_cluster.index):
        if lr_prob_rf[m2] > lr_prob_lr[m2] or lr_prob_nb[m2] > lr_prob_nb[m2]:
            if len(lr_y[lr_y==lr_y[m2]]) > 15: 
                lr.remove(m2)
                if lr_prob_nb[m2] > lr_prob_rf[m2]:
                    nb.append(m2)
                if lr_prob_nb[m2] <= lr_prob_rf[m2]:
                    rf.append(m2)
                lr_y=y_b.loc[set(lr)]
        if pb_res[m2]<(max([lr_prob_rf[m2],lr_prob_nb[m2],lr_prob_lr[m2]])):
            pb_res[m2]=(max([lr_prob_rf[m2],lr_prob_nb[m2],lr_prob_lr[m2]])) 
            
    for m3 in list(nb_cluster.index):
        if nb_prob_rf[m3] > nb_prob_nb[m3] or nb_prob_lr[m3] > nb_prob_nb[m3]:
            if len(nb_y[nb_y==nb_y[m3]]) > 15: 
                nb.remove(m3)
                if nb_prob_lr[m3] > nb_prob_rf[m3]:
                    lr.append(m3)
                if nb_prob_lr[m3] <= nb_prob_rf[m3]:
                    rf.append(m3)
                nb_y=y_b.loc[set(nb)]
        if pb_res[m3]<(max([nb_prob_rf[m3],nb_prob_nb[m3],nb_prob_lr[m3]])):
            pb_res[m3]=(max([nb_prob_rf[m3],nb_prob_nb[m3],nb_prob_lr[m3]]))
     
    turn+=1
    rf1,nb1,lr1=set(rf),set(nb),set(lr)
    if ssb > sum(pb_res.values()):
        break
    ssb=sum(pb_res.values())
    print(turn,ssb)

#parameters={'max_depth':[3,5,8,10,20,60],'min_samples_leaf':[2,3,4],'n_estimators':[10,20,50,100]}
#rdf=findBestEstimator (X_train[s],y_train[s],parameters,esm=RandomForestClassifier(class_weight={0:6,1:1}))
rdf=rdf_rf
fea_proba_rf=rdf.predict_proba(X_sampled)

#parameters={}
#rdf=findBestEstimator (X_train[s],y_train[s],parameters,esm=BernoulliNB(fit_prior=0,class_prior=[3,1]))
rdf=rdf_nb
fea_proba_nb=rdf.predict_proba(X_sampled)

#parameters={'C':[0.01,0.05,0.1,0.5,1,],'multi_class':['ovr','multinomial']}
#rdf=findBestEstimator (X_train[s],y_train[s],parameters,esm=LogisticRegression(solver='saga',max_iter=1000,class_weight={0:6,1:1}))
rdf=rdf_lr
fea_proba_lr=rdf.predict_proba(X_sampled)

fea_proba_rf=dict(zip(y_sampled.index,fea_proba_rf))
fea_proba_nb=dict(zip(y_sampled.index,fea_proba_nb))
fea_proba_lr=dict(zip(y_sampled.index,fea_proba_lr))

### highest probability ###
fea_res=[]
for i in y_sampled.index:
    k=0
    max_p=fea_proba_rf[i][0]
    q=int(y_sampled[i])
    
    if fea_proba_rf[i][0]<fea_proba_rf[i][1]:
        k=1
        max_p=fea_proba_rf[i][1]
    if max_p<fea_proba_nb[i][0]:
        k=2
        max_p=fea_proba_nb[i][0]
    if max_p<fea_proba_nb[i][1]:
        k=3
        max_p=fea_proba_nb[i][1]
    if max_p<fea_proba_lr[i][0]:
        k=4
        max_p=fea_proba_lr[i][0]
    if max_p<fea_proba_lr[i][1]:
        k=5
    fea_res.append(k%2)

Auc=0
precision=np.zeros(2)
recall=np.zeros(2)
f1=np.zeros(2)
score=0
pred=fea_res
precision+=precision_score(y_sampled.values, pred, average=None,zero_division=0)
recall+=recall_score(y_sampled.values, pred, average=None,zero_division=0)
f1+=f1_score(y_sampled.values, pred, average=None)
fpr, tpr, thresholds = metrics.roc_curve(y_sampled.values, pred, pos_label=1)
Auc+=metrics.auc(fpr, tpr)
print('precision:',precision,'\n','recall:',recall,'\n','f1:',f1,'\n','auc:',Auc)


''' stat '''

df_rf=rf_cluster
df_nb=nb_cluster
df_lr=lr_cluster

import scipy.stats as stats
import numpy as np
df_anova=pd.DataFrame()
for fea in df_rf.columns:
    fvalue, pvalue = stats.f_oneway(df_rf[fea], df_nb[fea], df_lr[fea])
    w, w_pvalue = stats.bartlett(df_rf[fea], df_nb[fea], df_lr[fea])
    print(pvalue,fea)
    df_anova[fea]=np.array([pvalue,w_pvalue])
df_anova.to_excel('anova.xlsx')
stat_res=pd.DataFrame(index=rf_cluster.columns,columns=['rf_mean','rf_std','nb_mean','nb_std','lr_mean','lr_std'])
stat_res.to_excel('stat_res.xlsx')
stat_res['rf_mean']=rf_cluster.mean()
stat_res['rf_std']=rf_cluster.std()
stat_res['nb_mean']=nb_cluster.mean()
stat_res['nb_std']=nb_cluster.std()
stat_res['lr_mean']=lr_cluster.mean()
stat_res['lr_std']=lr_cluster.std()

def tukey(fea):
    df_melt=pd.DataFrame([list(df_rf[fea]),list(df_nb[fea]),list(df_lr[fea])])
    df_melt=df_melt.T
    df_melt.columns=['rf','nb','lr']
    df_melt = pd.melt(df_melt.reset_index(), id_vars=['index'], value_vars=['rf','nb','lr'])
    df_melt.columns = ['index', 'treatments', 'value']
    res = stat()
    res.tukey_hsd(df=df_melt, res_var='value', xfac_var='treatments', anova_model='value ~ C(treatments)')
    return res.tukey_summary

from bioinfokit.analys import stat
tukey_res=pd.DataFrame(index=['rf vs nb','rf vs lr','nb vs lr'])
for m in df_rf.columns:
    s=tukey(m)
    tukey_res[m]=list(s['p-value'])
tukey_res.to_excel('tukey_res.xlsx')

''' Ensemble '''
clf1=RandomForestClassifier(max_depth=7,class_weight={0:6,1:1})
clf2=BernoulliNB(fit_prior=0,class_prior=[3,1])
clf3=LogisticRegression(random_state=0,C=0.5,class_weight={0:4,1:1})
eclf1 = VotingClassifier(estimators=[('rf', clf1),('lr', clf3),  ('nb', clf2)], voting='soft')
eclf1.fit(X,y_b)
r = eclf1.predict(X_sampled)

Auc=0
precision=np.zeros(2)
recall=np.zeros(2)
f1=np.zeros(2)
score=0
pred=r
precision+=precision_score(y_sampled.values, pred, average=None,zero_division=0)
recall+=recall_score(y_sampled.values, pred, average=None,zero_division=0)
f1+=f1_score(y_sampled.values, pred, average=None)
fpr, tpr, thresholds = metrics.roc_curve(y_sampled.values, pred, pos_label=1)
Auc+=metrics.auc(fpr, tpr)
print('precision:',precision,'\n','recall:',recall,'\n','f1:',f1,'\n','auc:',Auc)














