from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,LeaveOneOut
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import math
#from pygam import LinearGAM,LogisticGAM
import pickle
import os
from collections import OrderedDict
from pygam import terms
import csv
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans
from shapely.geometry import MultiPolygon, Polygon
from shapely.wkt import loads

from sklearn.tree import export_text


class Algorithms():
    
    def GAM_transfer(parameters):
        nomi=parameters['nomi']
        df=parameters['df']
        #x=df[parameters['field1']]
        df_scaled=CV_utils.scaler(df,parameters['linear']+parameters['continuous'],'custom')

        if parameters['family']=='binomial':
            prob_fit=parameters['gam'].predict_proba(df_scaled)#[::,1]
            df['SI']=prob_fit
        else:
            prob_fit=parameters['gam'].predict(df_scaled)#[::,1]
            #CI = parameters['gam'].prediction_intervals(X_trans, width=.95)
            df['SI']=prob_fit#np.exp(prob_fit)
        return(df)

    def alg_MLrun(classifier,X,y,train,test,df,fold,nomi,filename=''):
        print(X.head())
        classifier.fit(X.loc[train,nomi].to_numpy(), y.iloc[train].to_numpy())
        prob_predic=classifier.predict_proba(X.loc[test,nomi].to_numpy())[::,1]
        print(nomi)

        ML_utils.ML_save(classifier,fold,nomi,filename)
        return prob_predic
    
    def alg_GAMrun(classifier,X,y,train,test,df,splines=None,dtypes=None,nomi=None,fold=None,filename='',family=None):
        lams = np.empty(len(nomi))
        lams.fill(0.5)
        classifier_selected=classifier[family]
        gam = classifier_selected(splines, dtype=dtypes)
        gam.gridsearch(X.loc[train,nomi].to_numpy(), y.iloc[train].to_numpy(), lam=lams,progress=False)
        if family=='binomial':
            prob=gam.predict_proba(X.loc[test,nomi].to_numpy())#[::,1]
            #CI=gam.confidence_intervals(X.iloc[test,:].to_numpy(),width=0.95)
        else:
            prob=gam.predict(X.loc[test,nomi].to_numpy())#[::,1]
            #CI=gam.prediction_intervals(X.iloc[test,:].to_numpy())
        GAM_utils.GAM_plot(gam,df.loc[train,nomi],nomi,fold,filename,X.loc[train,nomi])
        GAM_utils.GAM_save(gam,prob,fold,nomi,filename)
        #GAM_utils.plot_predict(X.iloc[test,:].to_numpy(),prob,CI,fold, filename)
        #print(CI)
        CI=[]
        return prob,CI,gam
    
class CV_utils():

    def cross_validation(parameters,algorithm,classifier):
        df=parameters['df']
        nomi=parameters['nomi']
        x=df[parameters['field1']]
        #print(x,'x')
        #print(df,'df')
        y=df['y']
        if algorithm==Algorithms.alg_GAMrun:
            df_scaled=CV_utils.scaler(x,parameters['linear']+parameters['continuous'],'custom')
            #X[parameters['categorical']]=df[parameters['categorical']]
        else:
            df_scaled=CV_utils.scaler(df,nomi,'standard')
        train_ind={}
        test_ind={}
        prob={}
        CI={}
        cofl=[]
        df["SI"] = np.nan
        df["CI"] = np.nan
        coeff=None
        if parameters['testN']>1:
            train_ind,test_ind = CV_utils.cv_method(parameters,df_scaled,df,parameters['field1'])
            for i in range(parameters['testN']):
                if algorithm==Algorithms.alg_GAMrun:
                    prob[i],CI[i],gam=algorithm(classifier,df_scaled,y,train_ind[i],test_ind[i],df,splines=parameters['splines'],dtypes=parameters['dtypes'],nomi=nomi,fold=parameters['fold'],filename=str(i),family=parameters['family'])
                    #df.loc[test,'CI']=CI[i]
                else:
                    prob[i]=algorithm(classifier,df_scaled,y,train_ind[i],test_ind[i],df,fold=parameters['fold'],nomi=nomi,filename=str(i))
                    gam=None
                df.loc[test_ind[i],'SI']=prob[i]
        elif parameters['testN']==1:
            train=np.arange(len(y))
            test=np.arange(len(y))
            if algorithm==Algorithms.alg_GAMrun:
                prob[0],CI[0],gam=algorithm(classifier,df_scaled,y,train,test,df,splines=parameters['splines'],dtypes=parameters['dtypes'],nomi=nomi,fold=parameters['fold'],family=parameters['family'])
                #df.loc[test,'CI']=CI[0]
            else:
                prob[0]=algorithm(classifier,df_scaled,y,train,test,df,fold=parameters['fold'],nomi=nomi)
                gam=None
            df.loc[test,'SI']=prob[0]
            
            test_ind[0]=test

        return prob,test_ind,gam
    
    def cv_method(parameters,df_scaled,df,nomi):
        X_train={}
        X_test={}
        y = df['y'].to_numpy()
        if parameters['cv_method']=='spatial':
            for index, row in df.iterrows():
                multipolygon = loads(df.loc[index,'geom'])
                # Compute the centroid of the MultiPolygon
                centroid = multipolygon.centroid
                # Get the x and    y coordinates of the centroid
                x, y = centroid.x, centroid.y
                # Extract x and y coordinates
                df_scaled.loc[index,'X_coord'] = x
                df_scaled.loc[index,'Y_coord'] = y
            # Create a DataFrame with the coordinates
            coords = df_scaled[['X_coord', 'Y_coord']]
            # Standardize the coordinates
            scaler = StandardScaler()
            coords_scaled = scaler.fit_transform(coords)
            kmeans = KMeans(n_clusters=parameters['testN'], random_state=10, n_init=2, max_iter=10).fit(coords_scaled)
            loo = LeaveOneOut()
            #method=loo.split(kmeans.labels_)
            for i, (train, test) in enumerate(loo.split(np.arange(parameters['testN']))):
                X_train[i] = np.where(kmeans.labels_ != test)[0]
                X_test[i] = np.where(kmeans.labels_ == test)[0]
                #y_train[i] = np.where(kmeans.labels_ != test)#############da finireeeee
            print('spatial')
        elif parameters['cv_method']=='random':
            print('random')
            method=StratifiedKFold(n_splits=parameters['testN'])
            for i, (train, test) in enumerate(method.split(df_scaled, y)):
                X_train[i]=train
                X_test[i]=test
        #elif:
        #    loo = LeaveOneOut()
        #    for train, test in loo.split(X):
        #return(method)
        return X_train,X_test
    
    def scaler(df,nomes,scale_method='standard'):
        df_scaled=df.copy()
        if scale_method=='custom':
            for nome in nomes:
                s=df[nome].std()
                u=df[nome].mean()
                df_scaled[nome]=(df[nome]-u)/s
        elif scale_method=='standard':
            sc = StandardScaler()#####scaler
            array_scaled = sc.fit_transform(df[nomes])  
            df_scaled = pd.DataFrame(array_scaled, columns=nomes)
        none_values = df.isnull().sum()
        print('errorriiiiiii',none_values)
        return df_scaled

class GAM_utils():
    def GAM_formula(parameters):
        GAM_sel = parameters['nomi']
        spl = parameters['spline']
        splines = []
        dtypes = []
        vars_dict = OrderedDict({})

        for i in range(len(GAM_sel)):
            if GAM_sel[i] in parameters['continuous']:
                dtypes = dtypes + ['numerical']
                vars_dict[GAM_sel[i]]={'term':'s', 'n_splines':spl}
            elif GAM_sel[i] in parameters['categorical']:
                dtypes = dtypes + ['categorical']
                vars_dict[GAM_sel[i]]={'term':'f'}
            elif GAM_sel[i] in parameters['linear']:
                dtypes = dtypes + ['numerical']
                vars_dict[GAM_sel[i]]={'term':'l'}
        
        splines = terms.TermList()
        for i,v in enumerate(vars_dict .keys()):
            if vars_dict[v]['term'] == 's':
                term = terms.SplineTerm(i, n_splines=vars_dict[v].get('n_splines', 10))
            elif vars_dict[v]['term'] == 'l':
                term = terms.LinearTerm(i)
            elif vars_dict[v]['term'] == 'f':
                term = terms.FactorTerm(i)
            splines += term

        return splines,dtypes
    
    def GAM_plot(gam,df,nomi,fold,filename,scaled_df):

        GAM_sel=nomi
        #sc=StandardScaler()
        fig = plt.figure(figsize=(20, 25))

        maX=[]
        miN=[]
        for i, term in enumerate(gam.terms):
            #print(gam.terms[i])
            if term.isintercept:
                continue
            if isinstance(gam.terms[i], terms.FactorTerm):
                print('ciao')
                continue
            else:
                pdep0, confi0 = gam.partial_dependence(term=i, X=gam.generate_X_grid(term=i), width=0.95)
            maX=maX+[np.max(confi0[:,1])]
            miN=miN+[np.min(confi0[:,0])]
        MAX=max(maX)+1
        MIN=min(miN)-1

        count=0
        countIns=0
        for i, term in enumerate(gam.terms):
            if term.isintercept:
                continue
            if isinstance(gam.terms[i], terms.FactorTerm):
                countIns+=1
                continue
            count+=1
        count=count+countIns

        x_tic=[]
        x_linear=np.array([])
        y_linear=np.array([])
            
        for i, term in enumerate(gam.terms):
            if term.isintercept:
                continue
            ##
            XX = gam.generate_X_grid(term=i,n=len(df[GAM_sel[i]]))
            #pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
            ##
            #XX=df[GAM_sel[i]].to_numpy()
            try:
                sorted_indices = np.argsort(scaled_df.iloc[:, i])
                X = scaled_df.iloc[sorted_indices].to_numpy()
            except:
                sorted_indices = np.argsort(scaled_df[:, i])
                X = scaled_df[sorted_indices]

            for ii in range(len(GAM_sel)):
                if ii != i: 
                    X[:, ii] = 0
            pdep, confi = gam.partial_dependence(term=i, X=X, width=0.95)
            #print(pdep,'pdep')
            ##

            YY=pdep
            if int(np.ceil(count/3.))<4:
                rows=4
            else:
                rows=int(np.ceil(count/3.))
            
            plt.subplot(rows, 3, i+1)

            # if isinstance(gam.terms[i], terms.FactorTerm):
            #     plt.plot(np.sort(df[GAM_sel[i]].unique()),pdep[range(1, len(pdep), 3)], 'o', c='blue')
            #     for j in pdep[range(1, len(pdep), 3)]:
            #         if j>1.5:
            #             plt.axvline(np.sort(df[GAM_sel[0]].unique())[np.where(pdep[range(1, len(pdep), 3)] == j)[0][0]],
            #                         color='k', linewidth=0.5, linestyle="--")
            #     plt.xticks(np.sort(df[GAM_sel[i]].unique()), rotation=90)
            #     plt.xlabel(GAM_sel[i])
            #     plt.ylabel('Partial Effect')
            #     continue
            if isinstance(gam.terms[i], terms.FactorTerm):
                plt.plot(np.sort(df[GAM_sel[i]].unique()),
                        pdep[range(1, len(pdep), 3)], 'o', c='gray')
                for j in pdep[range(1, len(pdep), 3)]:
                    if j>1.5:
                        plt.axvline(np.sort(df[GAM_sel[0]].unique())[np.where(pdep[range(1, len(pdep), 3)] == j)[0][0]],
                                    color='k', linewidth=0.5, linestyle="--")
                plt.xticks(np.sort(df[GAM_sel[i]].unique()), rotation=90)
                plt.xlabel(GAM_sel[i])
                plt.ylabel('Regression coefficient')
                continue

            plt.plot(XX[:, term.feature], pdep, c='blue')
            plt.fill_between(XX[:, term.feature].ravel(), y1=confi[:,0], y2=confi[:,1], color='gray', alpha=0.2)

            plt.xlabel(GAM_sel[i])
            plt.ylabel('Partial Effect')
            #plt.ylim(MIN,MAX)

        #if len(x_linear)>0:
        #    print(x_linear,y_linear)
        #    plt.plot(x_linear,y_linear, 'o', c='blue')
        #    plt.xticks(x_tic,rotation=90)
        #    #plt.xlabel(GAM_sel[i])
        #    plt.ylabel('Regression coefficient')

        plt.savefig(fold+'/Model_covariates'+filename+'.pdf', bbox_inches='tight')
        #plt.show()  
        
    def GAM_save(gam,coeffs,fold,nomi,filename=''):
        filename_pkl = fold+'/gam_coeff'+filename+'.pkl'
        #filename_txt = parameters['fold']+'/gam_coeff.txt'

        with open(filename_pkl, 'wb') as filez:
            pickle.dump(gam, filez)
        
        #with open(filename_pkl, 'rb') as file_pickle:
        #    loaded_data = pickle.load(file_pickle)

        #with open(filename_txt, 'wb') as file_txt:
        #    json.dump(loaded_data, file_txt, indent=2)

        # with open(fold+'/r_coeffs.csv', 'w') as f:
        #     write = csv.writer(f)
        #     ll=['intercept']
        #     lll=ll+nomi
        #     write.writerow(lll)
        #     write.writerows(coeffs)
    
    # def plot_predict(x,predict,CI,fold, filename=''):
    #     plt.plot(x,predict,'r--')
    #     #plt.plot(x,CI,color='b',ls='--')
    #     plt.xlabel('Prediction8')
    #     plt.ylabel('PDF')
    #     plt.savefig(fold+'/Predict'+filename+'.pdf')

class ML_utils():
    def ML_save(classifier,fold,nomi, filename):
        try:#RF,DT
            regression_coeff=classifier.feature_importances_
            coeff=regression_coeff
            print(coeff,nomi)
            try:
                tree_rules = export_text(classifier, feature_names=nomi)
                tree_rules_list = tree_rules.split('\n')
                rules_df = pd.DataFrame({'Tree Rules': tree_rules_list})
                rules_df.to_csv(fold+'/decision_tree_rules'+filename+'.csv', index=False)
            except:
                print('no tree')
            feature_importance_df = pd.DataFrame({
                'Feature': nomi,
                'Importance': coeff
            })
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            feature_importance_df.to_csv(fold+'/feature_importances'+filename+'.csv', index=False)
        except:#SVM
            regression_coeff=classifier.coef_
            regression_intercept=classifier.intercept_
            coeff=np.hstack((regression_intercept,regression_coeff[0]))
            coeff_df = pd.DataFrame({
                'Feature': ['intercept'] + nomi,
                'Coefficient': coeff
            })
            coeff_df.to_csv(fold+'/coefficients'+filename+'.csv', index=False)


    

    

    
