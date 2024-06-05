from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
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


from sklearn.tree import plot_tree,export_text


class Algorithms():
    
    def GAM_simple(parameters):
        sc = StandardScaler()
        nomi=parameters['nomi']
        train=parameters['train']
        test=parameters['testy']
        #X_train=Algorithms.scaler(train,parameters['linear']+parameters['continuous'])
        X_train_sc = sc.fit_transform(train[parameters['linear']+parameters['continuous']])
        X_train = np.hstack((X_train_sc, train[parameters['categorical']]))
        lams = np.empty(len(nomi))
        lams.fill(0.5)        

        if parameters['family']=='binomial':
            gam = LogisticGAM(parameters['splines'], dtype=parameters['dtypes'])
            gam.gridsearch(X_train, train['y'], lam=lams)
            GAM_utils.GAM_plot(gam,parameters['train'],nomi,parameters['fold'],'',X_train)
            GAM_utils.GAM_save(gam,parameters['fold'])
            prob_fit=gam.predict_proba(X_train)#[::,1]
            train['SI']=prob_fit
            if parameters['testN']>0:
                X_test_sc = sc.transform(test[parameters['linear']+parameters['continuous']])
                X_test = np.hstack((X_test_sc, test[parameters['categorical']]))
                prob_predic=gam.predict_proba(X_test)#[::,1]
                test['SI']=prob_predic
                train['SI']=prob_fit
        else:
            gam = LinearGAM(parameters['splines'], dtype=parameters['dtypes'])
            gam.gridsearch(X_train, train['y'],lam=lams)
            GAM_utils.GAM_plot(gam,parameters['train'],nomi,parameters['fold'],'',X_train)
            GAM_utils.GAM_save(gam,parameters['fold'])
            prob_fit=gam.predict(X_train)#[::,1]
            #CI = gam.prediction_intervals(X_train, width=.95)
            train['SI']=prob_fit#np.exp(prob_fit)
            if parameters['testN']>0:
                X_test_sc = sc.transform(test[parameters['linear']+parameters['continuous']])
                X_test = np.hstack((X_test_sc, test[parameters['categorical']]))
                prob_predic=gam.predict(X_test)#[::,1]
                #CI = gam.prediction_intervals(X_test, width=.95)
                test['SI']=prob_predic#np.exp(prob_predic)
                train['SI']=prob_fit#np.exp(prob_fit)
        return(train,test,gam)
    
    def GAM_transfer(parameters):
        sc = StandardScaler()
        nomi=parameters['nomi']
        trans=parameters['trans']
        #X_trans = sc.fit_transform(trans[nomi])
        X_train_sc = sc.fit_transform(trans[parameters['linear']+parameters['continuous']])
        X_trans = np.hstack((X_train_sc, trans[parameters['categorical']]))
        if parameters['family']=='binomial':
            prob_fit=parameters['gam'].predict_proba(X_trans)#[::,1]
            trans['SI']=prob_fit
        else:
            prob_fit=parameters['gam'].predict(X_trans)#[::,1]
            #CI = parameters['gam'].prediction_intervals(X_trans, width=.95)
            trans['SI']=prob_fit#np.exp(prob_fit)
        return(trans)

    
    ####################################

    def alg_MLrun(classifier,X,y,train,test,fold,df,nomi):
        classifier.fit(X[train], y[train])
        prob_predic=classifier.predict_proba(X[test])[::,1]
        try:
            regression_coeff=classifier.feature_importances_
            coeff=regression_coeff
            try:
                tree_rules = export_text(classifier, feature_names=nomi)
                tree_rules_list = tree_rules.split('\n')
                rules_df = pd.DataFrame({'Tree Rules': tree_rules_list})
                rules_df.to_csv(fold+'/decision_tree_rules.csv', index=False)
            except:
                print('no tree')
            feature_importance_df = pd.DataFrame({
                'Feature': nomi,
                'Importance': coeff
            })
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            feature_importance_df.to_csv(fold+'/feature_importances.csv', index=False)
        except:
            regression_coeff=classifier.coef_
            regression_intercept=classifier.intercept_
            coeff=np.hstack((regression_intercept,regression_coeff[0]))
            coeff_df = pd.DataFrame({
                'Feature': ['intercept'] + nomi,
                'Coefficient': coeff
            })
            coeff_df.to_csv(fold+'/coefficients.csv', index=False)

        return prob_predic,None
    
 
    def GAM_cv(classifier,X,y,train,test,splines=None,dtypes=None,nomi=None,df=None,fold=None,filename=None,scaler=None):
        lams = np.empty(len(nomi))
        lams.fill(0.5)
        gam = classifier(splines, dtype=dtypes)
        gam.gridsearch(X[train,], y[train], lam=lams,progress=False)
        GAM_utils.GAM_plot(gam,df.iloc[train,],nomi,fold,filename,X[train,])
        GAM_utils.GAM_save(gam,fold,filename)
        prob_predic=gam.predict_proba(X[test])#[::,1]
        CI={}#gam.prediction_intervals(X[test])
        GAM_utils.plot_predict(X[test],prob_predic,CI,fold,filename)

        return prob_predic,None,CI
    
    def scaler(df,nomes):
        df_scaled=df.copy()
        for nome in nomes:
            s=df[nome].std()
            u=df[nome].mean()
            df_scaled[nome]=(df[nome]-u)/s
        return df_scaled
            
    
class CV_utils():
    def cross_validation(parameters,algorithm,classifier):
        df=parameters['df']
        nomi=parameters['nomi']
        x=df[parameters['field1']]
        y=df['y']
        if algorithm==Algorithms.GAM_cv:
            X=Algorithms.scaler(x,parameters['linear']+parameters['continuous'])
            X[parameters['categorical']]=df[parameters['categorical']]
        else:
            sc = StandardScaler()#####scaler
            X = sc.fit_transform(x)
        #X=x
        #sc_fit = sc.fit(df[parameters['linear']+parameters['continuous']])
        
        train_ind={}
        test_ind={}
        prob={}
        CI={}
        cofl=[]
        df["SI"] = np.nan
        df["CI"] = np.nan
        if parameters['testN']>1:
            cv = StratifiedKFold(n_splits=parameters['testN'])
            for i, (train, test) in enumerate(cv.split(X, y)):
                train_ind[i]=train
                test_ind[i]=test
                if algorithm==Algorithms.GAM_cv:
                    #X_train=X[train,]             
                    #X_train[parameters['linear']+parameters['continuous']]=sc.transform(X_train[parameters['linear']+parameters['continuous']])
                    #X_test=X[test,]             
                    #X_test[parameters['linear']+parameters['continuous']]=sc.transform(X_test[parameters['linear']+parameters['continuous']])
                    #print(X_train,X_test)
                    #prob[i],coeff=algorithm(classifier,X,y,train,test,splines=parameters['splines'],dtypes=parameters['dtypes'],nomi=nomi,df=df,fold=parameters['fold'],filename=str(i),scaler=sc)
                    lams = np.empty(len(nomi))
                    lams.fill(0.5)
                    gam = classifier(parameters['splines'], dtype=parameters['dtypes'])
                    gam.gridsearch(X.iloc[train,:].to_numpy(), y.iloc[train].to_numpy(), lam=lams,progress=False)
                    GAM_utils.GAM_plot(gam,df.iloc[train,:],nomi,parameters['fold'],str(i),X.iloc[train,:])
                    GAM_utils.GAM_save(gam,parameters['fold'],str(i))
                    prob[i]=gam.predict_proba(X.iloc[test,:].to_numpy())#[::,1]
                    CI[i]=[]#gam.prediction_intervals(X.iloc[test,:].to_numpy())
                    GAM_utils.plot_predict(X.iloc[test,:].to_numpy(),prob[i],CI,parameters['fold'], str(i))
                    coeff=None
                else:
                    print(parameters['fold'],'fold')
                    prob[i],coeff=algorithm(classifier,X,y,train,test,fold=parameters['fold'],df=df,nomi=nomi)
                df.loc[test,'SI']=prob[i]
                #df.loc[test,'CI']=[]#CI[i]
                cofl.append(coeff)
        elif parameters['testN']==1:
            train=np.arange(len(y))
            test=np.arange(len(y))
            if algorithm==Algorithms.GAM_cv:
                prob[0],coeff,CI[0]=algorithm(classifier,X,y,train,test,splines=parameters['splines'],dtypes=parameters['dtypes'],nomi=nomi,df=df,fold=parameters['fold'],filename='')
            else:
                prob[0],coeff=algorithm(classifier,X,y,train,test,fold=parameters['fold'],df=df,nomi=nomi)
            df.loc[test,'SI']=prob[0]
            #df.loc[test,'CI']=[]#CI[0]
            test_ind[0]=test
            cofl.append(coeff)
        if not os.path.exists(parameters['fold']):
            os.mkdir(parameters['fold'])
        if coeff is not None:
            with open(parameters['fold']+'/r_coeffs.csv', 'w') as f:
                write = csv.writer(f)
                ll=['intercept']
                lll=ll+nomi
                write.writerow(lll)
                write.writerows(cofl)
        return prob,test_ind
    
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
            if term.isintercept:
                continue
            pdep0, confi0 = gam.partial_dependence(term=i, X=gam.generate_X_grid(term=i), width=0.95)
            if isinstance(gam.terms[i], terms.FactorTerm):
                continue
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
            #XX = gam.generate_X_grid(term=i,n=len(df[GAM_sel[i]]))
            #pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
            ##
            XX=df[GAM_sel[i]].to_numpy()
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
            ##

            YY=pdep
            if int(np.ceil(count/3.))<4:
                rows=4
            else:
                rows=int(np.ceil(count/3.))
            
            plt.subplot(rows, 3, i+1)

            if isinstance(gam.terms[i], terms.FactorTerm):
                plt.plot(np.sort(df[GAM_sel[i]]),pdep, 'o', c='blue')
                plt.xticks(np.sort(df[GAM_sel[i]].unique()), rotation=90)
                plt.xlabel(GAM_sel[i])
                plt.ylabel('Partial Effect')
                continue
            #elif isinstance(gam.terms[i], terms.LinearTerm):
            #    print(pdep)
            #    x_linear=np.append(x_linear,i)
            #    y_linear=np.append(y_linear,np.unique(pdep))
            #    x_tic=x_tic+[GAM_sel[i]]
            #    continue
            ##
            #plt.plot(XX[:, term.feature], pdep, c='blue')
            #plt.fill_between(XX[:, term.feature].ravel(), y1=confi[:,0], y2=confi[:,1], color='gray', alpha=0.2)
            ##
            ## plt.plot(XX[:, term.feature], confi, c='r', ls='--')
            plt.plot(np.sort(XX), pdep, c='blue')
            plt.fill_between(np.sort(XX), y1=confi[:,0], y2=confi[:,1], color='gray', alpha=0.2)

            plt.xlabel(GAM_sel[i])
            plt.ylabel('Partial Effect')
            plt.ylim(MIN,MAX)

        #if len(x_linear)>0:
        #    print(x_linear,y_linear)
        #    plt.plot(x_linear,y_linear, 'o', c='blue')
        #    plt.xticks(x_tic,rotation=90)
        #    #plt.xlabel(GAM_sel[i])
        #    plt.ylabel('Regression coefficient')

        plt.savefig(fold+'/Model_covariates'+filename+'.pdf', bbox_inches='tight')
        #plt.show()
    
        
    def GAM_save(gam,fold,filename=''):
        filename_pkl = fold+'/gam_coeff'+filename+'.pkl'
        #filename_txt = parameters['fold']+'/gam_coeff.txt'

        with open(filename_pkl, 'wb') as filez:
            pickle.dump(gam, filez)
        
        #with open(filename_pkl, 'rb') as file_pickle:
        #    loaded_data = pickle.load(file_pickle)

        #with open(filename_txt, 'wb') as file_txt:
        #    json.dump(loaded_data, file_txt, indent=2)
    
    def plot_predict(x,predict,CI,fold, filename=''):
        plt.plot(x,predict,'r--')
        #plt.plot(x,CI,color='b',ls='--')
        plt.xlabel('Prediction8')
        plt.ylabel('PDF')
        plt.savefig(fold+'/Predict'+filename+'.pdf')


    

    

    
