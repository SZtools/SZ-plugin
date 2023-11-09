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
from pygam import LogisticGAM
import pickle
import os
from collections import OrderedDict
from pygam import terms
import csv
import matplotlib.pyplot as plt
import json


class Algorithms():

    def LR_simple(parameters):
        sc = StandardScaler()
        nomi=parameters['nomi']
        train=parameters['train']
        test=parameters['testy']
        X_train = sc.fit_transform(train[nomi])
        logistic_regression = LogisticRegression()
        logistic_regression.fit(X_train,train['y'])
        prob_fit=logistic_regression.predict_proba(X_train)[::,1]
        if parameters['testN']>0:
            X_test = sc.transform(test[nomi])
            predictions = logistic_regression.predict(X_test)
            prob_predic=logistic_regression.predict_proba(X_test)[::,1]
            test['SI']=prob_predic
        train['SI']=prob_fit
        return(train,test)
        
    def DT_simple(parameters):
        sc = StandardScaler()
        nomi=parameters['nomi']
        train=parameters['train']
        test=parameters['testy']
        X_train = sc.fit_transform(train[nomi])
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train,train['y'])
        prob_fit=classifier.predict_proba(X_train)[::,1]
        if parameters['testN']>0:
            X_test = sc.transform(test[nomi])
            predictions = classifier.predict(X_test)
            prob_predic=classifier.predict_proba(X_test)[::,1]
            test['SI']=prob_predic
        train['SI']=prob_fit
        return(train,test)
    
    def RF_simple(parameters):
        sc = StandardScaler()
        nomi=parameters['nomi']
        train=parameters['train']
        test=parameters['testy']
        X_train = sc.fit_transform(train[nomi])
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train,train['y'])
        prob_fit=classifier.predict_proba(X_train)[::,1]
        if parameters['testN']>0:
            X_test = sc.transform(test[nomi])
            predictions = classifier.predict(X_test)
            prob_predic=classifier.predict_proba(X_test)[::,1]
            test['SI']=prob_predic
        train['SI']=prob_fit
        return(train,test)
    
    def SVC_simple(parameters):
        sc = StandardScaler()
        nomi=parameters['nomi']
        train=parameters['train']
        test=parameters['testy']
        X_train = sc.fit_transform(train[nomi])
        classifier = SVC(kernel = 'linear', random_state = 0,probability=True)
        classifier.fit(X_train,train['y'])
        prob_fit=classifier.predict_proba(X_train)[::,1]
        if parameters['testN']>0:
            X_test = sc.transform(test[nomi])
            predictions = classifier.predict(X_test)
            prob_predic=classifier.predict_proba(X_test)[::,1]
            test['SI']=prob_predic
        train['SI']=prob_fit
        return(train,test)
    
    def fr_simple(parameters):
        df=parameters['train']
        test=parameters['testy']
        nomi=parameters['nomi']
        Npx1=None
        Npx2=None
        Npx3=None
        Npx4=None
        file = open(parameters['txt'],'w')#################save W+, W- and Wf
        file.write('covariate,class,Npx1,Npx2,Npx3,Npx4,Wf\n')
        #print('covariates:',nomi)
        for ii in nomi:
            classi=df[ii].unique()
            for i in classi:
                dd=pd.DataFrame()
                dd = df.apply(lambda x : True if x['y'] == 1 and x[ii] == i else False, axis = 1)
                Npx1 = len(dd[dd == True].index)
                dd=pd.DataFrame()
                dd = df.apply(lambda x : True if x[ii] == i else False, axis = 1)
                Npx2 = len(dd[dd == True].index)
                dd=pd.DataFrame()
                dd = df.apply(lambda x : True if x['y'] == 1 else False, axis = 1)
                Npx3 = len(dd[dd == True].index)
                dd=pd.DataFrame()
                #dd = df.apply(lambda x : True if x['y'] == 0 and x['y'] == 1 else False, axis = 1)
                Npx4 = df.shape[0]#len(dd[dd == True].index)
                #print(Npx1,Npx2,Npx3,Npx4)
                if Npx1==0 or Npx3==0:
                    Wf=0.
                    #print(ii,i)
                else:
                    Wf=(np.divide((np.divide(Npx1,Npx2)),(np.divide(Npx3,Npx4))))
                #Wf=Wplus-Wminus
                var=[ii,i,Npx1,Npx2,Npx3,Npx4,Wf]
                file.write(','.join(str(e) for e in var)+'\n')#################save W+, W- and Wf
                df[ii][df[ii]==i]=float(Wf)
                test[ii][test[ii]==i]=float(Wf)
            #df.to_csv(self.f+'/file'+ii+'.csv')
        file.close()
        df['SI']=df[nomi].sum(axis=1)
        test['SI']=test[nomi].sum(axis=1)
        return(df,test)
    
    def woe_simple(parameters):
        df=parameters['train']
        test=parameters['testy']
        nomi=parameters['nomi']
        Npx1=None
        Npx2=None
        Npx3=None
        Npx4=None
        file = open(parameters['txt'],'w')#################save W+, W- and Wf
        file.write('covariate,class,Npx1,Npx2,Npx3,Npx4,W+,W-,Wf\n')
        for ii in nomi:
            classi=df[ii].unique()
            for i in classi:
                dd=pd.DataFrame()
                dd = df.apply(lambda x : True if x['y'] == 1 and x[ii] == i else False, axis = 1)
                Npx1 = len(dd[dd == True].index)
                dd=pd.DataFrame()
                dd = df.apply(lambda x : True if x['y'] == 1 and x[ii] != i else False, axis = 1)
                Npx2 = len(dd[dd == True].index)
                dd=pd.DataFrame()
                dd = df.apply(lambda x : True if x['y'] == 0 and x[ii] == i else False, axis = 1)
                Npx3 = len(dd[dd == True].index)
                dd=pd.DataFrame()
                dd = df.apply(lambda x : True if x['y'] == 0 and x[ii] != i else False, axis = 1)
                Npx4 = len(dd[dd == True].index)
                if Npx1==0 or Npx3==0:
                    Wplus=0.
                else:
                    Wplus=math.log((Npx1/(Npx1+Npx2))/(Npx3/(Npx3+Npx4)))
                if Npx2==0 or Npx4==0:
                    Wminus=0.
                else:
                    Wminus=math.log((Npx2/(Npx1+Npx2))/(Npx4/(Npx3+Npx4)))
                Wf=Wplus-Wminus
                var=[ii,i,Npx1,Npx2,Npx3,Npx4,Wplus,Wminus,Wf]
                file.write(','.join(str(e) for e in var)+'\n')#################save W+, W- and Wf
                df[ii][df[ii]==i]=float(Wf)
                test[ii][test[ii]==i]=float(Wf)
            #df.to_csv(self.f+'/file'+ii+'.csv')
        file.close()
        df['SI']=df[nomi].sum(axis=1)
        test['SI']=test[nomi].sum(axis=1)
        return(df,test)
    
    # def GAM(parameters):
    #     nomi=parameters['nomi']
    #     X = parameters['df'][nomi].to_numpy()
    #     y = parameters['df'].PresAbs.to_numpy()
    #     lams = np.empty(len(nomi))
    #     lams.fill(0.5)
    #     gam = LogisticGAM(parameters['splines'], dtype=parameters['dtypes'])
    #     gam.gridsearch(X, y, lam=lams)
        

    #     # save
    #     filename = folder_models+'/cv_fold_'+hazard+'_'+str(n_fold)+'.pkl'
    #     with open(filename, 'wb') as filez:
    #         pickle.dump(gam, filez)
    #     return gam
    
    def GAM_simple(parameters):
        sc = StandardScaler()
        nomi=parameters['nomi']
        train=parameters['train']
        test=parameters['testy']
        X_train = sc.fit_transform(train[nomi])
        lams = np.empty(len(nomi))
        lams.fill(0.5)
        gam = LogisticGAM(parameters['splines'], dtype=parameters['dtypes'])
        gam.gridsearch(X_train, train['y'], lam=lams)
        GAM_utils.GAM_plot(gam,parameters['df'],nomi,parameters['fold'],'')
        GAM_utils.GAM_save(gam,parameters['fold'])
        prob_fit=gam.predict_proba(X_train)#[::,1]
        if parameters['testN']>0:
            X_test = sc.transform(test[nomi])
            prob_predic=gam.predict_proba(X_test)#[::,1]
            test['SI']=prob_predic
        train['SI']=prob_fit
        return(train,test)
    
    ####################################
    
    def LR_cv(classifier,X,y,train,test):
        classifier.fit(X[train], y[train])
        prob_predic=classifier.predict_proba(X[test])[::,1]
        regression_coeff=classifier.coef_
        regression_intercept=classifier.intercept_
        coeff=np.hstack((regression_intercept,regression_coeff[0]))
        print(coeff,'regression coeff')
        #prob_fit=classifier.predict_proba(X[train])[::,1]
        return prob_predic,coeff
    
    def DT_cv(classifier,X,y,train,test):
        classifier.fit(X[train], y[train])
        prob_predic=classifier.predict_proba(X[test])[::,1]
        # regression_coeff=classifier.coef_
        # regression_intercept=classifier.intercept_
        # coeff=np.hstack((regression_intercept,regression_coeff[0]))
        # print(coeff,'regression coeff')
        return prob_predic,None
    
    def RF_cv(classifier,X,y,train,test):
        classifier.fit(X[train], y[train])
        prob_predic=classifier.predict_proba(X[test])[::,1]
        # regression_coeff=classifier.coef_
        # regression_intercept=classifier.intercept_
        # coeff=np.hstack((regression_intercept,regression_coeff[0]))
        # print(coeff,'regression coeff')
        return prob_predic,None
    
    def SVC_cv(classifier,X,y,train,test):
        classifier.fit(X[train], y[train])
        prob_predic=classifier.predict_proba(X[test])[::,1]
        regression_coeff=classifier.coef_
        regression_intercept=classifier.intercept_
        coeff=np.hstack((regression_intercept,regression_coeff[0]))
        print(coeff,'regression coeff')
        return prob_predic,coeff
    
    def fr_cv(train,test,frame,nomes,txt):
        df=frame.loc[train,:]
        test=frame.loc[test,:]
        nomi=nomes
        Npx1=None
        Npx2=None
        Npx3=None
        Npx4=None
        file = open(txt,'w')#################save W+, W- and Wf
        file.write('covariate,class,Npx1,Npx2,Npx3,Npx4,Wf\n')
        for ii in nomi:
            classi=df[ii].unique()
            for i in classi:
                dd=pd.DataFrame()
                dd = df.apply(lambda x : True if x['y'] == 1 and x[ii] == i else False, axis = 1)
                Npx1 = len(dd[dd == True].index)
                dd=pd.DataFrame()
                dd = df.apply(lambda x : True if x[ii] == i else False, axis = 1)
                Npx2 = len(dd[dd == True].index)
                dd=pd.DataFrame()
                dd = df.apply(lambda x : True if x['y'] == 1 else False, axis = 1)
                Npx3 = len(dd[dd == True].index)
                dd=pd.DataFrame()
                Npx4 = df.shape[0]#len(dd[dd == True].index)
                if Npx1==0 or Npx3==0:
                    Wf=0.
                else:
                    Wf=(np.divide((np.divide(Npx1,Npx2)),(np.divide(Npx3,Npx4))))
                var=[ii,i,Npx1,Npx2,Npx3,Npx4,Wf]
                file.write(','.join(str(e) for e in var)+'\n')#################save W+, W- and Wf
                df[ii][df[ii]==i]=float(Wf)
                test[ii][test[ii]==i]=float(Wf)
        file.close()
        df['SI']=df[nomi].sum(axis=1)
        test['SI']=test[nomi].sum(axis=1)
        return(test['SI'],None)
    
    def woe_cv(train,test,frame,nomes,txt):
        df=frame.loc[train,:]
        test=frame.loc[test,:]
        nomi=nomes
        Npx1=None
        Npx2=None
        Npx3=None
        Npx4=None
        file = open(txt,'w')#################save W+, W- and Wf
        file.write('covariate,class,Npx1,Npx2,Npx3,Npx4,W+,W-,Wf\n')
        for ii in nomi:
            classi=df[ii].unique()
            for i in classi:
                dd=pd.DataFrame()
                dd = df.apply(lambda x : True if x['y'] == 1 and x[ii] == i else False, axis = 1)
                Npx1 = len(dd[dd == True].index)
                dd=pd.DataFrame()
                dd = df.apply(lambda x : True if x['y'] == 1 and x[ii] != i else False, axis = 1)
                Npx2 = len(dd[dd == True].index)
                dd=pd.DataFrame()
                dd = df.apply(lambda x : True if x['y'] == 0 and x[ii] == i else False, axis = 1)
                Npx3 = len(dd[dd == True].index)
                dd=pd.DataFrame()
                dd = df.apply(lambda x : True if x['y'] == 0 and x[ii] != i else False, axis = 1)
                Npx4 = len(dd[dd == True].index)
                if Npx1==0 or Npx3==0:
                    Wplus=0.
                else:
                    Wplus=math.log((Npx1/(Npx1+Npx2))/(Npx3/(Npx3+Npx4)))
                if Npx2==0 or Npx4==0:
                    Wminus=0.
                else:
                    Wminus=math.log((Npx2/(Npx1+Npx2))/(Npx4/(Npx3+Npx4)))
                Wf=Wplus-Wminus
                var=[ii,i,Npx1,Npx2,Npx3,Npx4,Wplus,Wminus,Wf]
                file.write(','.join(str(e) for e in var)+'\n')#################save W+, W- and Wf
                df[ii][df[ii]==i]=float(Wf)
                test[ii][test[ii]==i]=float(Wf)
            #df.to_csv(self.f+'/file'+ii+'.csv')
        file.close()
        df['SI']=df[nomi].sum(axis=1)
        test['SI']=test[nomi].sum(axis=1)
        return(test['SI'],None)
    
    def GAM_cv(classifier,X,y,train,test,splines=None,dtypes=None,nomi=None,df=None,fold=None,filename=None):
        lams = np.empty(len(nomi))
        lams.fill(0.5)
        gam = classifier(splines, dtype=dtypes)
        gam.gridsearch(X[train,], y[train], lam=lams)
        GAM_utils.GAM_plot(gam,df,nomi,fold,filename)
        GAM_utils.GAM_save(gam,fold,filename)
        prob_predic=gam.predict_proba(X[test])#[::,1]
        return prob_predic,None
    
class CV_utils():
    def cross_validation(parameters,algorithm,classifier):
        df=parameters['df']
        nomi=parameters['nomi']
        x=df[parameters['field1']]
        y=df['y']
        sc = StandardScaler()#####scaler
        X = sc.fit_transform(x)
        train_ind={}
        test_ind={}
        prob={}
        cofl=[]
        df["SI"] = np.nan
        if parameters['testN']>1:
            cv = StratifiedKFold(n_splits=parameters['testN'])
            for i, (train, test) in enumerate(cv.split(X, y)):
                train_ind[i]=train
                test_ind[i]=test
                if algorithm==Algorithms.GAM_cv:
                    prob[i],coeff=algorithm(classifier,X,y,train,test,splines=parameters['splines'],dtypes=parameters['dtypes'],nomi=nomi,df=df,fold=parameters['fold'],filename=str(i))
                else:
                    prob[i],coeff=algorithm(classifier,X,y,train,test)
                df.loc[test,'SI']=prob[i]
                cofl.append(coeff)
        elif parameters['testN']==1:
            train=np.arange(len(y))
            test=np.arange(len(y))
            if algorithm==Algorithms.GAM_cv:
                prob[0],coeff=algorithm(classifier,X,y,train,test,splines=parameters['splines'],dtypes=parameters['dtypes'],nomi=nomi,df=df,fold=parameters['fold'],filename='')
            else:
                prob[0],coeff=algorithm(classifier,X,y,train,test)
            df.loc[test,'SI']=prob[0]
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
                print(splines)
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

        print(splines)
        return splines,dtypes
    
    def GAM_plot(gam,df,nomi,fold,filename):
        GAM_sel=nomi
        fig = plt.figure(figsize=(20, 25))

        maX=[]
        miN=[]
        for i, term in enumerate(gam.terms):
            if term.isintercept:
                continue
            pdep0, confi0 = gam.partial_dependence(term=i, X=gam.generate_X_grid(term=i), width=0.95)
            if isinstance(gam.terms[i], terms.FactorTerm):
                continue
            maX=maX+[max(pdep0)]
            miN=miN+[np.min(pdep0)]
        MAX=max(maX)+1
        MIN=min(miN)-2

        for i, term in enumerate(gam.terms):
            if term.isintercept:
                continue

            XX = gam.generate_X_grid(term=i)
            pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
            plt.subplot(4, 3, i+1)

            if isinstance(gam.terms[i], terms.FactorTerm):
                plt.plot(np.sort(df[GAM_sel[i]].unique()),
                        pdep[range(1, len(pdep), 3)], 'o', c='blue')
                for j in pdep[range(1, len(pdep), 3)]:
                    if j>1.5:
                        plt.axvline(np.sort(df[GAM_sel[0]].unique())[np.where(pdep[range(1, len(pdep), 3)] == j)[0][0]],
                                    color='k', linewidth=0.5, linestyle="--")
                plt.xticks(np.sort(df[GAM_sel[i]].unique()), rotation=90)
                plt.xlabel(GAM_sel[i])
                plt.ylabel('Regression coefficient')
                continue

            plt.plot(XX[:, term.feature], pdep, c='blue')
            # plt.plot(XX[:, term.feature], confi, c='r', ls='--')
            plt.fill_between(XX[:, term.feature].ravel(), y1=confi[:,0], y2=confi[:,1], color='gray', alpha=0.2)

            plt.xlabel(GAM_sel[i])
            plt.ylabel('Regression coefficient')
            plt.ylim(MIN,MAX)
        plt.savefig(fold+'/Model_covariates'+filename+'.pdf', bbox_inches='tight')
        #plt.show()
        
    def GAM_save(gam,fold,filename=None):
        filename_pkl = fold+'/gam_coeff'+filename+'.pkl'
        #filename_txt = parameters['fold']+'/gam_coeff.txt'

        with open(filename_pkl, 'wb') as filez:
            pickle.dump(gam, filez)
        
        #with open(filename_pkl, 'rb') as file_pickle:
        #    loaded_data = pickle.load(file_pickle)

        #with open(filename_txt, 'wb') as file_txt:
        #    json.dump(loaded_data, file_txt, indent=2)
    
    

    

    
