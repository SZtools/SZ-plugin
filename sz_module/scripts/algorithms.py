from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import math
from pygam import LogisticGAM
import pickle
import os


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
        prob_fit=gam.predict_proba(X_train)[::,1]

        if parameters['testN']>0:
            X_test = sc.transform(test[nomi])
            prob_predic=gam.predict_proba(X_test)[::,1]
            test['SI']=prob_predic
        train['SI']=prob_fit
        if not os.path.exists(parameters['fold']):
            os.mkdir(parameters['fold'])
        filename = parameters['fold']+'/gam_coeff.pkl'

        with open(filename, 'wb') as filez:
            pickle.dump(gam, filez)
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
    
    def GAM_cv(classifier,X,y,train,test,splines=None,dtypes=None,nomi=None):
        lams = np.empty(len(nomi))
        lams.fill(0.5)
        gam = classifier(splines, dtypes)
        gam.gridsearch(X[train], y[train], lam=lams)
        prob_predic=gam.predict_proba(X[test])[::,1]
        return prob_predic,None

    

    
