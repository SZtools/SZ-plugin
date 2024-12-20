#!/usr/bin/python
#coding=utf-8
"""
/***************************************************************************
        begin                : 2021-11
        copyright            : (C) 2024 by Giacomo Titti,Bologna, November 2024
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
    Copyright (C) 2024 by Giacomo Titti, Bologna, November 2024

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 ***************************************************************************/
"""

__author__ = 'Giacomo Titti'
__date__ = '2024-11-01'
__copyright__ = '(C) 2024 by Giacomo Titti'

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,LeaveOneOut,TimeSeriesSplit,KFold
import pandas as pd
import numpy as np
import pickle
from collections import OrderedDict
from pygam import terms
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from shapely.wkt import loads
from sklearn.tree import export_text

class Algorithms():

    def NN_transfer(parameters):
        nomi=parameters['nomi']
        df=parameters['df']
        df_scaled=CV_utils.scaler(df,nomi,'standard')
        if parameters['family']=='MLP_classifier':
            prob_predic=parameters['predictors_weights'].predict_proba(df_scaled.loc[:,nomi].to_numpy())[::,1]
        else:
            prob_predic=parameters['predictors_weights'].predict(df_scaled.loc[:,nomi].to_numpy())
        df['SI']=prob_predic
        return df

    def ML_transfer(parameters):
        nomi=parameters['nomi']
        df=parameters['df']
        df_scaled=CV_utils.scaler(df,nomi,'standard')
        prob_predic=parameters['predictors_weights'].predict_proba(df_scaled.loc[:,nomi].to_numpy())[::,1]
        df['SI']=prob_predic
        return df
    
    def GAM_transfer(parameters):
        nomi=parameters['nomi']
        df=parameters['df']
        df_scaled=CV_utils.scaler(df,parameters['linear']+parameters['continuous'],'custom')
        if parameters['family']=='binomial':
            prob_fit=parameters['predictors_weights'].predict_proba(df_scaled[nomi])
            df['SI']=prob_fit
        else:
            prob_fit=parameters['predictors_weights'].predict(df_scaled[nomi])
            df['SI']=prob_fit
        return(df)
    
    def alg_NNrun(classifier,X,y,train,test,df,fold,nomi,filename='',family=None):
        classifier.fit(X.loc[train,nomi].to_numpy(), y.iloc[train].to_numpy())
        if family=='MLP_classifier':
            prob_predic=classifier.predict_proba(X.loc[test,nomi].to_numpy())[::,1]
        elif family=='MLP_regressor':
            prob_predic=classifier.predict(X.loc[test,nomi].to_numpy())
        NN_utils.NN_plot(classifier,fold,filename)
        return prob_predic,classifier

    def alg_MLrun(classifier,X,y,train,test,df,fold,nomi,filename=''):
        classifier.fit(X.loc[train,nomi].to_numpy(), y.iloc[train].to_numpy())
        prob_predic=classifier.predict_proba(X.loc[test,nomi].to_numpy())[::,1]
        ML_utils.ML_save(classifier,fold,nomi,filename)
        return prob_predic,classifier

    def alg_GAMrun(classifier,X,y,train,test,df,splines=None,dtypes=None,nomi=None,fold=None,filename='',family=None):
        lams = np.empty(len(nomi))
        lams.fill(0.5)
        classifier_selected=classifier[family]
        gam = classifier_selected(splines, dtype=dtypes)
        gam.gridsearch(X.loc[train,nomi].to_numpy(), y.iloc[train].to_numpy(), lam=lams,progress=False)
        if family=='binomial':
            prob=gam.predict_proba(X.loc[test,nomi].to_numpy())
        else:
            prob=gam.predict(X.loc[test,nomi].to_numpy())
        GAM_utils.GAM_plot(gam,df.loc[train,nomi],nomi,fold,filename,X.loc[train,nomi])
        GAM_utils.GAM_save(gam,fold,filename)
        CI=[]
        return prob,CI,gam
    
class CV_utils():

    def cross_validation(parameters,algorithm,classifier):
        df=parameters['df']
        nomi=parameters['nomi']
        x=df[parameters['nomi']]
        y=df['y']
        if algorithm==Algorithms.alg_GAMrun:
            df_scaled=CV_utils.scaler(x,parameters['linear']+parameters['continuous']+parameters['tensor'],'custom')
        else:
            df_scaled=CV_utils.scaler(df,nomi,'standard')
        train_ind={}
        test_ind={}
        prob={}
        CI={}
        cofl=[]
        df["SI"] = np.nan
        coeff=None
        if parameters['cv_method']=='temporal_TSS' or parameters['cv_method']=='temporal_LOO' or parameters['cv_method']=='spacetime_LOO':
            train_ind,test_ind,iters_count = CV_utils.cv_method(parameters,df_scaled,df,parameters['nomi'])
            for i in range(iters_count):
                print('cv: ',i)
                if algorithm==Algorithms.alg_GAMrun:
                    prob[i],CI[i],predictors_weights=algorithm(classifier,df_scaled,y,train_ind[i],test_ind[i],df,splines=parameters['splines'],dtypes=parameters['dtypes'],nomi=nomi,fold=parameters['fold'],filename=str(i),family=parameters['family'])
                elif algorithm==Algorithms.alg_NNrun:
                    prob[i],predictors_weights=algorithm(classifier,df_scaled,y,train_ind[i],test_ind[i],df,fold=parameters['fold'],nomi=nomi,filename=str(i),family=parameters['family'])
                else:
                    prob[i],predictors_weights=algorithm(classifier,df_scaled,y,train_ind[i],test_ind[i],df,fold=parameters['fold'],nomi=nomi,filename=str(i))
                    gam=None
                df.loc[test_ind[i],'SI']=prob[i]
        else:
            if parameters['testN']>1:
                train_ind,test_ind,iters_count = CV_utils.cv_method(parameters,df_scaled,df,parameters['nomi'])
                for i in range(iters_count):
                    print('cv: ',i)
                    if algorithm==Algorithms.alg_GAMrun:
                        prob[i],CI[i],predictors_weights=algorithm(classifier,df_scaled,y,train_ind[i],test_ind[i],df,splines=parameters['splines'],dtypes=parameters['dtypes'],nomi=nomi,fold=parameters['fold'],filename=str(i),family=parameters['family'])
                    elif algorithm==Algorithms.alg_NNrun:
                        prob[i],predictors_weights=algorithm(classifier,df_scaled,y,train_ind[i],test_ind[i],df,fold=parameters['fold'],nomi=nomi,filename=str(i),family=parameters['family'])
                    else:
                        prob[i],predictors_weights=algorithm(classifier,df_scaled,y,train_ind[i],test_ind[i],df,fold=parameters['fold'],nomi=nomi,filename=str(i))
                        gam=None
                    df.loc[test_ind[i],'SI']=prob[i]
            elif parameters['testN']==1:
                train=np.arange(len(y))
                test=np.arange(len(y))
                if algorithm==Algorithms.alg_GAMrun:
                    prob[0],CI[0],predictors_weights=algorithm(classifier,df_scaled,y,train,test,df,splines=parameters['splines'],dtypes=parameters['dtypes'],nomi=nomi,fold=parameters['fold'],family=parameters['family'])
                elif algorithm==Algorithms.alg_NNrun:
                    prob[0],predictors_weights=algorithm(classifier,df_scaled,y,train,test,df,fold=parameters['fold'],nomi=nomi,family=parameters['family'])
                else:
                    prob[0],predictors_weights=algorithm(classifier,df_scaled,y,train,test,df,fold=parameters['fold'],nomi=nomi)
                df.loc[test,'SI']=prob[0]
                
                test_ind[0]=test
        return prob,test_ind,predictors_weights
    
    def cv_method(parameters,df_scaled,df,nomi):
        X_train={}
        X_test={}
        y = df['y'].to_numpy()
        if parameters['cv_method']=='spatial':
            kmeans = CV_utils.kmeans_clustering(parameters,df,df_scaled)
            method = LeaveOneOut()
            for i, (train, test) in enumerate(method.split(np.arange(parameters['testN']))):
                X_train[i] = np.where(kmeans.labels_ != test[0])[0]
                X_test[i] = np.where(kmeans.labels_ == test[0])[0]
        elif parameters['cv_method']=='random':
            if parameters['family']=='gaussian':
                method=KFold(n_splits=parameters['testN'],shuffle=True)
                for i, (train, test) in enumerate(method.split(df_scaled, y)):
                    X_train[i]=train
                    X_test[i]=test
            else:
                method=StratifiedKFold(n_splits=parameters['testN'])
                for i, (train, test) in enumerate(method.split(df_scaled, y)):
                    X_train[i]=train
                    X_test[i]=test
        elif parameters['cv_method']=='temporal_TSS':
            time_index=sorted(df[parameters['time']].unique())
            method=TimeSeriesSplit(n_splits=len(time_index)-1)
            for i, (train, test) in enumerate(method.split(time_index)):
                X_train[i]=np.where(df[parameters['time']].isin([time_index[ii] for ii in train]))[0]
                X_test[i]=np.where(df[parameters['time']].isin([time_index[ii] for ii in test]))[0]
        elif parameters['cv_method']=='temporal_LOO':
            time_index=sorted(df[parameters['time']].unique())
            method = LeaveOneOut()
            for i, (train, test) in enumerate(method.split(time_index)):
                X_train[i]=np.where(df[parameters['time']] != time_index[test[0]])[0]
                X_test[i]=np.where(df[parameters['time']] == time_index[test[0]])[0]
        elif parameters['cv_method']=='spacetime_LOO':
            kmeans = CV_utils.kmeans_clustering(parameters,df,df_scaled)
            time_index=sorted(df[parameters['time']].unique())
            method = LeaveOneOut()
            count=0
            for ii, (train_time, test_time) in enumerate(method.split(time_index)):
                X_test_time_index=np.where(df[parameters['time']] == time_index[test_time[0]])[0]
                for i, (train, test) in enumerate(method.split(np.arange(parameters['testN']))):
                    X_test_space_index = np.where(kmeans.labels_ == test[0])[0]
                    X_test[count]=np.intersect1d(X_test_time_index, X_test_space_index)
                    mask = ~np.isin(np.arange(len(df)), X_test[count])
                    X_train[count] = np.arange(len(df))[mask]
                    count+=1
        return X_train,X_test,len(X_test)
    
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
        return df_scaled
    
    def kmeans_clustering(parameters,df,df_scaled):
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
        return kmeans

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
            elif GAM_sel[i] in parameters['tensor']:
                dtypes = dtypes + ['numerical']
                vars_dict[i]={'term':'te'}       
        tensor_id=[]
        splines = terms.TermList()
        for i,v in enumerate(vars_dict .keys()):
            if vars_dict[v]['term'] == 's':
                term = terms.SplineTerm(i, n_splines=vars_dict[v].get('n_splines', 10))
            elif vars_dict[v]['term'] == 'l':
                term = terms.LinearTerm(i)
            elif vars_dict[v]['term'] == 'f':
                term = terms.FactorTerm(i)
            elif vars_dict[v]['term'] == 'te':
                tensor_id=tensor_id + [i]
                if len(tensor_id)==2:
                    term=terms.TensorTerm(tensor_id[0],tensor_id[1]) 
            splines += term
        return splines,dtypes
    
    def GAM_plot(gam,df,nomi,fold,filename,scaled_df):
        print('plotting covariates.....')
        GAM_sel=nomi
        maX=[]
        miN=[]
        for i, term in enumerate(gam.terms):
            if term.isintercept:
                continue
            elif isinstance(gam.terms[i], terms.FactorTerm):
                continue
            elif isinstance(gam.terms[i], terms.TensorTerm):
                continue
            elif isinstance(gam.terms[i], terms.LinearTerm) or isinstance(gam.terms[i], terms.SplineTerm):
                pdep0, confi0 = gam.partial_dependence(term=i, X=gam.generate_X_grid(term=i), width=0.95)
            maX=maX+[np.max(confi0[:,1])]
            miN=miN+[np.min(confi0[:,0])]
        MAX=max(maX)+0.2
        MIN=min(miN)-0.2
        count=0
        countIns=0
        for i, term in enumerate(gam.terms):
            if term.isintercept:
                continue
            if isinstance(gam.terms[i], terms.TensorTerm):
                continue
            count+=1
        count=count+countIns
        if int(np.ceil(count/3.))<4:
            rows=4
        else:
            rows=int(np.ceil(count/3.))
        
        ########################################################################not scaled plot

        fig = plt.figure(figsize=(15,15))
        for i, term in enumerate(gam.terms):
            if term.isintercept:
                continue
            X=np.array([min(df.iloc[:, i])])
            m=np.min(df.iloc[:, i])
            interval=(np.max(df.iloc[:, i])-np.min(df.iloc[:, i]))/(len(df[GAM_sel[i]])-1)
            for n in range(len(df[GAM_sel[i]])-1):
                X=np.append(X,m+interval)
                m=m+interval
            if isinstance(gam.terms[i], terms.FactorTerm):
                ax=fig.add_subplot(rows, 3, i+1)   
                XX = gam.generate_X_grid(term=i,n=len(df[GAM_sel[i]]))
                pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
                x=df[GAM_sel[i]].unique()
                y=[]
                y1=[]
                y2=[]
                for j in range(len(df[GAM_sel[i]].unique())):
                    pdep_unq=np.unique(pdep)
                    confi025_unq=np.unique(confi[:,0])
                    confi095_unq=np.unique(confi[:,1])
                    y.append(pdep_unq[j])
                    y1.append(confi025_unq[j])
                    y2.append(confi095_unq[j])
                paired_xy = list(zip(x, y, y1, y2))
                paired_vectors_sorted = sorted(paired_xy, key=lambda x: x[0])
                x, y, y1, y2= zip(*paired_vectors_sorted)
                ax.plot(x,y1,'o', c='gray')
                ax.plot(x,y2,'o', c='gray')
                ax.plot(x,y,'o', c='blue')
                ax.set_xticks(np.sort(df[GAM_sel[i]].unique()))
                ax.set_xlabel(GAM_sel[i])
                ax.set_ylabel('Partial Effect')
                ax.set_ylim(MIN,MAX)
                continue
            
            elif isinstance(gam.terms[i], terms.LinearTerm):
                ax=fig.add_subplot(rows, 3, i+1)   
                XX = gam.generate_X_grid(term=i,n=len(df[GAM_sel[i]]))
                pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
                ax.plot(X, pdep, c='blue')                
                ax.fill_between(X.ravel(), y1=confi[:,0], y2=confi[:,1], color='gray', alpha=0.2)
                ax.set_xlabel(GAM_sel[i])
                ax.set_ylabel('Partial Effect')
                ax.set_ylim(MIN,MAX)
                continue
            elif isinstance(gam.terms[i], terms.TensorTerm):
                X0=np.array([min(df.iloc[:, i])])
                m=np.min(df.iloc[:, i])
                interval=(np.max(df.iloc[:, i])-np.min(df.iloc[:, i]))/(100-1)
                for n in range(100-1):
                    X0=np.append(X0,m+interval)
                    m=m+interval
                X1=np.array([min(df.iloc[:, i+1])])
                m=np.min(df.iloc[:, i+1])
                interval=(np.max(df.iloc[:, i+1])-np.min(df.iloc[:, i+1]))/(100-1)
                for n in range(100-1):
                    X1=np.append(X1,m+interval)
                    m=m+interval
                fig2 = plt.figure(figsize=(8, 8))
                XX = gam.generate_X_grid(term=i, meshgrid=True)
                Z = gam.partial_dependence(term=i, X=XX, meshgrid=True)
                ax3d = fig2.subplots()
                mesh = ax3d.pcolormesh(X0, X1, np.transpose(Z), cmap='viridis', shading='auto')
                ax3d.set_aspect('equal', adjustable='datalim')
                bbox = ax3d.get_position()
                x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
                width = x1 - x0
                height = y1 - y0
                cbar_ax = fig2.add_axes([x1+0.03, y0, 0.03, height])
                cbar = plt.colorbar(mesh, cax=cbar_ax)
                cbar.set_label('Partial Effect', fontsize=16)
                cbar.ax.tick_params(labelsize=16)
                ax3d.set_xlabel(GAM_sel[i], fontsize=16)
                ax3d.set_ylabel(GAM_sel[i + 1], fontsize=16)
                ax3d.tick_params(labelsize=14)
                fig2.savefig(fold+'/Model_covariates_interaction'+filename+'.pdf', bbox_inches='tight') 
                continue

            elif isinstance(gam.terms[i], terms.SplineTerm):
                ax=fig.add_subplot(rows, 3, i+1)   
                XX = gam.generate_X_grid(term=i,n=len(df[GAM_sel[i]]))
                pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
                ax.plot(X, pdep, c='blue')
                ax.fill_between(X.ravel(), y1=confi[:,0], y2=confi[:,1], color='gray', alpha=0.2)
                ax.set_xlabel(GAM_sel[i])
                ax.set_ylabel('Partial Effect')
                ax.set_ylim(MIN,MAX)
                continue
        fig.savefig(fold+'/Model_covariates'+filename+'.pdf', bbox_inches='tight')

        ########################################################################scaled plot
        fig1 = plt.figure(figsize=(15,15))
        for i, term in enumerate(gam.terms):
            if term.isintercept:
                continue
            if isinstance(gam.terms[i], terms.FactorTerm):
                ax1=fig1.add_subplot(rows, 3, i+1)  
                XX = gam.generate_X_grid(term=i,n=len(df[GAM_sel[i]]))
                pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
                x=df[GAM_sel[i]].unique()
                y=[]
                y1=[]
                y2=[]
                for j in range(len(df[GAM_sel[i]].unique())):
                    pdep_unq=np.unique(pdep)
                    confi025_unq=np.unique(confi[:,0])
                    confi095_unq=np.unique(confi[:,1])
                    y.append(pdep_unq[j])
                    y1.append(confi025_unq[j])
                    y2.append(confi095_unq[j])
                paired_xy = list(zip(x, y, y1, y2))
                paired_vectors_sorted = sorted(paired_xy, key=lambda x: x[0])
                x, y, y1, y2= zip(*paired_vectors_sorted)
                ax1.plot(x,y1,'o', c='gray')
                ax1.plot(x,y2,'o', c='gray')
                ax1.plot(x,y,'o', c='blue')
                ax1.set_xticks(np.sort(df[GAM_sel[i]].unique()))
                ax1.set_xlabel(GAM_sel[i])
                ax1.set_ylabel('Partial Effect')
                ax1.set_ylim(MIN,MAX)
                continue
            elif isinstance(gam.terms[i], terms.LinearTerm):
                ax1=fig1.add_subplot(rows, 3, i+1)
                XX = gam.generate_X_grid(term=i,n=len(df[GAM_sel[i]]))
                pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
                ax1.plot(XX[:, term.feature], pdep, c='blue')
                ax1.fill_between(XX[:, term.feature].ravel(), y1=confi[:,0], y2=confi[:,1], color='gray', alpha=0.2)
                ax1.set_xlabel(GAM_sel[i])
                ax1.set_ylabel('Partial Effect')
                ax1.set_ylim(MIN,MAX)
                continue
            elif isinstance(gam.terms[i], terms.TensorTerm):
                fig3 = plt.figure(figsize=(8, 8))
                XX = gam.generate_X_grid(term=i, meshgrid=True)
                Z = gam.partial_dependence(term=i, X=XX, meshgrid=True)
                ax3d = fig3.subplots()
                mesh = ax3d.pcolormesh(XX[0], XX[1], Z, cmap='viridis', shading='auto')
                ax3d.set_aspect('equal', adjustable='datalim')
                bbox = ax3d.get_position()
                x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
                width = x1 - x0
                height = y1 - y0
                cbar_ax = fig3.add_axes([x1+0.03, y0, 0.03, height])
                cbar = plt.colorbar(mesh, cax=cbar_ax)
                cbar.set_label('Partial Effect', fontsize=16)
                cbar.ax.tick_params(labelsize=16)
                ax3d.set_xlabel(GAM_sel[i], fontsize=16)
                ax3d.set_ylabel(GAM_sel[i + 1], fontsize=16)
                ax3d.tick_params(labelsize=14)
                fig3.savefig(fold + '/Model_covariates_interaction_scaled' + filename + '.pdf', bbox_inches='tight')
                continue
            elif isinstance(gam.terms[i], terms.SplineTerm):
                ax1=fig1.add_subplot(rows, 3, i+1)  
                XX = gam.generate_X_grid(term=i,n=len(df[GAM_sel[i]]))
                pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
                ax1.plot(XX[:, term.feature], pdep, c='blue')
                ax1.fill_between(XX[:, term.feature].ravel(), y1=confi[:,0], y2=confi[:,1], color='gray', alpha=0.2)
                ax1.set_xlabel(GAM_sel[i])
                ax1.set_ylabel('Partial Effect')
                ax1.set_ylim(MIN,MAX)
                continue
        fig1.savefig(fold+'/Model_covariates_scaled'+filename+'.pdf', bbox_inches='tight')
        del gam
        del df

    def GAM_save(gam,fold,filename=''):
        print('saving gam.pkl.....')
        filename_pkl = fold+'/gam_coeff'+filename+'.pkl'
        with open(filename_pkl, 'wb') as filez:
            pickle.dump(gam, filez)
        del gam

class ML_utils():
    def ML_save(classifier,fold,nomi, filename):
        try:#RF,DT
            regression_coeff=classifier.feature_importances_
            coeff=regression_coeff
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

class NN_utils():
    def NN_plot(NNclassifier,fold,filename):
        plt.figure(figsize=(10, 6))
        plt.plot(NNclassifier.loss_curve_)
        plt.plot(NNclassifier.validation_scores_)
        plt.xlabel('Iterations',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.grid()
        plt.legend(['Train','Test'],prop={'size': 16})
        plt.savefig(fold+'/loss_curve'+filename+'.pdf', bbox_inches='tight')