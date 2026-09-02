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

import os
from sklearn.base import clone
from sklearn.pipeline import Pipeline
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
        if parameters['family']=='MLP_classifier':
            prob_predic=parameters['predictors_weights'].predict_proba(df.loc[:,nomi].to_numpy())[::,1]
        else:
            prob_predic=parameters['predictors_weights'].predict(df.loc[:,nomi].to_numpy())
        df['SI']=prob_predic
        return df

    def ML_transfer(parameters):
        nomi=parameters['nomi']
        df=parameters['df']
        family=parameters['family']
        if family=='SVM_classifier' or family=='RF_classifier' or family=='DT_classifier':
            prob_predic=parameters['predictors_weights'].predict_proba(df.loc[:,nomi].to_numpy())[::,1]
        else:
            prob_predic=parameters['predictors_weights'].predict(df.loc[:,nomi].to_numpy())
        df['SI']=prob_predic
        return df
    
    def GAM_transfer(parameters):
        nomi=parameters['nomi']
        df=parameters['df']
        model_artifact=parameters['predictors_weights']
        missing_columns=[column for column in nomi if column not in df.columns]
        if missing_columns:
            raise ValueError(
                'The prediction layer is missing these GAM predictors: '
                + ', '.join(missing_columns)
            )
        df_scaled=CV_utils.apply_custom_scaler(df,model_artifact['scaler'])
        gam=model_artifact['estimator']
        if parameters['family']=='binomial':
            prob_fit=gam.predict_proba(df_scaled[nomi])
            df['SI']=prob_fit
        else:
            prob_fit=gam.predict(df_scaled[nomi])
            df['SI']=prob_fit
        return df
    
    def alg_NNrun(classifier,X,y,train,test,df,fold,nomi,filename='',family=None,use_scaler=True):
        estimator=clone(classifier)
        if use_scaler:
            model=Pipeline([
                ('scaler', StandardScaler()),
                ('estimator', estimator),
            ])
        else:
            model=estimator
        model.fit(X.loc[train,nomi].to_numpy(), y.iloc[train].to_numpy())
        if family=='MLP_classifier':
            prob_predic=model.predict_proba(X.loc[test,nomi].to_numpy())[::,1]
        elif family=='MLP_regressor':
            prob_predic=model.predict(X.loc[test,nomi].to_numpy())
        NN_utils.NN_plot(estimator,fold,filename)
        return prob_predic,model

    def alg_MLrun(classifier,X,y,train,test,df,fold,nomi,filename='',family=None,use_scaler=True):
        estimator=clone(classifier)
        if use_scaler:
            model=Pipeline([
                ('scaler', StandardScaler()),
                ('estimator', estimator),
            ])
        else:
            model=estimator
        model.fit(X.loc[train,nomi].to_numpy(), y.iloc[train].to_numpy())
        if family=='SVM_classifier' or family=='RF_classifier' or family=='DT_classifier':
            prob_predic=model.predict_proba(X.loc[test,nomi].to_numpy())[::,1]
        else:
            prob_predic=model.predict(X.loc[test,nomi].to_numpy())
        ML_utils.ML_save(estimator,fold,nomi,filename)
        return prob_predic,model

    def alg_GAMrun(classifier,X,y,train,test,df,splines=None,dtypes=None,nomi=None,fold=None,filename='',family=None,scale_columns=None,use_scaler=True):
        lams = np.empty(len(nomi))
        lams.fill(0.5)
        scaler=(
            CV_utils.fit_custom_scaler(X.loc[train],scale_columns or [])
            if use_scaler else {}
        )
        X_train=CV_utils.apply_custom_scaler(X.loc[train],scaler)
        X_test=CV_utils.apply_custom_scaler(X.loc[test],scaler)
        classifier_selected=classifier[family]
        gam = classifier_selected(splines, dtype=dtypes)
        gam.gridsearch(X_train.loc[:,nomi].to_numpy(), y.iloc[train].to_numpy(), lam=lams,progress=False)
        if family=='binomial':
            prob=gam.predict_proba(X_test.loc[:,nomi].to_numpy())
        else:
            prob=gam.predict(X_test.loc[:,nomi].to_numpy())
        model_artifact={
            'estimator':gam,
            'scaler':scaler,
            'feature_names':list(nomi),
            'family':family,
        }
        GAM_utils.GAM_plot(
            gam,
            df.loc[train,nomi],
            nomi,
            fold,
            filename,
            X_train.loc[:,nomi] if use_scaler else None,
        )
        GAM_utils.GAM_save(model_artifact,fold,filename)
        CI=[]
        return prob,CI,model_artifact
    
class CV_utils():

    def cross_validation(parameters,algorithm,classifier):
        df=parameters['df']
        nomi=parameters['nomi']
        y=df['y']
        df_scaled=df.copy()
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
                    prob[i],CI[i],predictors_weights=algorithm(classifier,df_scaled,y,train_ind[i],test_ind[i],df,splines=parameters['splines'],dtypes=parameters['dtypes'],nomi=nomi,fold=parameters['fold'],filename=str(i),family=parameters['family'],scale_columns=parameters['linear']+parameters['continuous'],use_scaler=parameters.get('feature_scaling',True))
                elif algorithm==Algorithms.alg_NNrun:
                    prob[i],predictors_weights=algorithm(classifier,df_scaled,y,train_ind[i],test_ind[i],df,fold=parameters['fold'],nomi=nomi,filename=str(i),family=parameters['family'],use_scaler=parameters.get('feature_scaling',True))
                elif algorithm==Algorithms.alg_MLrun:
                    prob[i],predictors_weights=algorithm(classifier,df_scaled,y,train_ind[i],test_ind[i],df,fold=parameters['fold'],nomi=nomi,filename=str(i),family=parameters['family'],use_scaler=parameters.get('feature_scaling',True))
                    gam=None
                df.loc[test_ind[i],'SI']=prob[i]
        else:
            if parameters['testN']>1:
                train_ind,test_ind,iters_count = CV_utils.cv_method(parameters,df_scaled,df,parameters['nomi'])
                for i in range(iters_count):
                    print('cv: ',i)
                    if algorithm==Algorithms.alg_GAMrun:
                        prob[i],CI[i],predictors_weights=algorithm(classifier,df_scaled,y,train_ind[i],test_ind[i],df,splines=parameters['splines'],dtypes=parameters['dtypes'],nomi=nomi,fold=parameters['fold'],filename=str(i),family=parameters['family'],scale_columns=parameters['linear']+parameters['continuous'],use_scaler=parameters.get('feature_scaling',True))
                    elif algorithm==Algorithms.alg_NNrun:
                        prob[i],predictors_weights=algorithm(classifier,df_scaled,y,train_ind[i],test_ind[i],df,fold=parameters['fold'],nomi=nomi,filename=str(i),family=parameters['family'],use_scaler=parameters.get('feature_scaling',True))
                    elif algorithm==Algorithms.alg_MLrun:
                        prob[i],predictors_weights=algorithm(classifier,df_scaled,y,train_ind[i],test_ind[i],df,fold=parameters['fold'],nomi=nomi,filename=str(i),family=parameters['family'],use_scaler=parameters.get('feature_scaling',True))
                        gam=None
                    df.loc[test_ind[i],'SI']=prob[i]
            elif parameters['testN']==1:
                train=np.arange(len(y))
                test=np.arange(len(y))
                if algorithm==Algorithms.alg_GAMrun:
                    prob[0],CI[0],predictors_weights=algorithm(classifier,df_scaled,y,train,test,df,splines=parameters['splines'],dtypes=parameters['dtypes'],nomi=nomi,fold=parameters['fold'],family=parameters['family'],scale_columns=parameters['linear']+parameters['continuous'],use_scaler=parameters.get('feature_scaling',True))
                elif algorithm==Algorithms.alg_NNrun:
                    prob[0],predictors_weights=algorithm(classifier,df_scaled,y,train,test,df,fold=parameters['fold'],nomi=nomi,family=parameters['family'],use_scaler=parameters.get('feature_scaling',True))
                elif algorithm==Algorithms.alg_MLrun:
                    prob[0],predictors_weights=algorithm(classifier,df_scaled,y,train,test,df,fold=parameters['fold'],nomi=nomi,family=parameters['family'],use_scaler=parameters.get('feature_scaling',True))
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
            regression_families = {
                'gaussian',
                'SVM_regressor',
                'RF_regressor',
                'DT_regressor',
                'MLP_regressor',
            }
            if parameters['family'] in regression_families:
                method=KFold(
                    n_splits=parameters['testN'],
                    shuffle=True,
                    random_state=10,
                )
                for i, (train, test) in enumerate(method.split(df_scaled, y)):
                    X_train[i]=train
                    X_test[i]=test
            else:
                method=StratifiedKFold(
                    n_splits=parameters['testN'],
                    shuffle=True,
                    random_state=10,
                )
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
    
    def fit_custom_scaler(df,columns):
        scaler={}
        for column in dict.fromkeys(columns):
            mean=float(df[column].mean())
            std=float(df[column].std())
            if not np.isfinite(std) or std == 0:
                raise ValueError(f"Cannot scale constant or invalid GAM variable: {column}")
            scaler[column]={'mean':mean,'std':std}
        return scaler

    def apply_custom_scaler(df,scaler):
        df_scaled=df.copy()
        for column,statistics in scaler.items():
            df_scaled[column]=(df[column]-statistics['mean'])/statistics['std']
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
    
    def GAM_plot(gam,df,nomi,fold,filename,scaled_df=None):
        print('plotting covariates.....')
        covariate_terms=[
            (index,term)
            for index,term in enumerate(gam.terms)
            if not term.isintercept
        ]
        interaction_terms=[
            (index,term)
            for index,term in enumerate(gam.terms)
            if isinstance(term,terms.TensorTerm)
        ]

        limits=[]
        for term_index,term in covariate_terms:
            if isinstance(term,terms.TensorTerm):
                continue
            grid=gam.generate_X_grid(term=term_index)
            _,confidence=gam.partial_dependence(term=term_index,X=grid,width=0.95)
            limits.extend([np.min(confidence[:,0]),np.max(confidence[:,1])])
        y_limits=(min(limits)-0.2,max(limits)+0.2) if limits else None

        GAM_utils.GAM_plot_pages(
            gam,df,nomi,covariate_terms,y_limits,fold,filename,scaled=False
        )
        GAM_utils.GAM_plot_interactions(
            gam,df,nomi,interaction_terms,fold,filename,scaled=False
        )

        if scaled_df is not None:
            GAM_utils.GAM_plot_pages(
                gam,scaled_df,nomi,covariate_terms,y_limits,fold,filename,scaled=True
            )
            GAM_utils.GAM_plot_interactions(
                gam,scaled_df,nomi,interaction_terms,fold,filename,scaled=True
            )

    def GAM_plot_pages(gam,df,nomi,covariate_terms,y_limits,fold,filename,scaled):
        prefix='Model_covariates_scaled' if scaled else 'Model_covariates'
        for page,start in enumerate(range(0,len(covariate_terms),12)):
            page_terms=covariate_terms[start:start+12]
            columns=min(3,len(page_terms))
            rows=int(np.ceil(len(page_terms)/columns))
            fig,axes=plt.subplots(rows,columns,figsize=(5*columns,4*rows),squeeze=False)
            axes=axes.ravel()

            for axis,(term_index,term) in zip(axes,page_terms):
                GAM_utils.GAM_plot_term(
                    gam,df,nomi,term_index,term,axis,y_limits,scaled
                )
            for axis in axes[len(page_terms):]:
                fig.delaxes(axis)

            fig.tight_layout()
            fig.savefig(
                os.path.join(fold,prefix+filename+'page'+str(page)+'.pdf'),
                bbox_inches='tight',
            )
            plt.close(fig)

    def GAM_plot_term(gam,df,nomi,term_index,term,axis,y_limits,scaled):
        if isinstance(term,terms.TensorTerm):
            feature_indices=[int(feature) for feature in term.feature]
            first_name=nomi[feature_indices[0]]
            second_name=nomi[feature_indices[1]]
            grid=gam.generate_X_grid(term=term_index,meshgrid=True)
            partial=gam.partial_dependence(
                term=term_index,X=grid,meshgrid=True
            )
            first_grid,second_grid,plot_values=GAM_utils.GAM_tensor_plot_data(
                df,first_name,second_name,grid,partial,scaled
            )
            mesh=axis.pcolormesh(
                first_grid,second_grid,plot_values,cmap='viridis',shading='auto'
            )
            axis.set_aspect('equal',adjustable='datalim')
            colorbar=axis.figure.colorbar(mesh,ax=axis)
            colorbar.set_label('Partial Effect',fontsize=16)
            colorbar.ax.tick_params(labelsize=16)
            axis.set_xlabel(first_name,fontsize=16)
            axis.set_ylabel(second_name,fontsize=16)
            axis.tick_params(labelsize=14)
            return

        feature_index=int(term.feature)
        variable=nomi[feature_index]
        sample_count=(len(df[variable].unique()) if isinstance(term,terms.FactorTerm)
                      else len(df[variable]))
        grid=gam.generate_X_grid(term=term_index,n=sample_count)
        partial,confidence=gam.partial_dependence(
            term=term_index,X=grid,width=0.95
        )

        if isinstance(term,terms.FactorTerm):
            x_values=grid[:,feature_index]
            axis.plot(x_values,confidence[:,0],'o',color='gray')
            axis.plot(x_values,confidence[:,1],'o',color='gray')
            axis.plot(x_values,partial,'o',color='blue')
            axis.set_xticks(np.sort(df[variable].unique()))
        else:
            x_values=(
                grid[:,feature_index]
                if scaled else np.linspace(df[variable].min(),df[variable].max(),len(partial))
            )
            axis.plot(x_values,partial,color='blue')
            axis.fill_between(
                x_values,confidence[:,0],confidence[:,1],color='gray',alpha=0.2
            )

        axis.set_xlabel(variable)
        axis.set_ylabel('Partial Effect')
        if y_limits is not None:
            axis.set_ylim(*y_limits)

    def GAM_plot_interactions(gam,df,nomi,interaction_terms,fold,filename,scaled):
        prefix=(
            'Model_covariates_interaction_scaled'
            if scaled else 'Model_covariates_interaction'
        )
        multiple_interactions=len(interaction_terms)>1
        for term_index,term in interaction_terms:
            feature_indices=[int(feature) for feature in term.feature]
            first_name=nomi[feature_indices[0]]
            second_name=nomi[feature_indices[1]]
            grid=gam.generate_X_grid(term=term_index,meshgrid=True)
            partial=gam.partial_dependence(
                term=term_index,X=grid,meshgrid=True
            )
            first_grid,second_grid,plot_values=GAM_utils.GAM_tensor_plot_data(
                df,first_name,second_name,grid,partial,scaled
            )

            fig=plt.figure(figsize=(8,8))
            axis=fig.subplots()
            mesh=axis.pcolormesh(
                first_grid,second_grid,plot_values,cmap='viridis',shading='auto'
            )
            axis.set_aspect('equal',adjustable='datalim')
            bbox=axis.get_position()
            colorbar_axis=fig.add_axes([bbox.x1+0.03,bbox.y0,0.03,bbox.height])
            colorbar=fig.colorbar(mesh,cax=colorbar_axis)
            colorbar.set_label('Partial Effect',fontsize=16)
            colorbar.ax.tick_params(labelsize=16)
            axis.set_xlabel(first_name,fontsize=16)
            axis.set_ylabel(second_name,fontsize=16)
            axis.tick_params(labelsize=14)
            suffix='_'+str(term_index) if multiple_interactions else ''
            fig.savefig(
                os.path.join(fold,prefix+filename+suffix+'.pdf'),bbox_inches='tight'
            )
            plt.close(fig)

    def GAM_tensor_plot_data(df,first_name,second_name,grid,partial,scaled):
        if scaled:
            return grid[0],grid[1],partial
        first_values=np.linspace(
            df[first_name].min(),df[first_name].max(),partial.shape[0]
        )
        second_values=np.linspace(
            df[second_name].min(),df[second_name].max(),partial.shape[1]
        )
        return first_values,second_values,np.transpose(partial)

    def GAM_save(model_artifact,fold,filename=''):
        print('saving GAM model and coefficients.....')
        filename_pkl = os.path.join(fold,'gam_coeff'+filename+'.pkl')
        with open(filename_pkl, 'wb') as filez:
            pickle.dump(model_artifact, filez)
        GAM_utils.GAM_save_coefficients(model_artifact,fold,filename)

    def GAM_save_coefficients(model_artifact,fold,filename=''):
        gam=model_artifact['estimator']
        feature_names=model_artifact.get('feature_names',[])
        scaler=model_artifact.get('scaler',{})
        p_values=gam.statistics_.get('p_values',[])
        edof=gam.statistics_.get('edof_per_coef',[])
        rows=[]

        for term_index,term in enumerate(gam.terms):
            coefficient_indices=gam.terms.get_coef_indices(term_index)
            features=GAM_utils.GAM_term_features(term,feature_names)
            variable=' + '.join(features) if features else 'intercept'
            term_p_value=p_values[term_index] if term_index < len(p_values) else np.nan

            for basis_index,coefficient_index in enumerate(coefficient_indices):
                scale_statistics=scaler.get(features[0],{}) if len(features)==1 else {}
                rows.append({
                    'coefficient_index':int(coefficient_index),
                    'term_index':term_index,
                    'term_type':type(term).__name__,
                    'variable':variable,
                    'basis_index':basis_index,
                    'coefficient':float(gam.coef_[coefficient_index]),
                    'effective_degrees_of_freedom':(
                        float(edof[coefficient_index])
                        if coefficient_index < len(edof) else np.nan
                    ),
                    'term_p_value':float(term_p_value),
                    'input_mean':scale_statistics.get('mean',np.nan),
                    'input_std':scale_statistics.get('std',np.nan),
                })

        filename_csv=os.path.join(fold,'gam_coefficients'+filename+'.csv')
        pd.DataFrame(rows).to_csv(filename_csv,index=False)

    def GAM_term_features(term,feature_names):
        if term.isintercept:
            return []
        features=term.feature
        if np.isscalar(features):
            features=[features]
        return [
            feature_names[int(feature)]
            for feature in features
            if int(feature) < len(feature_names)
        ]

    def GAM_load(filename):
        try:
            with open(filename,'rb') as model_file:
                model_artifact=pickle.load(model_file)
        except (OSError,pickle.UnpicklingError,EOFError) as error:
            raise ValueError(f"Unable to load GAM model '{filename}': {error}") from error

        required_keys={'estimator','scaler','feature_names','family'}
        if not isinstance(model_artifact,dict) or not required_keys.issubset(model_artifact):
            raise ValueError(
                'The selected file is not a compatible GAM model. '
                'Create it with the updated sz_train_GAM tool.'
            )
        if not hasattr(model_artifact['estimator'],'coef_'):
            raise ValueError('The selected GAM model has not been fitted.')
        if not model_artifact['feature_names']:
            raise ValueError('The selected GAM model does not contain predictor names.')
        if model_artifact['family'] not in {'binomial','gaussian'}:
            raise ValueError('The selected GAM model contains an unsupported family.')
        if not set(model_artifact['scaler']).issubset(model_artifact['feature_names']):
            raise ValueError('The selected GAM model contains invalid scaler metadata.')
        return model_artifact

class ML_utils():
    def ML_save(classifier,fold,nomi, filename):
        if hasattr(classifier, 'feature_importances_'):#RF,DT
            regression_coeff=classifier.feature_importances_
            coeff=regression_coeff
            try:
                tree_rules = export_text(classifier, feature_names=nomi)
                tree_rules_list = tree_rules.split('\n')
                rules_df = pd.DataFrame({'Tree Rules': tree_rules_list})
                rules_df.to_csv(os.path.join(fold,'decision_tree_rules'+filename+'.csv'), index=False)
            except (AttributeError, ValueError):
                print('no tree')
            feature_importance_df = pd.DataFrame({
                'Feature': nomi,
                'Importance': coeff
            })
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            feature_importance_df.to_csv(os.path.join(fold,'feature_importances'+filename+'.csv'), index=False)
        else:#SVM
            regression_coeff=classifier.coef_
            regression_intercept=classifier.intercept_
            coeff=np.hstack((regression_intercept,regression_coeff[0]))
            coeff_df = pd.DataFrame({
                'Feature': ['intercept'] + nomi,
                'Coefficient': coeff
            })
            coeff_df.to_csv(os.path.join(fold,'coefficients'+filename+'.csv'), index=False)
class NN_utils():
    def NN_plot(NNclassifier,fold,filename):
        plt.figure(figsize=(10, 6))
        plt.plot(NNclassifier.loss_curve_)
        plt.plot(NNclassifier.validation_scores_)
        plt.xlabel('Iterations',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.grid()
        plt.legend(['Train','Test'],prop={'size': 16})
        plt.savefig(os.path.join(fold,'loss_curve'+filename+'.pdf'), bbox_inches='tight')
