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

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, cohen_kappa_score
from scipy.stats import pearsonr
from copy import copy
from shapely.geometry import shape
from qgis.core import (QgsVectorLayer,
                       QgsFields,
                       QgsField,
                       QgsProject,
                       QgsVectorFileWriter,
                       QgsWkbTypes,
                       QgsFeature,
                       QgsGeometry,
                       QgsProcessingContext,
)
import numpy as np
import pandas as pd
from qgis.PyQt.QtCore import QVariant
import os
import sqlite3
from shapely.geometry import shape
from shapely.wkt import dumps
import fiona

class SZ_utils():
    def generate_ghost_input(input,output):
        input_shapefile_path = input
        layer = QgsVectorLayer(input_shapefile_path, 'Input Layer', 'ogr')
        transform_context = QgsProject.instance().transformContext()
        save_options = QgsVectorFileWriter.SaveVectorOptions()
        save_options.driverName = 'GPKG'
        save_options.fileEncoding = 'UTF-8'
        writer = QgsVectorFileWriter.writeAsVectorFormat(
          layer,  
          output,
          save_options
        )
        print(output)
        del writer
        del layer

    def load_geopackage(file_path, table_name='file'):
        print('loading dataframe')
        layer = QgsVectorLayer(file_path, 'Input Layer', 'ogr')
        crs=layer.crs()
        input_gpkg = file_path
        with fiona.open(input_gpkg) as source:
            records = []
            for feature in source:
                properties = feature['properties']
                geom_wkt = dumps(shape(feature['geometry']))
                properties['geom'] = geom_wkt  
                records.append(properties)
        df = pd.DataFrame(records)
        del layer
        del source
        del records
        return df,crs

    def get_id_column(file_path, table_name='file'):
        conn = sqlite3.connect(file_path)
        columns_info = pd.read_sql_query(f"PRAGMA table_info({table_name});", conn)
        conn.close()
        id_column = columns_info[columns_info['pk'] > 0]['name']
        if not id_column.empty:
            return id_column.values[0]
        else:
            return None

    def load_cv(directory,parameters):
        SZ_utils.generate_ghost_input(parameters['INPUT_VECTOR_LAYER'],directory+'/file.gpkg')
        gdp,crs=SZ_utils.load_geopackage(directory+'/file.gpkg')
        if 'time' in parameters:
            if parameters['time']==None:
                df=pd.DataFrame(gdp[parameters['nomi']].copy())
            else:
                df=pd.DataFrame(gdp[parameters['nomi']+[parameters['time']]].copy())
        else:
            df=pd.DataFrame(gdp[parameters['nomi']].copy())
        try:
            lsd=gdp[parameters['lsd']]
            try:
                if parameters['family']=='binomial':
                    lsd[lsd>0]=1
                elif parameters['family']=='gaussian' and parameters['scale']=='log_scale':
                    lsd[lsd>0]=np.log(lsd[lsd>0])
                elif parameters['family']=='gaussian' and parameters['scale']=='linear_scale':
                    print('do nothing')
                elif parameters['family']=='MLP_regressor' and parameters['scale']=='log_scale':
                    lsd[lsd>0]=np.log(lsd[lsd>0])
                elif parameters['family']=='MLP_regressor' and parameters['scale']=='linear_scale':
                    print('do nothing')
                else:
                    lsd[lsd>0]=1
            except:
                lsd[lsd>0]=1
            df['y']=lsd#.astype(int)
        except:
            print('no lsd required')
        df['ID']=gdp.index
        df['geom']=gdp['geom']
        print('input layer loaded')
        del gdp
        return(df,crs)

    def stampfit(parameters):
        df=parameters['df']
        y_true=df['y']
        scores=df['SI']
        ################################figure
        fpr1, tpr1, tresh1 = roc_curve(y_true,scores)
        norm=(scores-scores.min())/(scores.max()-scores.min())
        r=roc_auc_score(y_true, scores)
        idx = np.argmax(tpr1 - fpr1)  # x YOUDEN INDEX
        suscept01 = copy(scores)
        suscept01[scores > tresh1[idx]] = 1
        suscept01[scores <= tresh1[idx]] = 0
        f1_tot = f1_score(y_true, suscept01)
        ck_tot = cohen_kappa_score(y_true, suscept01)
        fig=plt.figure()
        lw = 2
        plt.plot(fpr1, tpr1, color='green',lw=lw, label= 'Complete dataset (AUC = %0.2f, F1 = %0.2f, K = %0.2f)' %(r, f1_tot,ck_tot))
        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        try:
            fig.savefig(parameters['OUT']+'/fig_fit.png')
        except:
            os.mkdir(parameters['OUT'])
            fig.savefig(parameters['OUT']+'/fig_fit.png')

    def stamp_cv(parameters):
        print('plotting....')
        df=parameters['df']
        df=df.dropna(subset=['SI'])
        test_ind=parameters['test_ind']
        y_v=df['y']
        scores_v=df['SI']
        lw = 2
        ################################figure
        fig=plt.figure()
        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        for i in range(len(test_ind)):
            fprv, tprv, treshv = roc_curve(y_v[test_ind[i]],scores_v[test_ind[i]])
            aucv=roc_auc_score(y_v[test_ind[i]],scores_v[test_ind[i]])
            print('ROC '+ str(i) +' AUC=',aucv)
            idx = np.argmax(tprv - fprv)  # x YOUDEN INDEX
            suscept01 = copy(scores_v)
            suscept01[scores_v > treshv[idx]] = 1
            suscept01[scores_v <= treshv[idx]] = 0
            f1_tot = f1_score(y_v, suscept01)
            ck_tot = cohen_kappa_score(y_v, suscept01)
            plt.plot(fprv, tprv,lw=lw, alpha=0.5, label='ROC fold '+str(i+1)+' AUC = %0.2f, F1 = %0.2f, K = %0.2f' %(aucv, f1_tot,ck_tot))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        print('ROC curve figure = ',parameters['OUT']+'/fig_cv.pdf')
        try:
            fig.savefig(parameters['OUT']+'/fig_cv.pdf')
        except:
            os.mkdir(parameters['OUT'])
            fig.savefig(parameters['OUT']+'/fig_cv.pdf')

    def stamp_qq(parameters):
        print('plotting....')
        df=parameters['df']
        df=df.dropna(subset=['SI'])
        test_ind=parameters['test_ind']
        y_v=df['y']
        scores_v=df['SI']

        fig, ax = plt.subplots(figsize=(11, 6))
        M=[]
        m=[]
        for i in range(len(test_ind)):
            df_train=y_v[test_ind[i]]
            df_trans=scores_v[test_ind[i]]
            errors=SZ_utils.errors(df_train,df_trans)
            # Compute percentiles
            percentiles_train = np.percentile(df_train, np.arange(0, 101, 1))
            percentiles_trans = np.percentile(df_trans, np.arange(0, 101, 1))
            plt.scatter(percentiles_train, percentiles_trans,marker='o',s=3, alpha=0.6, label='QQ '+str(i+1)+' MAE = %0.2f, RMSE = %0.2f, R2 = %0.2f, PCC = %0.2f' %(errors[0], errors[1],errors[2],errors[3]))
            M.append(max(max(percentiles_train),max(percentiles_trans)))
            m.append(min(min(percentiles_train),min(percentiles_trans)))
        MM=max(M)
        mm=min(m)
        plt.plot([mm, MM], [mm, MM], color='black', lw=2, linestyle='--')
        ax.set_aspect('equal','box')
        plt.xlabel('Observed')
        plt.ylabel('Predicted')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize='small')
        plt.tight_layout()
        print('QQ figure = ',parameters['OUT']+'/fig_qq.pdf')
        try:
            fig.savefig(parameters['OUT']+'/fig_qq.pdf')
        except:
            os.mkdir(parameters['OUT'])
            fig.savefig(parameters['OUT']+'/fig_qq.pdf')

    def stamp_qq_fit(parameters):
        print('plotting....')
        df=parameters['df'].dropna(how='any',axis=0)
        df_true=df['y']
        df_predict=df['SI']
        errors=SZ_utils.errors(df_true,df_predict)
        percentiles_train = np.percentile(df_true, np.arange(0, 101, 1))
        percentiles_trans = np.percentile(df_predict, np.arange(0, 101, 1))
        fig, ax = plt.subplots()
        plt.scatter(percentiles_train, percentiles_trans,marker='o',s=3, alpha=0.5, label='QQ MAE = %0.2f, RMSE = %0.2f, R2 = %0.2f, PearsCorr = %0.2f' %(errors[0], errors[1],errors[2],errors[3]))
        MM=np.max([np.max(percentiles_train),np.max(percentiles_trans)])
        mm=np.min([np.min(percentiles_train),np.min(percentiles_trans)])
        plt.plot([mm, MM], [mm, MM], color='black', lw=2, linestyle='--')
        ax.set_aspect('equal','box')
        plt.xlabel('Observed')
        plt.ylabel('Predicted')
        plt.legend(bbox_to_anchor =(0.5,-0.3), loc='lower center',fontsize='small')
        plt.tight_layout()
        print('QQ figure = ',parameters['OUT']+'/fig_qq_fit.pdf')
        try:
            fig.savefig(parameters['OUT']+'/fig_qq_fit.pdf')
        except:
            os.mkdir(parameters['OUT'])
            fig.savefig(parameters['OUT']+'/fig_qq_fit.pdf')

    def save(parameters):
        print('writing output geopackage.....')
        df=parameters['df']
        nomi=list(df.head())
        fields = QgsFields()

        for field in nomi:
            if field=='ID':
                fields.append(QgsField(field, QVariant.Int))
            if field=='geom':
                continue
            if field=='y':
                fields.append(QgsField(field, QVariant.Double))
            else:
                fields.append(QgsField(field, QVariant.Double))
        transform_context = QgsProject.instance().transformContext()
        save_options = QgsVectorFileWriter.SaveVectorOptions()
        save_options.driverName = 'GPKG'
        save_options.fileEncoding = 'UTF-8'
        writer = QgsVectorFileWriter.create(
          parameters['OUT'],
          fields,
          QgsWkbTypes.Polygon,
          parameters['crs'],
          transform_context,
          save_options
        )
        if writer.hasError() != QgsVectorFileWriter.NoError:
            print("Error when creating shapefile: ",  writer.errorMessage())
        for i, row in df.iterrows():
            fet = QgsFeature()
            fet.setGeometry(QgsGeometry.fromWkt(row['geom']))
            fet.setAttributes(list(map(float,list(df.loc[ i, df.columns != 'geom']))))
            writer.addFeature(fet)
        del writer

    def addmap(parameters):
        context=parameters()
        fileName = parameters['trainout']
        layer = QgsVectorLayer(fileName,"train","ogr")
        subLayers =layer.dataProvider().subLayers()

        for subLayer in subLayers:
            name = subLayer.split('!!::!!')[1]
            uri = "%s|layername=%s" % (fileName, name,)
            # Create layer
            sub_vlayer = QgsVectorLayer(uri, name, 'ogr')
            if not sub_vlayer.isValid():
                print('layer failed to load')
            # Add layer to map
            context.temporaryLayerStore().addMapLayer(sub_vlayer)
            context.addLayerToLoadOnCompletion(sub_vlayer.id(), QgsProcessingContext.LayerDetails('layer', context.project(),'LAYER'))

    def errors(y_true,predict):
        min_absolute_error = mean_absolute_error(y_true, predict)
        rmse = np.sqrt(mean_squared_error(y_true, predict))
        r_squared = r2_score(y_true, predict)
        pearson_coefficient, _ = pearsonr(y_true, predict)
        errors=[min_absolute_error,rmse,r_squared,pearson_coefficient]
        return(errors)
    
    def check_validity(parameters):
        for i in parameters['tensor']:
            if i not in parameters['linear']+parameters['continuous']+parameters['categorical']:
                continue
            else:
                return False
        return True
    
    def make_directory(parameters):
        if not os.path.exists(parameters['path']):
            os.mkdir(parameters['path'])