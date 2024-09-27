import matplotlib.pyplot as plt
from processing.algs.gdal.GdalUtils import GdalUtils
import plotly.graph_objs as go
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, cohen_kappa_score
from scipy.stats import pearsonr
import csv
from copy import copy


from shapely import wkb
from shapely.geometry import shape

from matplotlib.ticker import ScalarFormatter

#from pygam import LogisticGAM, s, f, terms

from qgis.core import (QgsVectorLayer,
                       QgsFields,
                       QgsField,
                       QgsProject,
                       QgsVectorFileWriter,
                       QgsWkbTypes,
                       QgsFeature,
                       QgsGeometry,
                       QgsProcessingContext,
                       QgsVectorLayerExporter
)
import numpy as np
import pandas as pd
from qgis.PyQt.QtCore import QVariant
import os
from collections import OrderedDict
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
          #fields,
          #layer.fields(),
          #layer.wkbType(),
          #layer.crs(),
          #transform_context,
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
        #primary_key=SZ_utils.get_id_column(directory+'/file.gpkg')
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
                elif parameters['family']=='gaussian' and parameters['gauss_scale']=='log scale':
                    lsd[lsd>0]=np.log(lsd[lsd>0])
            except:
                lsd[lsd>0]=1
            df['y']=lsd#.astype(int)
        except:
            print('no lsd required')
        df['ID']=gdp.index
        df['geom']=gdp['geom']
        #df=df.dropna(how='any',axis=0)
        print('input layer loaded')
        del gdp
        del lsd
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
        #plt.show()
        print('ROC curve figure = ',parameters['OUT']+'/fig_cv.pdf')
        try:
            fig.savefig(parameters['OUT']+'/fig_cv.pdf')
        except:
            os.mkdir(parameters['OUT'])
            fig.savefig(parameters['OUT']+'/fig_cv.pdf')

    
    # def stamp_simple(parameters):
    #     train=parameters['train']
    #     y_t=train['y']
    #     scores_t=train['SI']

    #     test=parameters['test']
    #     y_v=test['y']
    #     scores_v=test['SI']
    #     lw = 2
        
    #     fprv, tprv, treshv = roc_curve(y_v,scores_v)
    #     fprt, tprt, tresht = roc_curve(y_t,scores_t)

    #     aucv=roc_auc_score(y_v, scores_v)
    #     auct=roc_auc_score(y_t, scores_t)
    #     normt=(scores_t-scores_t.min())/(scores_t.max()-scores_t.min())
    #     normv=(scores_v-scores_v.min())/(scores_v.max()-scores_v.min())

    #     fig=plt.figure()
    #     plt.plot(fprv, tprv, color='green',lw=lw, label= 'Prediction performance (AUC = %0.2f)' %aucv)
    #     plt.plot(fprt, tprt, color='red',lw=lw, label= 'Success performance (AUC = %0.2f)' %auct)
    #     plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('ROC')
    #     plt.legend(loc="lower right")
    #     #plt.show()
    #     try:
    #         fig.savefig(parameters['OUT']+'/fig.pdf')
    #     except:
    #         os.mkdir(parameters['OUT'])
    #         fig.savefig(parameters['OUT']+'/fig.pdf')

    def stamp_qq(parameters):
        print('plotting....')
        df=parameters['df_train']
        df=df.dropna(subset=['SI'])
        df_train=df['SI']

        df=parameters['df_trans']
        df=df.dropna(subset=['SI'])
        df_trans=df['SI']

        # Compute percentiles
        percentiles_train = np.percentile(df_train, np.arange(0, 101, 1))
        percentiles_trans = np.percentile(df_trans, np.arange(0, 101, 1))

        #print(max(np.concatenate((percentiles_train, percentiles_trans))))

        #ercentiles_train_norm = (percentiles_train - percentiles_train.min()) / (percentiles_train.max() - percentiles_train.min())
        #percentiles_trans_norm = (percentiles_trans - percentiles_trans.min()) / (percentiles_trans.max() - percentiles_trans.min())
        
        max_val = max(np.max(percentiles_train), np.max(percentiles_trans))
        min_val = min(np.min(percentiles_train), np.min(percentiles_trans))
        x_buffer = (max_val - min_val) / 10


        # Plot percentiles
        fig=plt.figure(figsize=(8, 6))
        plt.plot(percentiles_train,percentiles_trans, 'bo')
        plt.plot([min_val-x_buffer, max_val+x_buffer], [min_val-x_buffer, max_val+x_buffer], 'k--')

        # Labels and title

        plt.xlim(min_val-x_buffer, max_val+x_buffer)
        plt.ylim(min_val-x_buffer, max_val+x_buffer)
        plt.xscale('log')
        plt.yscale('log')

        ax = plt.gca()
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        plt.xlabel('Observed (quantiles)')
        plt.ylabel('Predicted (quantiles)')
        #plt.grid(True)
        #plt.legend()
        
        print('QQ figure = ',parameters['OUT']+'/fig_qq.pdf')
        try:
            fig.savefig(parameters['OUT']+'/fig_qq.pdf')
        except:
            os.mkdir(parameters['OUT'])
            fig.savefig(parameters['OUT']+'/fig_qq.pdf')

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
                fields.append(QgsField(field, QVariant.Int))
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


    def errors(parameters):
        df=parameters['df']
        nomi=list(df.head())
        y=df['y']
        predic=df['SI']
        min_absolute_error = np.min(np.abs(y - predic))
        rmse = np.sqrt(mean_squared_error(y, predic))
        r_squared = r2_score(y, predic)
        pearson_coefficient, _ = pearsonr(y, predic)
        errors=[min_absolute_error,rmse,r_squared,pearson_coefficient]

        output_file = parameters['file']

        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(["Metric", "Value"])
            # Write data
            writer.writerow(["Minimum Absolute Error", min_absolute_error])
            writer.writerow(["RMSE", rmse])
            writer.writerow(["R-squared", r_squared])
            writer.writerow(["Pearson Coefficient", pearson_coefficient])
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
            
    
