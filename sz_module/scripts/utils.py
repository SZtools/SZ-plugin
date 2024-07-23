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

#from pygam import LogisticGAM, s, f, terms

from qgis.core import (QgsVectorLayer,
                       QgsFields,
                       QgsField,
                       QgsProject,
                       QgsVectorFileWriter,
                       QgsWkbTypes,
                       QgsFeature,
                       QgsGeometry,
                       QgsProcessingContext
)
import numpy as np
import pandas as pd
from qgis.PyQt.QtCore import QVariant
import os
from collections import OrderedDict


class SZ_utils():

    # def load_simple(directory,parameters):
    #     layer = QgsVectorLayer(parameters['INPUT_VECTOR_LAYER'], '', 'ogr')
    #     crs=layer.crs()
    #     campi=[]
    #     for field in layer.fields():
    #         campi.append(field.name())
    #     campi.append('geom')
    #     gdp=pd.DataFrame(columns=campi,dtype=float)
    #     features = layer.getFeatures()
    #     count=0
    #     feat=[]
    #     for feature in features:
    #         attr=feature.attributes()
    #         geom = feature.geometry()
    #         feat=attr+[geom.asWkt()]
    #         gdp.loc[len(gdp)] = feat
    #         count=+ 1
    #     gdp.to_csv(directory+'/file.csv')
    #     del gdp
    #     gdp=pd.read_csv(directory+'/file.csv')
    #     gdp['ID']=np.arange(1,len(gdp.iloc[:,0])+1)
    #     df=gdp[parameters['field1']]
    #     nomi=list(df.head())
    #     lsd=gdp[parameters['lsd']]
    #     print(parameters,'printalo')
    #     if parameters['family']=='binomial':
    #         lsd[lsd>0]=1
    #     else:
    #         lsd[lsd>0]=np.log(lsd[lsd>0])
    #         print('lsd',lsd,'lsd')
    #     df['y']=lsd#.astype(int)
    #     df['ID']=gdp['ID']
    #     df['geom']=gdp['geom']
    #     df=df.dropna(how='any',axis=0)
    #     X=[parameters['field1']]
    #     if parameters['testN']==0:
    #         train=df
    #         test=pd.DataFrame(columns=nomi,dtype=float)
    #     else:
    #         # split the data into train and test set
    #         per=int(np.ceil(df.shape[0]*parameters['testN']/100))
    #         train, test = train_test_split(df, test_size=per, random_state=42, shuffle=True)
    #     return train, test, nomi,crs,df
    
    def load_cv(directory,parameters):
        layer = QgsVectorLayer(parameters['INPUT_VECTOR_LAYER'], '', 'ogr')
        crs=layer.crs()
        campi=[]
        for field in layer.fields():
            campi.append(field.name())
        campi.append('geom')
        gdp=pd.DataFrame(columns=campi,dtype=float)
        features = layer.getFeatures()
        count=0
        feat=[]
        for feature in features:
            attr=feature.attributes()
            geom = feature.geometry()
            feat=attr+[geom.asWkt()]
            gdp.loc[len(gdp)] = feat
            count=+ 1
        gdp.to_csv(directory+'/file.csv')
        del gdp
        gdp=pd.read_csv(directory+'/file.csv')
        gdp['ID']=np.arange(1,len(gdp.iloc[:,0])+1)
        if 'time' in parameters:
            if parameters['time']==None:
                #df=gdp[parameters['field1']]
                df=pd.DataFrame(gdp[parameters['field1']].copy())
            else:
                df=pd.DataFrame(gdp[parameters['field1']+[parameters['time']]].copy())
                #df=gdp[parameters['field1']+[parameters['time']]]
        else:
            df=pd.DataFrame(gdp[parameters['field1']].copy())
        #df = df.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
        lsd=gdp[parameters['lsd']]
        lsd[lsd>0]=1
        if parameters['family']=='binomial':
            lsd[lsd>0]=1
        elif parameters['family']=='binomial' and parameters['gauss_scale']=='log scale':
            lsd[lsd>0]=np.log(lsd[lsd>0])
        df['y']=lsd#.astype(int)
        df['ID']=gdp['ID']
        df['geom']=gdp['geom']
        df=df.dropna(how='any',axis=0)
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
        plt.plot(fpr1, tpr1, color='green',lw=lw, label= 'Complete dataset (AUC = %0.2f, f1 = %0.2f, ckappa = %0.2f)' %(r, f1_tot,ck_tot))
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
        df=parameters['df']
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

            plt.plot(fprv, tprv,lw=lw, alpha=0.5, label='ROC fold '+str(i+1)+' AUC = %0.2f, f1 = %0.2f, ckappa = %0.2f' %(aucv, f1_tot,ck_tot))
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

    
    def stamp_simple(parameters):
        train=parameters['train']
        y_t=train['y']
        scores_t=train['SI']

        test=parameters['test']
        y_v=test['y']
        scores_v=test['SI']
        lw = 2
        
        fprv, tprv, treshv = roc_curve(y_v,scores_v)
        fprt, tprt, tresht = roc_curve(y_t,scores_t)

        aucv=roc_auc_score(y_v, scores_v)
        auct=roc_auc_score(y_t, scores_t)
        normt=(scores_t-scores_t.min())/(scores_t.max()-scores_t.min())
        normv=(scores_v-scores_v.min())/(scores_v.max()-scores_v.min())

        fig=plt.figure()
        plt.plot(fprv, tprv, color='green',lw=lw, label= 'Prediction performance (AUC = %0.2f)' %aucv)
        plt.plot(fprt, tprt, color='red',lw=lw, label= 'Success performance (AUC = %0.2f)' %auct)
        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        #plt.show()
        try:
            fig.savefig(parameters['OUT']+'/fig.pdf')
        except:
            os.mkdir(parameters['OUT'])
            fig.savefig(parameters['OUT']+'/fig.pdf')

    def save(parameters):

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
    
