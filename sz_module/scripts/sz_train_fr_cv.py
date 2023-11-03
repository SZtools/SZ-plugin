#!/usr/bin/python
#coding=utf-8
"""
/***************************************************************************
    02 FR Fitting/CrossValid
        begin                : 2021-11
        copyright            : (C) 2021 by Giacomo Titti,
                               Padova, November 2021
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
    02 FR Fitting/CrossValid
    Copyright (C) 2021 by Giacomo Titti, Padova, November 2021

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
import sys
sys.setrecursionlimit(10000)
from qgis.PyQt.QtCore import QCoreApplication,QVariant
from qgis.core import (QgsProcessing,
                       QgsFeatureSink,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterRasterLayer,
                       QgsMessageLog,
                       Qgis,
                       QgsProcessingMultiStepFeedback,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFileDestination,
                       QgsProcessingParameterVectorLayer,
                       QgsVectorLayer,
                       QgsRasterLayer,
                       QgsProject,
                       QgsField,
                       QgsFields,
                       QgsVectorFileWriter,
                       QgsWkbTypes,
                       QgsFeature,
                       QgsGeometry,
                       QgsPointXY,
                       QgsProcessingParameterField,
                       QgsProcessingParameterString,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterField,
                       QgsProcessingParameterVectorDestination,
                       QgsProcessingContext
                       )
from qgis.core import *
from qgis.utils import iface
from qgis import processing
from osgeo import gdal,ogr,osr
import numpy as np
import math
import operator
import random
from qgis import *
# ##############################
import matplotlib.pyplot as plt
import csv
from processing.algs.gdal.GdalUtils import GdalUtils
#import plotly.express as px
#import chart_studio
import plotly.offline
import plotly.graph_objs as go
#import geopandas as gd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from scipy import interpolate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

import tempfile
from sz_module.utils import SZ_utils


class FRcvAlgorithm():
   
    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Input layer'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))

        self.addParameter(QgsProcessingParameterField(self.STRING, 'Independent variables', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=True,type=QgsProcessingParameterField.Any))
        self.addParameter(QgsProcessingParameterField(self.STRING2, 'Field of dependent variable (0 for absence, > 0 for presence)', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER, self.tr('K-fold CV (1 to fit or > 1 to cross-validate)'), minValue=1,type=QgsProcessingParameterNumber.Integer,defaultValue=2))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, 'Output test/fit',fileFilter='GeoPackage (*.gpkg *.GPKG)', defaultValue=None))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT2, 'Calculated weights','*.txt', defaultValue=None))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT3, 'Outputs folder destination', defaultValue=None, createByDefault = True))

    def process(self, parameters, context, feedback):
        self.f=tempfile.gettempdir()
        feedback = QgsProcessingMultiStepFeedback(1, feedback)
        results = {}
        outputs = {}

        source = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        parameters['covariates']=source.source()
        if parameters['covariates'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        parameters['field1'] = self.parameterAsFields(parameters, self.STRING, context)
        if parameters['field1'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING))

        parameters['fieldlsd'] = self.parameterAsString(parameters, self.STRING2, context)
        if parameters['fieldlsd'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING2))

        parameters['testN'] = self.parameterAsInt(parameters, self.NUMBER, context)
        if parameters['testN'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER))

        parameters['out'] = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        if parameters['out'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))

        parameters['out2'] = self.parameterAsFileOutput(parameters, self.OUTPUT2, context)
        if parameters['out2'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT2))

        parameters['folder'] = self.parameterAsString(parameters, self.OUTPUT3, context)
        if parameters['folder'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT3))

        alg_params = {
            'INPUT_VECTOR_LAYER': parameters['covariates'],
            'field1': parameters['field1'],
            'txt':parameters['out2'],
            'lsd' : parameters['fieldlsd'],
            'testN':parameters['testN']
        }

        outputs['prob'],outputs['test_ind'],outputs['df'],outputs['nomi'],outputs['crs']=self.load(alg_params)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        if parameters['testN']>0:
            alg_params = {
                'df': outputs['df'],
                'crs': outputs['crs'],
                'OUT': parameters['out']
            }
            self.save(alg_params)

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        alg_params = {
            'test_ind': outputs['test_ind'],
            'df': outputs['df'],
            'OUT':parameters['folder']

        }
        SZ_utils.stamp_cv(alg_params)

        feedback.setCurrentStep(3)
        if feedback.isCanceled():
            return {}
        
        results['out'] = parameters['out']
        
        fileName = parameters['out']
        layer1 = QgsVectorLayer(fileName,"test","ogr")
        subLayers =layer1.dataProvider().subLayers()

        for subLayer in subLayers:
            name = subLayer.split('!!::!!')[1]
            print(name,'name')
            uri = "%s|layername=%s" % (fileName, name,)
            print(uri,'uri')
            # Create layer
            sub_vlayer = QgsVectorLayer(uri, name, 'ogr')
            if not sub_vlayer.isValid():
                print('layer failed to load')
            # Add layer to map
            context.temporaryLayerStore().addMapLayer(sub_vlayer)
            context.addLayerToLoadOnCompletion(sub_vlayer.id(), QgsProcessingContext.LayerDetails('test', context.project(),'LAYER1'))

        feedback.setCurrentStep(4)
        if feedback.isCanceled():
            return {}

        return results

    def load(self,parameters):
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
        gdp.to_csv(self.f+'/file.csv')
        del gdp
        gdp=pd.read_csv(self.f+'/file.csv')
        gdp['ID']=np.arange(1,len(gdp.iloc[:,0])+1)
        df=gdp[parameters['field1']]
        nomi=list(df.head())
        lsd=gdp[parameters['lsd']]
        lsd[lsd>0]=1
        df['y']=lsd#.astype(int)
        df['ID']=gdp['ID']
        df['geom']=gdp['geom']
        df=df.dropna(how='any',axis=0)
        x=df[parameters['field1']]
        y=df['y']
        X = x
        train_ind={}
        test_ind={}
        prob={}
        df["SI"] = np.nan
        if parameters['testN']>1:
            cv = StratifiedKFold(n_splits=parameters['testN'])
            for i, (train, test) in enumerate(cv.split(X, y)):
                train_ind[i]=train
                test_ind[i]=test
                prob[i]=self.fr(train,test,df,nomi,parameters['txt'])
                df.loc[test,'SI']=prob[i]
        elif parameters['testN']==1:
            train=np.arange(len(y))
            test=np.arange(len(y))
            prob[0]=self.fr(train,test,df,nomi,parameters['txt'])
            df.loc[test,'SI']=prob[0]
            test_ind[0]=test
        return prob,test_ind,df,nomi,crs


    def fr(self,train,test,frame,nomes,txt):
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
        return(test['SI'])


       

    # def stampfit(self,parameters):
    #     df=parameters['df']
    #     y_true=df['y']
    #     scores=df['SI']
    #     #W=df['w']
    #     ################################figure
    #     #fpr1, tpr1, tresh1 = roc_curve(y_true,scores,sample_weight=W)
    #     fpr1, tpr1, tresh1 = roc_curve(y_true,scores)
    #     norm=(scores-scores.min())/(scores.max()-scores.min())
    #
    #     #fpr2, tpr2, tresh2 = roc_curve(self.y_true,self.norm)
    #     #print(tresh1)
    #
    #     #fprv, tprv, treshv = roc_curve(self.y_v,self.scores_v)
    #     #fprt, tprt, tresht = roc_curve(self.y_t,self.scores_t)
    #
    #     #print self.fpr
    #     #print self.tpr
    #     #print self.classes
    #     #aucv=roc_auc_score(self.y_v, self.scores_v, None)
    #     #auct=roc_auc_score(self.y_t, self.scores_t, None)
    #     r=roc_auc_score(y_true, scores, None)
    #
    #     fig=plt.figure()
    #     lw = 2
    #     plt.plot(fpr1, tpr1, color='green',lw=lw, label= 'Complete dataset (AUC = %0.2f)' %r)
    #     #plt.plot(self.fpr, self.tpr, 'ro')
    #     #plt.plot(self.fpr, self.tpr, color='darkorange',lw=lw, label='Classified dataset (AUC = %0.2f)' % self.fitness)
    #     plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('ROC')
    #     plt.legend(loc="lower right")
    #     #plt.show()
    #     try:
    #         fig.savefig(parameters['OUT']+'/fig01.png')
    #     except:
    #         os.mkdir(parameters['OUT'])
    #         fig.savefig(parameters['OUT']+'/fig01.png')
    #     # fig=plt.figure()
    #     # frequency, bins = np.histogram(norm)
    #     # frequency=(frequency/len(norm))*100
    #     # bincenters = 0.5*(bins[1:]+bins[:-1])
    #     # plt.hist(bins[:-1], bins, weights=frequency,color='blue',alpha = 0.8)
    #     # #plt.plot(bincenters,frequency,'-')#segmented curve
    #     # #print(bincenters,frequency)
    #     #
    #     # #xnew = interpolate.splrep(bincenters, frequency, s=0)
    #     # xnew = np.linspace(bincenters.min(),bincenters.max())
    #     # #print(bincenters, xnew)
    #     # power_smooth=interpolate.splev(bincenters, xnew, der=0)
    #     # #power_smooth = spline(bincenters,frequency,xnew)
    #     # plt.plot(xnew,power_smooth,color='black',lw=lw, label= 'LSI')
    #     # plt.xlabel('Standardized Susceptibility Index')
    #     # plt.ylabel('Area %')
    #     # plt.title('')
    #     # plt.legend(loc="upper right")
    #     # #plt.show()
    #     # fig.savefig(parameters['OUT']+'/fig02.png')

    # def stampcv(self,parameters):
    #     df=parameters['df']
    #     test_ind=parameters['test_ind']

    #     #test=parameters['test']
    #     y_v=df['y']
    #     scores_v=df['SI']
    #     lw = 2
    #     #W=df['w']
    #     ################################figure
    #     #fpr1, tpr1, tresh1 = roc_curve(y_true,scores,sample_weight=W)
    #     #fpr1, tpr1, tresh1 = roc_curve(y_true,scores)
    #     #fpr2, tpr2, tresh2 = roc_curve(self.y_true,self.norm)
    #     #print(tresh1)
    #     fig=plt.figure()
    #     #print(len(test_ind))
    #     #print(test_ind)
    #     plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    #     for i in range(len(test_ind)):

    #         fprv, tprv, treshv = roc_curve(y_v[test_ind[i]],scores_v[test_ind[i]])
    #         #fprt, tprt, tresht = roc_curve(y_t,scores_t)

    #         #print self.fpr
    #         #print self.tpr
    #         #print self.classes
    #         aucv=roc_auc_score(y_v[test_ind[i]],scores_v[test_ind[i]])
    #         print('ROC '+ str(i) +' AUC=',aucv)
    #         #auct=roc_auc_score(y_t, scores_t, None)
    #         #r=roc_auc_score(y_true, scores, None)
    #         #normt=(scores_t-scores_t.min())/(scores_t.max()-scores_t.min())
    #         #normv=(scores_v-scores_v.min())/(scores_v.max()-scores_v.min())



    #         plt.plot(fprv, tprv,lw=lw, alpha=0.5, label='ROC fold '+str(i+1)+' (AUC = %0.2f)' %aucv)
    #         #plt.plot(fprt, tprt, color='red',lw=lw, label= 'Success performance (AUC = %0.2f)' %auct)
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     #plt.title('ROC')
    #     plt.legend(loc="lower right")
    #     #plt.show()
    #     print('ROC curve figure = ',parameters['OUT']+'/fig02.pdf')
    #     try:
    #         fig.savefig(parameters['OUT']+'/fig02.pdf')

    #     except:
    #         os.mkdir(parameters['OUT'])
    #         fig.savefig(parameters['OUT']+'/fig02.pdf')

    #     # fig=plt.figure()
    #     # frequency, bins = np.histogram(normt)
    #     # frequency=(frequency/len(normt))*100
    #     # bincenters = 0.5*(bins[1:]+bins[:-1])
    #     # plt.hist(bins[:-1], bins, weights=frequency,color='blue',alpha = 0.8)
    #     # #plt.plot(bincenters,frequency,'-')#segmented curve
    #     # xnew = np.linspace(bincenters.min(),bincenters.max())
    #     # power_smooth=interpolate.splev(bincenters, xnew, der=0)
    #     # #power_smooth = spline(bincenters,frequency,xnew)
    #     # plt.plot(xnew,power_smooth,color='black',lw=lw, label= 'Train SI')
    #     # plt.xlabel('Standardized Susceptibility Index')
    #     # plt.ylabel('Area %')
    #     # plt.title('')
    #     # plt.legend(loc="upper right")
    #     # fig.savefig(parameters['OUT']+'/fig02.png') # Use fig. here
    #     # #plt.show()
    #     #
    #     # fig=plt.figure()
    #     # frequency, bins = np.histogram(normv)
    #     # frequency=(frequency/len(normv))*100
    #     # bincenters = 0.5*(bins[1:]+bins[:-1])
    #     # plt.hist(bins[:-1], bins, weights=frequency,color='blue',alpha = 0.8)
    #     # #plt.plot(bincenters,frequency,'-')#segmented curve
    #     # xnew = np.linspace(bincenters.min(),bincenters.max())
    #     # power_smooth=interpolate.splev(bincenters, xnew, der=0)
    #     # #power_smooth = spline(bincenters,frequency,xnew)
    #     # plt.plot(xnew,power_smooth,color='black',lw=lw, label= 'Test SI')
    #     # plt.xlabel('Standardized Susceptibility Index')
    #     # plt.ylabel('Area %')
    #     # plt.title('')
    #     # plt.legend(loc="upper right")
    #     # fig.savefig(parameters['OUT']+'/fig03.png') # Use fig. here

    def save(self,parameters):

        #print(parameters['nomi'])
        df=parameters['df']
        nomi=list(df.head())
        # define fields for feature attributes. A QgsFields object is needed
        fields = QgsFields()

        #fields.append(QgsField('ID', QVariant.Int))

        for field in nomi:
            if field=='ID':
                fields.append(QgsField(field, QVariant.Int))
            if field=='geom':
                continue
            if field=='y':
                fields.append(QgsField(field, QVariant.Int))
            else:
                fields.append(QgsField(field, QVariant.Double))

        #crs = QgsProject.instance().crs()
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

        # delete the writer to flush features to disk
        del writer

    def addmap(self,parameters):
        context=parameters()
        fileName = parameters['trainout']
        layer = QgsVectorLayer(fileName,"train","ogr")
        subLayers =layer.dataProvider().subLayers()

        for subLayer in subLayers:
            name = subLayer.split('!!::!!')[1]
            print(name,'name')
            uri = "%s|layername=%s" % (fileName, name,)
            print(uri,'uri')
            # Create layer
            sub_vlayer = QgsVectorLayer(uri, name, 'ogr')
            if not sub_vlayer.isValid():
                print('layer failed to load')
            # Add layer to map
            context.temporaryLayerStore().addMapLayer(sub_vlayer)
            context.addLayerToLoadOnCompletion(sub_vlayer.id(), QgsProcessingContext.LayerDetails('layer', context.project(),'LAYER'))

            #QgsProject.instance().addMapLayer(sub_vlayer)
            #iface.mapCanvas().refresh()


        # fileName = parameters['out']
        # layer = QgsVectorLayer(fileName,"test","ogr")
        # subLayers =layer.dataProvider().subLayers()
        #
        # for subLayer in subLayers:
        #     name = subLayer.split('!!::!!')[1]
        #     uri = "%s|layername=%s" % (fileName, name,)
        #     # Create layer
        #     sub_vlayer = QgsVectorLayer(uri, name, 'ogr')
        #     if not sub_vlayer.isValid():
        #         print('layer failed to load')
        #     # Add layer to map
        #     QgsProject.instance().addMapLayer(sub_vlayer)
