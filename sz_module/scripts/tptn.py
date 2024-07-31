#!/usr/bin/python
#coding=utf-8
"""
/***************************************************************************
    LRcvAlgorithm
        begin                : 2021-11
        copyright            : (C) 2021 by Giacomo Titti,
                               Padova, November 2021
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
    LRcvAlgorithm
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

__author__ = 'Giacomo Titti'
__date__ = '2021-11-01'
__copyright__ = '(C) 2021 by Giacomo Titti'
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

import tempfile


class FPAlgorithm(QgsProcessingAlgorithm):
    # INPUT = 'covariates'
    # STRING = 'field1'
    # #STRING1 = 'field2'
    # STRING2 = 'fieldlsd'
    # #INPUT1 = 'Slope'
    # #EXTENT = 'Extension'
    # NUMBER = 'testN'
    # #NUMBER1 = 'minSlopeAcceptable'
    # OUTPUT = 'OUTPUT'
    # #OUTPUT1 = 'OUTPUT1'
    # #OUTPUT2 = 'OUTPUT2'
    # #OUTPUT3 = 'OUTPUT3'

    # def tr(self, string):
    #     return QCoreApplication.translate('Processing', string)

    # def createInstance(self):
    #     return FPAlgorithm()

    # def name(self):
    #     return 'TpTnFpFn'

    # def displayName(self):
    #     return self.tr('03 Confusion Matrix')

    # def group(self):
    #     return self.tr('04 Classify SI')

    # def groupId(self):
    #     return '04 Classify SI'

    # def shortHelpString(self):
    #     return self.tr("This function labels each feature as True Positive (0), True Negative (1), False Positive (2), False Negative (3)")

    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Input layer'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))

        #self.addParameter( QgsProcessingParameterFeatureSource(self.INPUT,self.tr('Covariates'),[QgsProcessing.TypeVectorPolygon],defaultValue='covariatesclassed'))
        #self.addParameter(QgsProcessingParameterField(self.STRING, 'Independent variables', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=True,type=QgsProcessingParameterField.Any))
        #self.addParameter(QgsProcessingParameterField(self.STRING1, 'Last field of covariates', parentLayerParameterName=self.INPUT, defaultValue=None))
        #self.addParameter(QgsProcessingParameterField('field', 'field', type=QgsProcessingParameterField.Any, parentLayerParameterName='v', allowMultiple=True, defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING, 'Index', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING2, 'Field of dependent variable (0 for absence, > 0 for presence)', parentLayerParameterName=self.INPUT, defaultValue=None))

        self.addParameter(QgsProcessingParameterNumber(self.NUMBER, self.tr('Cutoff percentile (if empty use the YOUDEN index)'), minValue=1,type=QgsProcessingParameterNumber.Integer,optional=True))

        #self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, 'Output layer', type=QgsProcessing.TypeVectorAnyGeometry, createByDefault=True, defaultValue=None))

        #self.addParameter(QgsProcessingParameterVectorDestination(self.OUTPUT, self.tr('Output layer'), type=QgsProcessing.TypeVectorPolygon, createByDefault=True, defaultValue=None))

        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, 'Output',fileFilter='GeoPackage (*.gpkg *.GPKG)', defaultValue=None))
        #self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT1, 'Output train/fit',fileFilter='GeoPackage (*.gpkg *.GPKG)', defaultValue=None))
        #self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT2, 'Calculated weights','*.txt', defaultValue=None))
        #self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT3, 'Outputs folder destination', defaultValue=None, createByDefault = True))


    def process(self, parameters, context, feedback):
        self.f=tempfile.gettempdir()
        feedback = QgsProcessingMultiStepFeedback(1, feedback)
        results = {}
        outputs = {}

        source = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        parameters['covariates']=source.source()
        if parameters['covariates'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))



        # source = self.parameterAsVectorLayer(
        #     parameters,
        #     self.INPUT,
        #     context
        # )
        # parameters['covariates']=source.source()

        # If source was not found, throw an exception to indicate that the algorithm
        # encountered a fatal error. The exception text can be any string, but in this
        # case we use the pre-built invalidSourceError method to return a standard
        # helper text for when a source cannot be evaluated
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        parameters['field1'] = self.parameterAsString(parameters, self.STRING, context)
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

        alg_params = {
            #'INPUT_RASTER_LAYER': parameters['Slope'],
            #'INPUT_EXTENT': parameters['Extension'],
            'INPUT_VECTOR_LAYER': parameters['covariates'],
            'field1': parameters['field1'],
            #'field2': parameters['field2'],
            'lsd' : parameters['fieldlsd'],
            'testN':parameters['testN'],
            'fold':self.f
            #'INPUT_INT': parameters['BufferRadiousInPxl'],
            #'INPUT_INT_1': parameters['minSlopeAcceptable'],
        }

        outputs['df'],outputs['nomi'],outputs['crs']=Functions.load(alg_params)


        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}


        alg_params = {
            'df': outputs['df'],
            'crs': outputs['crs'],
            'OUT': parameters['out']
        }
        Functions.save(alg_params)

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        results['out'] = parameters['out']
 
        fileName = parameters['out']
        layer1 = QgsVectorLayer(fileName,"confusion_matrix","ogr")
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
            context.addLayerToLoadOnCompletion(sub_vlayer.id(), QgsProcessingContext.LayerDetails("confusion_matrix", context.project(),'LAYER1'))

        feedback.setCurrentStep(4)
        if feedback.isCanceled():
            return {}

        return results

class Functions():
    def load(parameters):
        f=parameters['fold']
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
            #print(attr)
            geom = feature.geometry()
            #print(type(geom.asWkt()))
            feat=attr+[geom.asWkt()]
            #print(feat)
            gdp.loc[len(gdp)] = feat
            #gdp = gdp.append(feat, ignore_index=True)
            count=+ 1
        gdp.to_csv(f+'/file.csv')
        del gdp
        gdp=pd.read_csv(f+'/file.csv')
        #print(feat)
        #print(gdp['S'].dtypes)
        gdp['ID']=np.arange(1,len(gdp.iloc[:,0])+1)
        #df=gdp[parameters['field1']]
        df=pd.DataFrame(data=gdp[parameters['field1']].to_numpy(), columns=[parameters['field1']])
        nomi=list(df.head())
        #print(list(df['Sf']),'1')
        lsd=gdp[parameters['lsd']]
        lsd[lsd>0]=1
        df['y']=lsd#.astype(int)
        df['ID']=gdp['ID']
        df['geom']=gdp['geom']
        df=df.dropna(how='any',axis=0)
        titles=list(df.head())
        ind=titles.index(parameters['field1'])
        dd=df.to_numpy()
        dd_sort_index=np.argsort(dd[:, ind])[::-1]
        dd_sort = dd[dd_sort_index]
        df_sort = pd.DataFrame(data=dd_sort, columns=titles)
        xx=df_sort[parameters['field1']].to_numpy()
        x=df[parameters['field1']].to_numpy()
        y=df['y'].to_numpy()
        print(parameters['testN'])
        if parameters['testN']==0:
            fpr1, tpr1, tresh1 = roc_curve(y,x)
            cutoff = np.max(tpr1 - fpr1)  # x YOUDEN INDEX
            print(tpr1 - fpr1)
            print(y,x)
            print(cutoff,'cutoff')
        else:
            cutoff=np.percentile(xx, parameters['testN'])
        print('cutoff: ',cutoff)
        df['class_cut']='positive'
        df['presabs']='false'
        df['class_cut'].iloc[np.where(x<=cutoff)[0]]='negative'
        df['presabs'].iloc[np.where(y==1)]='true'

        tp = np.where((df['class_cut']=='positive')&(df['presabs']=='true'))
        tn = np.where((df['class_cut']=='negative')&(df['presabs']=='true'))
        fp = np.where((df['class_cut']=='positive')&(df['presabs']=='false'))
        fn = np.where((df['class_cut']=='negative')&(df['presabs']=='false'))
        df['tptnfpfn']=0
        df['tptnfpfn'].iloc[tp[0]]=0
        df['tptnfpfn'].iloc[tn[0]]=1
        df['tptnfpfn'].iloc[fp[0]]=2
        df['tptnfpfn'].iloc[fn[0]]=3

        print('tp=', str((df['tptnfpfn'] == 0).sum()))
        print('tn=', str((df['tptnfpfn'] == 1).sum()))
        print('fp=', str((df['tptnfpfn'] == 2).sum()))
        print('fn=', str((df['tptnfpfn'] == 3).sum()))



        #df=parameters['df']
        #y_true=df['y']
        #scores=df['SI']
        ################################figure
        #fpr1, tpr1, tresh1 = roc_curve(y_true,scores)
        #norm=(scores-scores.min())/(scores.max()-scores.min())
        #r=roc_auc_score(y_true, scores)

        #idx = np.argmax(tpr1 - fpr1)  # x YOUDEN INDEX
        #suscept01 = copy(scores)
        #suscept01[scores > tresh1[idx]] = 1
        #suscept01[scores <= tresh1[idx]] = 0
        #f1_tot = f1_score(y_true, suscept01)
        #ck_tot = cohen_kappa_score(y_true, suscept01)
        #cm = confusion_matrix(y_true, y_pred)


     
        return df,nomi,crs

    

    def save(parameters):

        #print(parameters['nomi'])
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
            if field=='tptnfpfn':
                fields.append(QgsField(field, QVariant.Int))
            if  field=='class_cut':
                continue
            if  field=='presabs':
                continue
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
        
        strings=['geom', 'class_cut', 'presabs']
        columns_float = [item for item in df.columns if item not in strings]
        for i, row in df.iterrows():
             
            fet = QgsFeature()
            fet.setGeometry(QgsGeometry.fromWkt(row['geom']))
            fet.setAttributes(list(map(float,list(df.loc[ i, columns_float]))))
            writer.addFeature(fet)

        del writer

    def addmap(parameters):
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