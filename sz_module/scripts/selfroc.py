# -*- coding: utf-8 -*-

"""
/***************************************************************************
 classe
                                 A QGIS plugin
 susceptibility
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2021-07-01
        copyright            : (C) 2021 by Giacomo Titti
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

__author__ = 'Giacomo Titti'
__date__ = '2021-07-01'
__copyright__ = '(C) 2021 by Giacomo Titti'

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = '$Format:%H$'

from qgis.PyQt.QtCore import QCoreApplication
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
                       QgsProcessingParameterVectorDestination)
from sklearn.metrics import roc_curve, auc, f1_score, cohen_kappa_score, roc_auc_score
from copy import copy
import math
import operator
import matplotlib.pyplot as plt

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
import tempfile
import os

class rocGenerator(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    STRING = 'STRING'
    STRING2 = 'STRING2'
    OUTPUT3 = 'OUTPUT3'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return rocGenerator()

    def name(self):
        return 'ROC'

    def displayName(self):
        return self.tr('04 ROC')

    def group(self):
        return self.tr('04 Classify SI')

    def groupId(self):
        return '04 Classify SI'

    def shortHelpString(self):
        return self.tr("ROC curve creator")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Input layer'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING, 'SI field', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING2, 'Field of dependent variable (0 for absence, > 0 for presence)', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT3, 'Folder destination', defaultValue=None, createByDefault = True))

    def processAlgorithm(self, parameters, context, model_feedback):
        self.f=tempfile.gettempdir()
        #parameters['classes']=5
        # Use a multi-step feedback, so that individual child algorithm progress reports are adjusted for the
        # overall progress through the model
        feedback = QgsProcessingMultiStepFeedback(1, model_feedback)
        results = {}
        outputs = {}


        source = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        parameters['covariates']=source.source()
        if parameters['covariates'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        parameters['field1'] = self.parameterAsString(parameters, self.STRING, context)
        if parameters['field1'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING))

        parameters['fieldlsd'] = self.parameterAsString(parameters, self.STRING2, context)
        if parameters['fieldlsd'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING2))

        #parameters['w'] = self.parameterAsString(parameters, self.STRING3, context)
        #if parameters['w'] is None:
        #    raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING3))

        # parameters['edgesJenks'] = self.parameterAsFileOutput(parameters, self.OUTPUT1, context)
        # if parameters['edgesJenks'] is None:
        #     raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT1))
        #
        # parameters['edgesEqual'] = self.parameterAsFileOutput(parameters, self.OUTPUT2, context)
        # if parameters['edgesEqual'] is None:
        #     raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT2))

        # parameters['edgesGA'] = self.parameterAsFileOutput(parameters, self.OUTPUT3, context)
        # if parameters['edgesGA'] is None:
        #     raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT3))

        parameters['fold'] = self.parameterAsString(parameters, self.OUTPUT3, context)
        if parameters['fold'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT3))


        #QgsMessageLog.logMessage(parameters['lsi'], 'MyPlugin', level=Qgis.Info)
        #QgsMessageLog.logMessage(parameters['lsi'], 'MyPlugin', level=Qgis.Info)


        alg_params = {
            #'INPUT_RASTER_LAYER': parameters['Slope'],
            #'INPUT_EXTENT': parameters['Extension'],
            'INPUT_VECTOR_LAYER': parameters['covariates'],
            'field1': parameters['field1'],
            'lsd' : parameters['fieldlsd']
            #'W':parameters['w']
            #'INPUT_INT': parameters['BufferRadiousInPxl'],
            #'INPUT_INT_1': parameters['minSlopeAcceptable'],
        }
        outputs['gdp'],outputs['crs']=self.load(alg_params)

        #list_of_values=list(np.arange(10))
        self.list_of_values=outputs['gdp']['SI']
        QgsMessageLog.logMessage(str(len(self.list_of_values)), 'MyPlugin', level=Qgis.Info)

        # alg_params = {
        #     'OUTPUT': parameters['edgesEqual'],
        #     'NUMBER': parameters['classes']
        # }
        # #outputs['equal']=self.equal(alg_params)

        alg_params = {
            'df': outputs['gdp'],
            'OUT': parameters['fold']
        }
        self.roc(alg_params)

        # alg_params = {
        #     'OUTPUT': parameters['edgesJenks'],
        #     'NUMBER': parameters['classes']
        # }
        # #outputs['jenk']=self.jenk(alg_params)

        feedback.setCurrentStep(1)
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
        df=pd.DataFrame(dtype=float)
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
        gdp.to_csv(self.f+'/file.csv')
        del gdp
        gdp=pd.read_csv(self.f+'/file.csv')
        #print(feat)
        #print(gdp['S'].dtypes)
        gdp['ID']=np.arange(1,len(gdp.iloc[:,0])+1)
        df['SI']=gdp.loc[:,parameters['field1']]
        #df['w']=gdp.loc[:,parameters['W']]
        nomi=list(df.head())
        #print(list(df['Sf']),'1')
        lsd=gdp[parameters['lsd']]
        lsd[lsd>0]=1
        df['y']=lsd#.astype(int)
        df['ID']=gdp['ID']
        df['geom']=gdp['geom']
        df=df.dropna(how='any',axis=0)
        #df['ID']=df['ID'].astype('Int32')
        return df,crs



    def roc(self,parameters):
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
            fig.savefig(parameters['OUT']+'/roc.png')
        except:
            os.mkdir(parameters['OUT'])
            fig.savefig(parameters['OUT']+'/roc.png')

