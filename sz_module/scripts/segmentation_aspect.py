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

"""
Calculation taken from the article:
Alvioli, M., Marchesini, I., Reichenbach, P., Rossi, M., Ardizzone, F., Fiorucci,
F., and Guzzetti, F.: Automatic delineation of geomorphological slope units with
r.slopeunits v1.0 and their optimization for landslide susceptibility modeling, 
Geosci. Model Dev., 9, 3975–3991, https://doi.org/10.5194/gmd-9-3975-2016, 2016. 
"""

__author__ = 'Giacomo Titti'
__date__ = '2021-07-01'
__copyright__ = '(C) 2021 by Giacomo Titti'

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
                       QgsProcessingParameterVectorDestination,
                       QgsProcessingParameterMultipleLayers)
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
#import geopandas as gpd
import pandas as pd
import tempfile
import os
import processing
#import libpysal as lp

class segmentationAspectAlgorithm():
   
    def init(self, config=None):
        self.addParameter(QgsProcessingParameterMultipleLayers(self.INPUT, self.tr('Slope Units'), layerType=QgsProcessing.TypeVectorPolygon, defaultValue=None))
        #self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Slope Units'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None,allowMultiple=True))
        #self.addParameter(QgsProcessingParameterField(self.STRING, 'area', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT1, self.tr('DEM'), defaultValue=None))
        #self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT, 'Folder destination', defaultValue=None, createByDefault = True))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, 'Output csv', '*.csv', defaultValue=None))

    def process(self, parameters, context, feedback):
        self.f=tempfile.gettempdir()
        #parameters['classes']=5
        # Use a multi-step feedback, so that individual child algorithm progress reports are adjusted for the
        # overall progress through the model
        feedback = QgsProcessingMultiStepFeedback(1, feedback)
        results = {}
        outputs = {}

        #parameters['SU']=[]
        parameters['SU'] = self.parameterAsLayerList(parameters, self.INPUT, context)
        #print(source)
        # for i in source:
        #     print(i)
        #     parameters['SU']=parameters['SU'].append(i.source())
        #     #if i is None:
        #     #    raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT)[i])

        # parameters['field_area'] = self.parameterAsString(parameters, self.STRING, context)
        # if parameters['field_area'] is None:
        #     raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING))

        parameters['dem'] = self.parameterAsRasterLayer(parameters, self.INPUT1, context)
        if parameters['dem'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT1))
        parameters['dem']=parameters['dem'].source()

        parameters['outcsv'] = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        if parameters['outcsv'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))
        #print(parameters['outcsv'])

        #QgsMessageLog.logMessage(parameters['lsi'], 'MyPlugin', level=Qgis.Info)
        #QgsMessageLog.logMessage(parameters['lsi'], 'MyPlugin', level=Qgis.Info)


        # Aspect
        alg_params = {
            'INPUT': parameters['dem'],
            'Z_FACTOR': 1,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['Aspect'] = processing.run('native:aspect', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        # SIN Raster calculator
        alg_params = {
            'BAND_A': 1,
            'BAND_B': None,
            'BAND_C': None,
            'BAND_D': None,
            'BAND_E': None,
            'BAND_F': None,
            'EXTRA': '',
            'FORMULA': 'sin(A)',
            'INPUT_A': outputs['Aspect']['OUTPUT'],
            'INPUT_B': parameters['dem'],
            'INPUT_C': parameters['dem'],
            'INPUT_D': parameters['dem'],
            'INPUT_E': parameters['dem'],
            'INPUT_F': parameters['dem'],
            'NO_DATA': None,
            'OPTIONS': '',
            'PROJWIN': None,
            'RTYPE': 5,  # Float32
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['SinRasterCalculator'] = processing.run('gdal:rastercalculator', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        # COS Raster calculator
        alg_params = {
            'BAND_A': 1,
            'BAND_B': None,
            'BAND_C': None,
            'BAND_D': None,
            'BAND_E': None,
            'BAND_F': None,
            'EXTRA': '',
            'FORMULA': 'cos(A)',
            'INPUT_A': outputs['Aspect']['OUTPUT'],
            'INPUT_B': parameters['dem'],
            'INPUT_C': parameters['dem'],
            'INPUT_D': parameters['dem'],
            'INPUT_E': parameters['dem'],
            'INPUT_F': parameters['dem'],
            'NO_DATA': None,
            'OPTIONS': '',
            'PROJWIN': None,
            'RTYPE': 5,  # Float32
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['CosRasterCalculator'] = processing.run('gdal:rastercalculator', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(3)
        if feedback.isCanceled():
            return {}
        

        table=pd.DataFrame({'name':[],'V':[],'I':[],'F':[]})
        for SU in parameters['SU']:

            # Add geometry attributes
            alg_params = {
                'CALC_METHOD': 0,  # Layer CRS
                'INPUT': SU,
                'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
            }
            outputs['AddGeometryAttributes'] = processing.run('qgis:exportaddgeometrycolumns', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

            # Sin Zonal statistics
            alg_params = {
                'COLUMN_PREFIX': 'sin_',
                'INPUT': outputs['AddGeometryAttributes']['OUTPUT'],
                'INPUT_RASTER': outputs['SinRasterCalculator']['OUTPUT'],
                'RASTER_BAND': 1,
                'STATISTICS': [1],  # Sum
                'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
            }
            outputs['SinZonalStatistics'] = processing.run('native:zonalstatisticsfb', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

            feedback.setCurrentStep(4)
            if feedback.isCanceled():
                return {}

            # Cos Zonal statistics
            alg_params = {
                'COLUMN_PREFIX': 'cos_',
                'INPUT': outputs['SinZonalStatistics']['OUTPUT'],
                'INPUT_RASTER': outputs['CosRasterCalculator']['OUTPUT'],
                'RASTER_BAND': 1,
                'STATISTICS': [1,0],  # Sum,Count
                'OUTPUT': self.f+'/zonalstat.shp'
            }
            outputs['CosZonalStatistics'] = processing.run('native:zonalstatisticsfb', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

            results['cv'] = outputs['CosZonalStatistics']['OUTPUT']

            feedback.setCurrentStep(5)
            if feedback.isCanceled():
                return {}

            alg_params = {
                'INPUT':self.f+'/zonalstat.shp',
            }
            outputs['gdp'],outputs['crs']= self.load(alg_params)

            feedback.setCurrentStep(7)
            if feedback.isCanceled():
                return {}
            
            ######################### V calculation

            alg_params = {
                'INPUT':outputs['gdp'],
                'FIELD':'area',
            }
            outputs['V']= self.V_calculator(alg_params)

            ########################### I calculation

            alg_params = {
                'INPUT':outputs['gdp'],
            }
            outputs['adj']=self.adjacent_matrix(alg_params)

            alg_params = {
                'INPUT':outputs['gdp'],
                'INPUT1': outputs['adj'],
            }
            outputs['I']= self.I_calculator(alg_params)

            table=table._append({'name': SU.name(),'V':outputs['V'],'I':outputs['I']}, ignore_index = True)

        alg_params = {
                'INPUT':table,
            }
        outputs['F']= self.F_calculator(alg_params)

        outputs['F'].to_csv(parameters['outcsv'])
        results['OUTPUT']=outputs['F']
        return results

    def load(self,parameters):
        layer = QgsVectorLayer(parameters['INPUT'], '', 'ogr')
        crs=layer.crs()
        gdp=gpd.read_file(parameters['INPUT'])
        return gdp,crs
    
    def V_calculator(self,parameters):
        df=parameters['INPUT']
        area=parameters['FIELD']
        cv=1-(np.sqrt(np.power(df['sin_sum'].to_numpy(),2)+np.power(df['cos_sum'].to_numpy(),2))/df['cos_count'].to_numpy())
        V = np.sum(cv*df[area].to_numpy())/np.sum(df[area].to_numpy())
        print('V: ',V)

        return V

    def adjacent_matrix(self,parameters):
        gdf_neighbors = lp.weights.Queen.from_dataframe(parameters['INPUT'])
        gdf_adj_mtx, gdf_adj_mtx_indices = gdf_neighbors.full()
        gdf_adj_list = gdf_neighbors.to_adjlist()
        return gdf_adj_list
    
    def I_calculator(self,parameters):
        df=parameters['INPUT']
        adj=parameters['INPUT1']

        aspect_mean_SU = np.arctan(df['sin_sum'].to_numpy()/df['cos_sum'].to_numpy()) #aspect mean per SU (formula 4)
        aspect_mean_Stu = np.arctan(np.sum(df['sin_sum'].to_numpy())/np.sum(df['cos_sum'].to_numpy())) #aspect mean per study area (formula 4)
        teta_SU = np.arctan((np.sin(aspect_mean_SU)-np.sin(aspect_mean_Stu))/(np.cos(aspect_mean_SU)-np.cos(aspect_mean_Stu))) #teta per solope unit (formula 7-8)
        first_denom = np.sum(np.power(teta_SU,2)) #first argument of denominator in formula 2
        second_denom = len(adj) #second argument of denominator in formula 2
        denom = first_denom * second_denom # denominator of formula 2
        N = df.shape[0] # numberof SU
        aspect_product = np.cos(teta_SU[adj['focal']])*np.cos(teta_SU[adj['neighbor']])+np.sin(teta_SU[adj['focal']])*np.sin(teta_SU[adj['neighbor']]) # formula 6
        numerator = N*np.sum(aspect_product) # numerator of formula 2
        I = numerator/denom # formula 2
        print('I: ',I)
        return I
    
    def F_calculator(self,parameters):# F calculation from formula 3
        df=parameters['INPUT']
        F=(np.max(df['V'].to_numpy())-df['V'].to_numpy())/(np.max(df['V'].to_numpy())-np.min(df['V'].to_numpy()))+(np.max(df['I'].to_numpy())-df['I'].to_numpy())/(np.max(df['I'].to_numpy())-np.min(df['I'].to_numpy()))
        df['F']=F
        return df