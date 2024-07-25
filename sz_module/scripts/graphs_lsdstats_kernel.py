#!/usr/bin/python
#coding=utf-8
"""
/***************************************************************************
    statistickernel
        begin                : 2021-11
        copyright            : (C) 2021 by Giacomo Titti,
                               Padova, November 2021
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
    statistickernel
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

#in input la tabella attributi costruita usando le funzioni
#di saga: grid statistic for points, add raster values to points

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
                       QgsProcessingParameterFolderDestination
                       )
from qgis import processing
import numpy as np
import math
import operator
import random
from qgis import *
##############################
#from osgeo import gdal,osr,ogr
import sys,os
import math
import csv
import matplotlib.pyplot as plt
from osgeo import gdal,osr,ogr
import sys
import math
import csv
from .utils import SZ_utils

# import plotly.tools as plotly_tools
# import plotly.graph_objs as go
# import chart_studio.plotly as py
# from IPython.display import HTML
# import chart_studio
# import plotly.offline
# from plotly.subplots import make_subplots

class statistickernel(QgsProcessingAlgorithm):
    # INPUT = 'INPUT'
    # STRING = 'STRING'
    # FOLDER = 'FOLDER'

    # def tr(self, string):
    #     return QCoreApplication.translate('Processing', string)

    # def createInstance(self):
    #     return statistickernel()

    # def name(self):
    #     return 'points kernel graphs'

    # def displayName(self):
    #     return self.tr('04 Points kernel graphs')

    # def group(self):
    #     return self.tr('01 Data preparation')

    # def groupId(self):
    #     return '01 Data preparation'

    # def shortHelpString(self):
    #     return self.tr("It creates graphs of '03 Points Kernel Statistics' output")

    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Points'), types=[QgsProcessing.TypeVectorPoint], defaultValue=None))
        #self.addParameter(QgsProcessingParameterNumber('BufferRadiousInPxl', 'Buffer radiou in pixels', type=QgsProcessingParameterNumber.Integer, defaultValue = 2,  minValue=1))
        #self.addParameter(QgsProcessingParameterFileDestination('outcsv', 'outcsv', '*.csv', defaultValue='/media/jack/MyBook/irpi/OutputPaperBRI/Travis/lsd06-2020.csv'))
        #self.addParameter(QgsProcessingParameterField('fieldID', 'fieldID', type=QgsProcessingParameterField.Any, parentLayerParameterName='lsd', allowMultiple=False, defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING, 'ID field', parentLayerParameterName=self.INPUT, defaultValue=None))
        #self.addParameter(QgsProcessingParameterString(self.STRING, 'fieldID', multiLine=False, defaultValue=None))
        self.addParameter(QgsProcessingParameterFolderDestination(self.FOLDER, 'Folder destination of the graphs', defaultValue=None, createByDefault = True))

        #self.addParameter(QgsProcessingParameterRasterLayer('slope', 'raster', defaultValue='/home/irpi/SinoItalian_Lab/Tier1_SouthAsia/Data/Slope_SE_250m_3857Travis.tif'))
        #self.addParameter(QgsProcessingParameterVectorDestination('Out', 'out', type=QgsProcessing.TypeVectorAnyGeometry, createByDefault=True, defaultValue=None))
        #parameters['fieldID']='ev_id'

    def process(self, parameters, context, model_feedback):
        feedback = QgsProcessingMultiStepFeedback(1, model_feedback)
        results = {}
        outputs = {}

        parameters['lsd'] = self.parameterAsVectorLayer(parameters, self.INPUT, context).source()
        if parameters['lsd'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        parameters['fieldID'] = self.parameterAsString(parameters, self.STRING, context)
        if parameters['fieldID'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING))

        parameters['folder'] = self.parameterAsString(parameters, self.FOLDER, context)
        if parameters['folder'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.FOLDER))
        
        SZ_utils.make_directory({'path':parameters['folder']})


        alg_params = {
            #'OUTPUT': parameters['outcsv'],
            'ID': parameters['fieldID'],
            'INPUT2': parameters['lsd'],
            'OUT': parameters['folder']
        }
        Functions.input(alg_params)

        #vlayer = QgsVectorLayer(parameters['Out1'], 'vector', "ogr")
        #QgsProject.instance().addMapLayer(vlayer)
        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}
        return results

        ##############################################

class Functions():
    def input(parameters):
        layer = QgsVectorLayer(parameters['INPUT2'], 'vector', "ogr")

        list_field=[]
        fields=layer.fields()
        for i in range(layer.fields().count()):
            fieldname=[fields[i].name()]
            list_field=list_field+fieldname
        count=0
        valuesrow={}
        for feature in layer.getFeatures():
            valuesrow[count] = [feature.attribute(j) for j in list_field]
            count+=1
        count=0
        valuesfield={}
        for ii in range(len(list_field)):
            vf=[]
            for i in range(len(valuesrow.keys())):
                vf=vf+[valuesrow[i][ii]]
                count+=1
            valuesfield[list_field[ii]]=vf
        #print(valuesfield['mean']==)
        matrice=np.array([np.asarray(valuesfield['real']),np.asarray(valuesfield['max']),
        valuesfield['min'],valuesfield['std'],valuesfield['sum'],
        valuesfield['average'],valuesfield['range']])
        #print(valuesfield['mean'])

        matrice = matrice[::, matrice[0,].argsort()]
        # Plot
        lista=['real','max','min','std','sum','average','range']
        for i in range(7):
            fig=plt.figure()


            #plt.plot(np.arange(len(valuesfield['ev_id'])), matrice[7,:])
            #plt.title('')
            plt.xlabel('points')
            plt.ylabel('raster value')

            plt.grid()

            plt.plot(np.arange(len(valuesfield['id'])), matrice[i,:])
            plt.legend([lista[i]],loc="upper left")
            print(parameters['OUT']+'/fig'+lista[i]+'.pdf')
            plt.savefig(parameters['OUT']+'/fig'+lista[i]+'.pdf',bbox_inches='tight')

        