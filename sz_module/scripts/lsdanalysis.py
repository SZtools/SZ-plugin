#!/usr/bin/python
#coding=utf-8
"""
/***************************************************************************
    statistic
        begin                : 2021-11
        copyright            : (C) 2021 by Giacomo Titti,
                               Padova, November 2021
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
    statistic
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
                       QgsProcessingParameterField
                       )
from qgis import processing
from osgeo import gdal,ogr,osr
import numpy as np
import math
import operator
import random
from qgis import *
##############################
import matplotlib.pyplot as plt
import csv
from processing.algs.gdal.GdalUtils import GdalUtils
#import plotly.express as px
#import chart_studio
import plotly.offline
import plotly.graph_objs as go

class statistic(QgsProcessingAlgorithm):
    # INPUT = 'INPUT'
    # OUTPUT = 'OUTPUT'
    # STRING = 'STRING'
    # FOLDER = 'FOLDER'

    # def tr(self, string):
    #     return QCoreApplication.translate('Processing', string)

    # def createInstance(self):
    #     return statistic()

    # def name(self):
    #     return 'attributes analysis'

    # def displayName(self):
    #     return self.tr('02 Attribute Table Statistics')

    # def group(self):
    #     return self.tr('01 Data preparation')

    # def groupId(self):
    #     return '01 Data preparation'

    # def shortHelpString(self):
    #     return self.tr("analysis of the points density distribution by attribute fields")

    def init(self, config=None):

        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Vector'), types=[QgsProcessing.TypeVectorAnyGeometry], defaultValue=None))
        #self.addParameter(QgsProcessingParameterField('fieldID', 'fieldID', type=QgsProcessingParameterField.Any, parentLayerParameterName='lsd', allowMultiple=False, defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING, 'ID field', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, 'Output csv', '*.csv', defaultValue=None))
        self.addParameter(QgsProcessingParameterFolderDestination(self.FOLDER, 'Folder destination', defaultValue=None,createByDefault = True))
        #parameters['fieldID']='ev_id'

    def process(self, parameters, context, model_feedback):
        feedback = QgsProcessingMultiStepFeedback(1, model_feedback)
        results = {}
        outputs = {}
        parameters['lsd'] = self.parameterAsVectorLayer(parameters, self.INPUT, context).source()
        if parameters['lsd'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        parameters['outcsv'] = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        if parameters['outcsv'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))
        print(parameters['outcsv'])

        parameters['fieldID'] = self.parameterAsString(parameters, self.STRING, context)
        if parameters['fieldID'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING))

        parameters['folder'] = self.parameterAsString(parameters, self.FOLDER, context)
        if parameters['folder'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.FOLDER))


        #cio
        #output, outputFormat = GdalUtils.ogrConnectionStringAndFormat(outFile, context)

        alg_params = {
            'OUTPUT': parameters['outcsv'],
            'ID': parameters['fieldID'],
            'INPUT2': parameters['lsd'],
            'PATH' : parameters['folder']
        }
        self.input(alg_params)

        ##############################################
        return{}

    def input(self,parameters):
        shapefile = parameters['INPUT2']
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataSource = driver.Open(shapefile, 0)
        layer = dataSource.GetLayer()
        layerDefinition = layer.GetLayerDefn()
        list_field=[]
        for i in range(layerDefinition.GetFieldCount()):
            fieldname=[layerDefinition.GetFieldDefn(i).GetName()]
            list_field=list_field+fieldname
        count=0
        valuesrow={}
        for feature in layer:
            valuesrow[count] = [feature.GetField(j) for j in list_field]
            count+=1
        count=0
        valuesfield={}
        for ii in range(len(list_field)):
            vf=[]
            for i in range(len(valuesrow.keys())):
                vf=vf+[valuesrow[i][ii]]
                count+=1
            valuesfield[list_field[ii]]=vf
        counter={}
        finder={}
        for ii in range(len(list_field)):
            #print(valuesfield['ls_trig'].count(indice['ls_trig'][0]))
            l=valuesfield[list_field[ii]]
            counter[list_field[ii]]=dict((x,l.count(x)) for x in set(l))
            chiavi=[]
            for j in range(len(counter[list_field[ii]])):
                chiavi=[counter[list_field[ii]].keys()]
            finder[list_field[ii]]=chiavi
        f={}
        for ii in range(len(list_field)):
            a=[]
            c=None
            b=list(finder[list_field[ii]][0])
            for jj in range(len(finder[list_field[ii]][0])):
                #print(np.asarray(valuesfield['ls_trig']))
                d=np.asarray(valuesfield[parameters['ID']])
                c=d[np.asarray(valuesfield[list_field[ii]])==b[jj]]
                a.append((c.tolist()))
            f[list_field[ii]]=a
        #print(counter['src_name'].keys())
        #print(counter['src_name'].values())
        w = csv.writer(open(parameters['OUTPUT'], "w"))
        w.writerow(['Field','Record','Count',parameters['ID']])
        for key, val in counter.items():
            count=0
            for key1, val1 in counter[key].items():
                #if key=='ls_trig':
                w.writerow([key, key1, val1,f[key][count]])
                count+=1

            fig = plt.figure()
            #ax = fig.add_axes()
            try:
                #listax=counter[key].keys()
                #listax=list(filter(None,listax))
                #x = listax
                x=list(counter[key].keys())
                #lista=counter[key].values()
                #lista=list(filter(None,lista))
                #y =lista
                y=list(counter[key].values())

                plt.bar(x, y, align='center', alpha=0.8)
                ##ax.bar(langs,students)
                plt.xticks(rotation=60)
                plt.grid(True)
                plt.title(key)
                #print(parameters['PATH'])
                plt.savefig(parameters['PATH']+'/fig'+key+'.png',bbox_inches='tight')
                fig=go.Figure()
                fig.add_trace(go.Bar( x=x, y=y))
                #fig.update_layout(title='<b>Panaro<b>')
                # Set x-axis title
                #fig.update_xaxes(title_text="<b>Data<b>")
                # Set y-axes titles
                #fig.update_yaxes(title_text="<b>Livello idromerico piezometri (m s.l.m.)</b>", secondary_y=False)
                #fig.update_yaxes(title_text="<b>Livello idrometrico (m s.l.m.)</b>", secondary_y=True)
                plotly.offline.plot(fig, filename=parameters['PATH']+'/fig'+key)
            except:
                print('error, skip field: ', key)
