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

from qgis.core import (QgsProcessing,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingMultiStepFeedback,
                       QgsProcessingParameterFileDestination,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterField,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterField
                       )
from osgeo import ogr
import numpy as np
from qgis import *
import matplotlib.pyplot as plt
import csv
import plotly.offline
import plotly.graph_objs as go
from .utils import SZ_utils

class statistic(QgsProcessingAlgorithm):

    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Input layer'), types=[QgsProcessing.TypeVectorAnyGeometry], defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING, 'ID', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, 'Output csv', '*.csv', defaultValue=None))
        self.addParameter(QgsProcessingParameterFolderDestination(self.FOLDER, 'Folder destination', defaultValue=None,createByDefault = True))

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
        parameters['fieldID'] = self.parameterAsString(parameters, self.STRING, context)
        if parameters['fieldID'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING))
        parameters['folder'] = self.parameterAsString(parameters, self.FOLDER, context)
        if parameters['folder'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.FOLDER))

        SZ_utils.make_directory({'path':parameters['folder']})

        alg_params = {
            'OUTPUT': parameters['outcsv'],
            'ID': parameters['fieldID'],
            'INPUT2': parameters['lsd'],
            'PATH' : parameters['folder']
        }
        Functions.input(alg_params)
        return{}

class Functions():
    def input(parameters):
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
                d=np.asarray(valuesfield[parameters['ID']])
                c=d[np.asarray(valuesfield[list_field[ii]])==b[jj]]
                a.append((c.tolist()))
            f[list_field[ii]]=a
        w = csv.writer(open(parameters['OUTPUT'], "w"))
        w.writerow(['Field','Record','Count',parameters['ID']])
        for key, val in counter.items():
            count=0
            for key1, val1 in counter[key].items():
                w.writerow([key, key1, val1,f[key][count]])
                count+=1
            fig = plt.figure()
            try:
                x=list(counter[key].keys())
                y=list(counter[key].values())
                plt.bar(x, y, align='center', alpha=0.8)
                plt.xticks(rotation=60)
                plt.grid(True)
                plt.title(key)
                plt.savefig(parameters['PATH']+'/fig'+key+'.png',bbox_inches='tight')
                fig=go.Figure()
                fig.add_trace(go.Bar( x=x, y=y))
                plotly.offline.plot(fig, filename=parameters['PATH']+'/fig'+key)
            except:
                print('error, skip field: ', key)