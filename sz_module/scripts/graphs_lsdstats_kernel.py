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
                       QgsProcessingParameterVectorLayer,
                       QgsVectorLayer,
                       QgsProcessingParameterField,
                       QgsProcessingParameterFolderDestination
                       )
import numpy as np
from qgis import *
import math
import matplotlib.pyplot as plt
import math
from .utils import SZ_utils

class statistickernel(QgsProcessingAlgorithm):

    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Points'), types=[QgsProcessing.TypeVectorPoint], defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING, 'ID field', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterFolderDestination(self.FOLDER, 'Folder destination of the graphs', defaultValue=None, createByDefault = True))

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
            'ID': parameters['fieldID'],
            'INPUT2': parameters['lsd'],
            'OUT': parameters['folder']
        }
        Functions.input(alg_params)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}
        return results

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
        matrice=np.array([np.asarray(valuesfield['real']),np.asarray(valuesfield['max']),
        valuesfield['min'],valuesfield['std'],valuesfield['sum'],
        valuesfield['average'],valuesfield['range']])
        matrice = matrice[::, matrice[0,].argsort()]
        # Plot
        lista=['real','max','min','std','sum','average','range']
        for i in range(7):
            fig=plt.figure()
            plt.xlabel('points')
            plt.ylabel('raster value')
            plt.grid()
            plt.plot(np.arange(len(valuesfield['id'])), matrice[i,:])
            plt.legend([lista[i]],loc="upper left")
            print(parameters['OUT']+'/fig'+lista[i]+'.pdf')
            plt.savefig(parameters['OUT']+'/fig'+lista[i]+'.pdf',bbox_inches='tight')