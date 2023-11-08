#!/usr/bin/python
#coding=utf-8
"""
/***************************************************************************
    LRAlgorithm
        begin                : 2021-11
        copyright            : (C) 2021 by Giacomo Titti,
                               Padova, November 2021
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
    LRAlgorithm
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
from qgis.core import (QgsProcessing,
                       QgsProcessingException,
                       QgsProcessingMultiStepFeedback,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFileDestination,
                       QgsProcessingParameterVectorLayer,
                       QgsVectorLayer,
                       QgsProcessingParameterField,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterField,
                       QgsProcessingContext
                       )
from qgis.core import *
from qgis.utils import iface
from qgis import *
from processing.algs.gdal.GdalUtils import GdalUtils
import plotly.graph_objs as go
import pandas as pd
import tempfile
from sz_module.scripts.utils import SZ_utils
from sz_module.scripts.algorithms import Algorithms

class CoreAlgorithmGAM():
   
    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Input layer'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING3, 'Linear independent variables', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=True,type=QgsProcessingParameterField.Any,optional=True))
        self.addParameter(QgsProcessingParameterField(self.STRING, 'Continuous independent variables', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=True,type=QgsProcessingParameterField.Any,optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER1, self.tr('Splines grade'), type=QgsProcessingParameterNumber.Integer,defaultValue=10))
        self.addParameter(QgsProcessingParameterField(self.STRING1, 'Categorical independent variables', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=True,type=QgsProcessingParameterField.Any,optional=True))
        self.addParameter(QgsProcessingParameterField(self.STRING2, 'Field of dependent variable (0 for absence, > 0 for presence)', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER, self.tr('Percentage of test sample (0 to fit, > 0 to cross-validate)'), type=QgsProcessingParameterNumber.Integer,defaultValue=30))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, 'Output test [mandatory if Test percentage > 0]',fileFilter='GeoPackage (*.gpkg *.GPKG)', defaultValue=None))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT1, 'Output train/fit',fileFilter='GeoPackage (*.gpkg *.GPKG)', defaultValue=None))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT3, 'Outputs folder destination', defaultValue=None, createByDefault = True))

    def process(self, parameters, context, feedback, algorithm=None, classifier=None):
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
        
        parameters['field3'] = self.parameterAsFields(parameters, self.STRING3, context)
        if parameters['field3'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING3))

        parameters['field1'] = self.parameterAsFields(parameters, self.STRING, context)
        if parameters['field1'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING))
        
        parameters['field2'] = self.parameterAsFields(parameters, self.STRING1, context)
        if parameters['field2'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING1))

        parameters['fieldlsd'] = self.parameterAsString(parameters, self.STRING2, context)
        if parameters['fieldlsd'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING2))
        
        parameters['num1'] = self.parameterAsInt(parameters, self.NUMBER1, context)
        if parameters['num1'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER1))

        parameters['testN'] = self.parameterAsInt(parameters, self.NUMBER, context)
        if parameters['testN'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER))

        parameters['out'] = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        if parameters['out'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))

        parameters['out1'] = self.parameterAsFileOutput(parameters, self.OUTPUT1, context)
        if parameters['out1'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT1))

        parameters['folder'] = self.parameterAsString(parameters, self.OUTPUT3, context)
        if parameters['folder'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT3))
        
        alg_params = {
            'INPUT_VECTOR_LAYER': parameters['covariates'],
            'field1': parameters['field3']+parameters['field1']+parameters['field2'],
            'lsd' : parameters['fieldlsd'],
            'testN':parameters['testN']
        }
        outputs['train'],outputs['testy'],outputs['nomes'],outputs['crs']=SZ_utils.load_simple(self.f,alg_params)

        alg_params = {
            'linear': parameters['field3'],
            'continuous': parameters['field1'],
            'categorical': parameters['field2'],
            'nomi': outputs['nomes'],
            'spline': parameters['num1']
        }
        outputs['splines'],outputs['dtypes']=SZ_utils.GAM_formula(alg_params)    


        alg_params = {
            'train': outputs['train'],
            'testy': outputs['testy'],
            'nomi':outputs['nomes'],
            'testN':parameters['testN'],
            'fold':parameters['folder'],
            'splines':outputs['splines'],
            'dtypes':outputs['dtypes']
        }
        outputs['trainsi'],outputs['testsi']=algorithm(alg_params)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        if parameters['testN']>0:
            alg_params = {
                'df': outputs['testsi'],
                'crs': outputs['crs'],
                'OUT': parameters['out']
            }
            SZ_utils.save(alg_params)

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        alg_params = {
            'df': outputs['trainsi'],
            'crs': outputs['crs'],
            'OUT': parameters['out1']
        }
        SZ_utils.save(alg_params)

        if parameters['testN']==0:
            alg_params = {
                'df': outputs['trainsi'],
                'OUT':parameters['folder']

            }
            SZ_utils.stampfit(alg_params)
        else:
            alg_params = {
                'train': outputs['trainsi'],
                'test': outputs['testsi'],
                'OUT':parameters['folder']
            }
            SZ_utils.stamp_simple(alg_params)

        feedback.setCurrentStep(3)
        if feedback.isCanceled():
            return {}
        results['out'] = parameters['out']
        results['out1'] = parameters['out1']

        if parameters['testN']>0:
            fileName = parameters['out1']
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
                context.addLayerToLoadOnCompletion(sub_vlayer.id(), QgsProcessingContext.LayerDetails('train', context.project(),'LAYER'))


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

        else:
            fileName = parameters['out1']
            layer = QgsVectorLayer(fileName,"fitting","ogr")
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
                context.addLayerToLoadOnCompletion(sub_vlayer.id(), QgsProcessingContext.LayerDetails('fitting', context.project(),'LAYER'))

        feedback.setCurrentStep(3)
        if feedback.isCanceled():
            return {}

        return results

    