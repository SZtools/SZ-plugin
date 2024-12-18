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

import sys
sys.setrecursionlimit(10000)
from qgis.core import (QgsProcessing,
                       QgsProcessingException,
                       QgsProcessingMultiStepFeedback,
                       QgsProcessingParameterFileDestination,
                       QgsProcessingParameterVectorLayer,
                       QgsVectorLayer,
                       QgsProcessingParameterField,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterField,
                       QgsProcessingContext,
                       QgsProcessingParameterEnum
                       )
from qgis.core import *
from qgis import *
import tempfile
from sz_module.scripts.utils import SZ_utils
from sz_module.scripts.algorithms import CV_utils,Algorithms

class CoreAlgorithmML_trans():

    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Input layer'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING, 'Independent variables', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=True,type=QgsProcessingParameterField.Any))
        self.addParameter(QgsProcessingParameterField(self.STRING2, 'Field of dependent variable (0 for absence, > 0 for presence)', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterEnum(self.STRING5, 'ML algorithm', options=['SVC','DT','RF'], allowMultiple=False, usesStaticStrings=False, defaultValue=[]))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT1, self.tr('Input layer for transferability'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None, optional=False))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, 'Output test/fit',fileFilter='GeoPackage (*.gpkg *.GPKG)', defaultValue=None))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT3, 'Outputs folder destination', defaultValue=None, createByDefault = True))

    def process(self, parameters, context, feedback, algorithm=None, classifier=None):

        self.f=tempfile.gettempdir()
        feedback = QgsProcessingMultiStepFeedback(1, feedback)
        results = {}
        outputs = {}

        ML={'0':'SVC','1':'DT','2':'RF'}

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
        
        parameters['family'] = self.parameterAsString(parameters, self.STRING5, context)
        if parameters['family'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING5))
        
        source1 = self.parameterAsVectorLayer(parameters, self.INPUT1, context)
        parameters['input1']=source1.source()
        if parameters['input1'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT1))
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT1))

        parameters['out'] = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        if parameters['out'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))

        parameters['folder'] = self.parameterAsString(parameters, self.OUTPUT3, context)
        if parameters['folder'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT3))
        
        SZ_utils.make_directory({'path':parameters['folder']})

        parameters['testN']=1

        alg_params = {
            'INPUT_VECTOR_LAYER': parameters['covariates'],
            'nomi': parameters['field1'],
            'lsd' : parameters['fieldlsd'],
            'family':ML[parameters['family']],
        }

        outputs['df'],outputs['crs']=SZ_utils.load_cv(self.f,alg_params)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}
        
        alg_params = {
            'testN':parameters['testN'],
            'fold':parameters['folder'],
            'nomi':parameters['field1'],
            'df':outputs['df'],
            'family':ML[parameters['family']],
            'cv_method':'',
        }

        outputs['prob'],outputs['test_ind'],outputs['predictors_weights']=CV_utils.cross_validation(alg_params,algorithm,classifier[ML[parameters['family']]])

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}
        
        alg_params = {
            'INPUT_VECTOR_LAYER': parameters['input1'],
            'nomi': parameters['field1'],
            'lsd' : parameters['fieldlsd'],
            'family':ML[parameters['family']],
        }
        outputs['df_trans'],outputs['crs_trans']=SZ_utils.load_cv(self.f,alg_params)

        feedback.setCurrentStep(3)
        if feedback.isCanceled():
            return {}
        
        alg_params = {
            'predictors_weights':outputs['predictors_weights'],
            'nomi': parameters['field1'],
            'df':outputs['df_trans']
        }
        outputs['trans']=Algorithms.ML_transfer(alg_params)

        feedback.setCurrentStep(4)
        if feedback.isCanceled():
            return {}
        
        alg_params = {
            'df': outputs['trans'],
            'crs': outputs['crs_trans'],
            'OUT': parameters['folder']+'/trans.gpkg'
        }
        SZ_utils.save(alg_params)

        feedback.setCurrentStep(5)
        if feedback.isCanceled():
            return {}

        alg_params = {
            'df': outputs['df'],
            'crs': outputs['crs_trans'],
            'OUT': parameters['folder']+'/train.gpkg'
        }
        SZ_utils.save(alg_params)

        feedback.setCurrentStep(6)
        if feedback.isCanceled():
            return {}

        results['out'] = parameters['out']

        fileName = parameters['out']
        layer1 = QgsVectorLayer(fileName,"test","ogr")
        subLayers =layer1.dataProvider().subLayers()

        for subLayer in subLayers:
            name = subLayer.split('!!::!!')[1]
            uri = "%s|layername=%s" % (fileName, name,)
            # Create layer
            sub_vlayer = QgsVectorLayer(uri, name, 'ogr')
            if not sub_vlayer.isValid():
                print('layer failed to load')
            # Add layer to map
            context.temporaryLayerStore().addMapLayer(sub_vlayer)
            context.addLayerToLoadOnCompletion(sub_vlayer.id(), QgsProcessingContext.LayerDetails('test', context.project(),'LAYER1'))

        feedback.setCurrentStep(7)
        if feedback.isCanceled():
            return {}
        
        fileName = parameters['folder']+'/train.gpkg'
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
        
        feedback.setCurrentStep(4)
        if feedback.isCanceled():
            return {}

        return results



