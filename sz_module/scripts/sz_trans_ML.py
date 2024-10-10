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
                       QgsProcessingContext,
                       QgsProcessingParameterEnum
                       )
from qgis.core import *
from qgis.utils import iface
from qgis import *
from processing.algs.gdal.GdalUtils import GdalUtils
import tempfile
from sz_module.scripts.utils import SZ_utils
from sz_module.scripts.algorithms import CV_utils,Algorithms
import os
from sz_module.utils import log



class CoreAlgorithmML_trans():

    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Input layer'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING, 'Independent variables', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=True,type=QgsProcessingParameterField.Any))
        self.addParameter(QgsProcessingParameterField(self.STRING2, 'Field of dependent variable (0 for absence, > 0 for presence)', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterEnum(self.STRING5, 'ML algorithm', options=['SVC','DT','RF'], allowMultiple=False, usesStaticStrings=False, defaultValue=[]))
        #self.addParameter(QgsProcessingParameterEnum(self.STRING3, 'CV method', options=['random CV','spatial CV','temporal CV (Time Series Split)','temporal CV (Leave One Out)', 'space-time CV (Leave One Out)'], allowMultiple=False, usesStaticStrings=False, defaultValue=[]))
        #self.addParameter(QgsProcessingParameterField(self.STRING4, 'Time field (for temporal CV only)', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=False,type=QgsProcessingParameterField.Any, optional=True ))
        #self.addParameter(QgsProcessingParameterNumber(self.NUMBER, self.tr('K-fold CV: K=1 to fit, k>1 to cross-validate for spatial CV only'), minValue=1,type=QgsProcessingParameterNumber.Integer,defaultValue=2,optional=True))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT1, self.tr('Input layer for transferability'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None, optional=False))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, 'Output test/fit',fileFilter='GeoPackage (*.gpkg *.GPKG)', defaultValue=None))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT3, 'Outputs folder destination', defaultValue=None, createByDefault = True))

    def process(self, parameters, context, feedback, algorithm=None, classifier=None):

        self.f=tempfile.gettempdir()
        feedback = QgsProcessingMultiStepFeedback(1, feedback)
        results = {}
        outputs = {}

        #cv_method={'0':'random','1':'spatial','2':'temporal_TSS','3':'temporal_LOO','4':'spacetime_LOO'}
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
        
        parameters['algorithm'] = self.parameterAsString(parameters, self.STRING5, context)
        if parameters['algorithm'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING5))
        
        source1 = self.parameterAsVectorLayer(parameters, self.INPUT1, context)
        parameters['input1']=source1.source()
        if parameters['input1'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT1))
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT1))

        # parameters['cv_method'] = self.parameterAsString(parameters, self.STRING3, context)
        # if parameters['cv_method'] is None:
        #     raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING3))
        
        # parameters['time'] = self.parameterAsString(parameters, self.STRING4, context)
        # if parameters['time'] is None:
        #     raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING4))

        # parameters['testN'] = self.parameterAsInt(parameters, self.NUMBER, context)
        # if parameters['testN'] is None:
        #     raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER))
 
        

        parameters['out'] = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        if parameters['out'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))

        parameters['folder'] = self.parameterAsString(parameters, self.OUTPUT3, context)
        if parameters['folder'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT3))
        
        SZ_utils.make_directory({'path':parameters['folder']})


        # if cv_method[parameters['cv_method']]=='random' or cv_method[parameters['cv_method']]=='spatial':
        #     parameters['time']=None
        # else:
        #     if parameters['time']=='':
        #         log(f"Time field is missing for temporal CV")
        #         raise RuntimeError("Time field is missing for temporal CV")

        parameters['testN']=1

        alg_params = {
            'INPUT_VECTOR_LAYER': parameters['covariates'],
            'nomi': parameters['field1'],
            'lsd' : parameters['fieldlsd'],
            #'time':parameters['time'],
        }

        outputs['df'],outputs['crs']=SZ_utils.load_cv(self.f,alg_params)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}
        
        #print(cv_method[parameters['cv_method']])

        alg_params = {
            #'field1': parameters['field1'],
            'testN':parameters['testN'],
            'fold':parameters['folder'],
            'nomi':parameters['field1'],
            'df':outputs['df'],
            #'cv_method':cv_method[parameters['cv_method']],
            #'time':parameters['time']
            'cv_method':'',
        }

        outputs['prob'],outputs['test_ind'],outputs['predictors_weights']=CV_utils.cross_validation(alg_params,algorithm,classifier[ML[parameters['algorithm']]])

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}
        
        alg_params = {
            'INPUT_VECTOR_LAYER': parameters['input1'],
            'field1': parameters['field1'],
            'lsd' : parameters['fieldlsd'],
            #'family':family[parameters['family']]
        }
        outputs['df_trans'],outputs['crs_trans']=SZ_utils.load_cv(self.f,alg_params)

        feedback.setCurrentStep(3)
        if feedback.isCanceled():
            return {}
        
        alg_params = {
            'predictors_weights':outputs['predictors_weights'],
            'nomi': parameters['field1'],
            #'family':family[parameters['family']],
            #'field1':parameters['field1'],
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

        alg_params = {
            'df': outputs['trans'],
            'OUT':parameters['folder']
        }
        SZ_utils.stampfit(alg_params)

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



