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

import os
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
from qgis import *
import tempfile
from sz_module.scripts.utils import SZ_utils
from sz_module.scripts.algorithms import Algorithms,GAM_utils,CV_utils

class CoreAlgorithmGAM_trans():
   
    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Training input layer'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING3, 'Linear independent variables', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=True,type=QgsProcessingParameterField.Any,optional=True))
        self.addParameter(QgsProcessingParameterField(self.STRING, 'Ordinal independent variables', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=True,type=QgsProcessingParameterField.Any,optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER1, self.tr('Spline smoothing parameter'), type=QgsProcessingParameterNumber.Integer,defaultValue=10))
        self.addParameter(QgsProcessingParameterField(self.STRING8, 'Interacting variable A', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=False,type=QgsProcessingParameterField.Any,optional=True))
        self.addParameter(QgsProcessingParameterField(self.STRING9, 'Interacting variable B', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=False,type=QgsProcessingParameterField.Any,optional=True))
        self.addParameter(QgsProcessingParameterField(self.STRING1, 'Categorical independent variables', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=True,type=QgsProcessingParameterField.Any,optional=True))
        self.addParameter(QgsProcessingParameterEnum(self.STRING4, 'Family', options=['binomial','gaussian'], allowMultiple=False, usesStaticStrings=False, defaultValue=0))
        self.addParameter(QgsProcessingParameterEnum(self.STRING7, 'Scale (for Gaussian Family only)', options=['linear scale','log scale'], allowMultiple=False, usesStaticStrings=False, defaultValue=[],optional=True))
        self.addParameter(QgsProcessingParameterEnum(self.FEATURE_SCALING, 'Feature scaling', options=['Standard scaling','No scaling'], allowMultiple=False, usesStaticStrings=False, defaultValue=0))
        self.addParameter(QgsProcessingParameterField(self.STRING2, 'Field of dependent variable (0 for absence, > 0 for presence)', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT1, self.tr('Input layer for transferability'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None, optional=False))
        self.addParameter(QgsProcessingParameterField(self.STRING6, 'Year field to copy to output', parentLayerParameterName=self.INPUT1, defaultValue=None, allowMultiple=False, type=QgsProcessingParameterField.Any, optional=True))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT1, 'Output trans',fileFilter='GeoPackage (*.gpkg *.GPKG)', defaultValue=None))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT3, 'Outputs folder destination', defaultValue=None, createByDefault = True))

    def process(self, parameters, context, feedback, algorithm=None, classifier=None):
        self.f=tempfile.gettempdir()
        feedback = QgsProcessingMultiStepFeedback(1, feedback)
        results = {}
        outputs = {}

        family={'0':'binomial','1':'gaussian'}
        scale={'0':'linear_scale','1':'log_scale'}
        feature_scaling={'0':True,'1':False}

        source = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))
        parameters['covariates']=source.source()
        
        parameters['field3'] = self.parameterAsFields(parameters, self.STRING3, context)
        if parameters['field3'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING3))

        parameters['field1'] = self.parameterAsFields(parameters, self.STRING, context)
        if parameters['field1'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING))
        
        parameters['field2'] = self.parameterAsFields(parameters, self.STRING1, context)
        if parameters['field2'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING1))
        
        parameters['family'] = self.parameterAsString(parameters, self.STRING4, context)
        if parameters['family'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING4))
        
        parameters['scale'] = self.parameterAsString(parameters, self.STRING7, context)
        if parameters['scale'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING7))

        parameters['feature_scaling']=self.parameterAsString(parameters,self.FEATURE_SCALING,context)
        if parameters['feature_scaling'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters,self.FEATURE_SCALING))
        
        parameters['var_interaction_A'] = self.parameterAsFields(parameters, self.STRING8, context)
        if parameters['var_interaction_A'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING8))
        
        parameters['var_interaction_B'] = self.parameterAsFields(parameters, self.STRING9, context)
        if parameters['var_interaction_B'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING9))
        
        parameters['fieldlsd'] = self.parameterAsString(parameters, self.STRING2, context)
        if parameters['fieldlsd'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING2))
        
        parameters['num1'] = self.parameterAsInt(parameters, self.NUMBER1, context)
        if parameters['num1'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER1))
        
        source1 = self.parameterAsVectorLayer(parameters, self.INPUT1, context)
        if source1 is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT1))
        parameters['input1']=source1.source()

        parameters['year_field']=self.parameterAsString(parameters,self.STRING6,context)

        parameters['out'] = self.parameterAsFileOutput(parameters, self.OUTPUT1, context)
        if parameters['out'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT1))

        parameters['folder'] = self.parameterAsString(parameters, self.OUTPUT3, context)
        if parameters['folder'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT3))
        
        SZ_utils.make_directory({'path':parameters['folder']})
        
        tensor=[]
        if parameters['var_interaction_A'] != [] and parameters['var_interaction_B'] != []:
            tensor=[parameters['var_interaction_A'][0],parameters['var_interaction_B'][0]]

            alg_params = {
                'linear': parameters['field3'],
                'continuous': parameters['field1'],
                'categorical': parameters['field2'],
                'tensor': tensor,
            }
            if SZ_utils.check_validity(alg_params) is False:
                return {}

        if parameters['family'] not in family:
            raise QgsProcessingException('Select a GAM family for training.')
        if parameters['scale'] not in scale:
            raise QgsProcessingException('Select the target scale for training.')
        if not parameters['fieldlsd']:
            raise QgsProcessingException('Select the dependent variable for training.')

        predictor_names=parameters['field3']+parameters['field1']+parameters['field2']+tensor
        selected_family=family[parameters['family']]
        alg_params = {
            'INPUT_VECTOR_LAYER':parameters['covariates'],
            'nomi':predictor_names,
            'lsd':parameters['fieldlsd'],
            'family':selected_family,
            'scale':scale[parameters['scale']],
        }
        outputs['df'],outputs['crs']=SZ_utils.load_cv(self.f,alg_params)

        alg_params = {
            'linear':parameters['field3'],
            'continuous':parameters['field1'],
            'categorical':parameters['field2'],
            'tensor':tensor,
            'nomi':predictor_names,
            'spline':parameters['num1'],
        }
        outputs['splines'],outputs['dtypes']=GAM_utils.GAM_formula(alg_params)

        alg_params = {
            'testN':1,
            'fold':parameters['folder'],
            'nomi':predictor_names,
            'df':outputs['df'],
            'splines':outputs['splines'],
            'dtypes':outputs['dtypes'],
            'categorical':parameters['field2'],
            'linear':parameters['field3'],
            'continuous':parameters['field1'],
            'tensor':tensor,
            'family':selected_family,
            'feature_scaling':feature_scaling[parameters['feature_scaling']],
            'cv_method':'',
        }
        outputs['prob'],outputs['test_ind'],outputs['predictors_weights']=CV_utils.cross_validation(alg_params,algorithm,classifier)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}
        
        alg_params = {
            'INPUT_VECTOR_LAYER': parameters['input1'],
            'nomi':predictor_names,
            'lsd':'__sz_no_target__',
            'family':selected_family,
            'time':(
                parameters['year_field']
                if parameters['year_field'] not in predictor_names
                else None
            ),
        }
        try:
            outputs['df_trans'],outputs['crs_trans']=SZ_utils.load_cv(self.f,alg_params)
        except KeyError as error:
            raise QgsProcessingException(
                'The prediction layer does not contain every predictor required by '
                f"the GAM model: {', '.join(predictor_names)}"
            ) from error

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}
        
        alg_params = {
            'predictors_weights':outputs['predictors_weights'],
            'nomi':predictor_names,
            'family':selected_family,
            'df':outputs['df_trans']
        }
        try:
            outputs['trans']=Algorithms.GAM_transfer(alg_params)
        except ValueError as error:
            raise QgsProcessingException(str(error)) from error

        feedback.setCurrentStep(3)
        if feedback.isCanceled():
            return {}

        alg_params = {
            'df': outputs['trans'],
            'crs': outputs['crs_trans'],
            'OUT': parameters['out']
        }
        SZ_utils.save(alg_params)

        feedback.setCurrentStep(4)
        if feedback.isCanceled():
            return {}

        alg_params = {
            'df':outputs['df'],
            'crs':outputs['crs'],
            'OUT':os.path.join(parameters['folder'],'train.gpkg'),
        }
        SZ_utils.save(alg_params)

        feedback.setCurrentStep(5)
        if feedback.isCanceled():
            return {}
        results[self.OUTPUT1] = parameters['out']
        results[self.OUTPUT3] = parameters['folder']

        fileName = parameters['out']
        layer = QgsVectorLayer(fileName,"transfer","ogr")
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
            context.addLayerToLoadOnCompletion(sub_vlayer.id(), QgsProcessingContext.LayerDetails('transfer', context.project(),'LAYER'))

        feedback.setCurrentStep(6)
        if feedback.isCanceled():
            return {}
        
        fileName = os.path.join(parameters['folder'],'train.gpkg')
        layer = QgsVectorLayer(fileName,"train","ogr")
        subLayers =layer.dataProvider().subLayers()

        for subLayer in subLayers:
            name = subLayer.split('!!::!!')[1]
            uri = "%s|layername=%s" % (fileName,name,)
            sub_vlayer = QgsVectorLayer(uri,name,'ogr')
            if not sub_vlayer.isValid():
                print('layer failed to load')
            context.temporaryLayerStore().addMapLayer(sub_vlayer)
            context.addLayerToLoadOnCompletion(sub_vlayer.id(),QgsProcessingContext.LayerDetails('train',context.project(),'LAYER'))
        
        feedback.setCurrentStep(7)
        if feedback.isCanceled():
            return {}

        return results
