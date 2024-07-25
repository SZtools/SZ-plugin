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
from sz_module.scripts.algorithms import CV_utils,GAM_utils
import os
from sz_module.utils import log

class CoreAlgorithmGAM_cv():

    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Input layer'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING3, 'Linear independent variables', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=True,type=QgsProcessingParameterField.Any,optional=True))
        self.addParameter(QgsProcessingParameterField(self.STRING, 'Ordinal independent variables', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=True,type=QgsProcessingParameterField.Any,optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER1, self.tr('Spline smoothing parameter'), type=QgsProcessingParameterNumber.Integer,defaultValue=10))
        self.addParameter(QgsProcessingParameterField(self.STRING8, 'Variables interaction A', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=False,type=QgsProcessingParameterField.Any,optional=True))
        self.addParameter(QgsProcessingParameterField(self.STRING9, 'Variables interaction B', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=False,type=QgsProcessingParameterField.Any,optional=True))
        self.addParameter(QgsProcessingParameterField(self.STRING1, 'Categorical independent variables', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=True,type=QgsProcessingParameterField.Any,optional=True))
        self.addParameter(QgsProcessingParameterEnum(self.STRING4, 'Family', options=['binomial','gaussian'], allowMultiple=False, usesStaticStrings=False, defaultValue=[]))
        self.addParameter(QgsProcessingParameterEnum(self.STRING7, 'Scale (for Gaussian Family only)', options=['linear scale','log scale'], allowMultiple=False, usesStaticStrings=False, defaultValue=[],optional=True))
        self.addParameter(QgsProcessingParameterField(self.STRING2, 'Field of dependent variable (0 for absence, > 0 for presence)', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterEnum(self.STRING5, 'CV method', options=['random CV','spatial CV','temporal CV (Time Series Split)','temporal CV (Leave One Out)', 'space-time CV (Leave One Out)'], allowMultiple=False, usesStaticStrings=False, defaultValue=[]))
        self.addParameter(QgsProcessingParameterField(self.STRING6, 'Time field (for temporal CV)', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=False,type=QgsProcessingParameterField.Any, optional=True ))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER, self.tr('K-fold CV: K=1 to fit, k>1 to cross-validate for spatial CV only'), minValue=1,type=QgsProcessingParameterNumber.Integer,defaultValue=2,optional=True))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, 'Output test/fit',fileFilter='GeoPackage (*.gpkg *.GPKG)', defaultValue=None))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT3, 'Outputs folder destination', defaultValue=None, createByDefault = True))

    def process(self, parameters, context, feedback, algorithm=None, classifier=None):

        self.f=tempfile.gettempdir()
        feedback = QgsProcessingMultiStepFeedback(1, feedback)
        results = {}
        outputs = {}

        family={'0':'binomial','1':'gaussian'}
        cv_method={'0':'random','1':'spatial','2':'temporal_TSS','3':'temporal_LOO','4':'spacetime_LOO'}
        gauss_scale={'0':'linear scale','1':'log scale'}

        source = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        parameters['covariates']=source.source()
        if parameters['covariates'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        parameters['field1'] = self.parameterAsFields(parameters, self.STRING, context)
        if parameters['field1'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING))
        
        parameters['field2'] = self.parameterAsFields(parameters, self.STRING1, context)
        if parameters['field2'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING1))
        
        parameters['field3'] = self.parameterAsFields(parameters, self.STRING3, context)
        if parameters['field3'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING3))

        parameters['fieldlsd'] = self.parameterAsString(parameters, self.STRING2, context)
        if parameters['fieldlsd'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING2))
        
        parameters['family'] = self.parameterAsString(parameters, self.STRING4, context)
        if parameters['family'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING4))
        
        parameters['gauss_scale'] = self.parameterAsString(parameters, self.STRING7, context)
        if parameters['gauss_scale'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING7))
        
        parameters['var_interaction_A'] = self.parameterAsFields(parameters, self.STRING8, context)
        if parameters['var_interaction_A'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING8))
        
        parameters['var_interaction_B'] = self.parameterAsFields(parameters, self.STRING9, context)
        if parameters['var_interaction_B'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING9))
        
        parameters['field1'] = self.parameterAsFields(parameters, self.STRING, context)
        if parameters['field1'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING))
        
        parameters['num1'] = self.parameterAsInt(parameters, self.NUMBER1, context)
        if parameters['num1'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER1))
        
        parameters['cv_method'] = self.parameterAsString(parameters, self.STRING5, context)
        if parameters['cv_method'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING5))
        
        parameters['time'] = self.parameterAsString(parameters, self.STRING6, context)
        if parameters['time'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING6))

        parameters['testN'] = self.parameterAsInt(parameters, self.NUMBER, context)
        if parameters['testN'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER))
 
        parameters['out'] = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        if parameters['out'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))

        parameters['folder'] = self.parameterAsString(parameters, self.OUTPUT3, context)
        if parameters['folder'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT3))
        
        if not os.path.exists(parameters['folder']):
            os.mkdir(parameters['folder'])

        if cv_method[parameters['cv_method']]=='random' or cv_method[parameters['cv_method']]=='spatial':
            parameters['time']=None
        else:
            if parameters['time']=='':
                log(f"Time field is missing for temporal CV")
                raise RuntimeError("Time field is missing for temporal CV")
        
        if parameters['var_interaction_A'] != [] and parameters['var_interaction_B'] != []: 
            tensor=[parameters['var_interaction_A'][0],parameters['var_interaction_B'][0]]

            alg_params = {
                'linear': parameters['field3'],
                'continuous': parameters['field1'],
                'categorical': parameters['field2'],
                'tensor': tensor,
            }
            if SZ_utils.check_validity(alg_params) is False:
                return ''
        else:
            tensor=[]
        
        alg_params = {
            'INPUT_VECTOR_LAYER': parameters['covariates'],
            'field1': parameters['field3']+parameters['field1']+parameters['field2']+tensor,
            'lsd' : parameters['fieldlsd'],
            'family':family[parameters['family']],
            'time':parameters['time'],
            'gauss_scale':gauss_scale[parameters['gauss_scale']],
        }

        outputs['df'],outputs['crs']=SZ_utils.load_cv(self.f,alg_params)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}
        
        alg_params = {
            'linear': parameters['field3'],
            'continuous': parameters['field1'],
            'categorical': parameters['field2'],
            'tensor': tensor,
            'nomi': parameters['field3']+parameters['field1']+parameters['field2']+tensor,
            'spline': parameters['num1'],
            #'var_interaction_A':parameters['var_interaction_A'],
            #'var_interaction_B':parameters['var_interaction_B'],
        }

        outputs['splines'],outputs['dtypes']=GAM_utils.GAM_formula(alg_params)

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        alg_params = {
            'testN':parameters['testN'],
            'fold':parameters['folder'],
            'nomi':parameters['field3']+parameters['field1']+parameters['field2']+tensor,
            'df':outputs['df'],
            'splines':outputs['splines'],
            'dtypes':outputs['dtypes'],
            'categorical':parameters['field2'],
            'linear':parameters['field3'],
            'continuous':parameters['field1'],
            'tensor': tensor,
            'family':family[parameters['family']],
            'cv_method':cv_method[parameters['cv_method']],
            'time':parameters['time']
        }

        outputs['prob'],outputs['test_ind'],outputs['gam']=CV_utils.cross_validation(alg_params,algorithm,classifier)

        feedback.setCurrentStep(3)
        if feedback.isCanceled():
            return {}

        if parameters['testN']>0:
            alg_params = {
                'df': outputs['df'],
                'crs': outputs['crs'],
                'OUT': parameters['out']
            }
            SZ_utils.save(alg_params)

        feedback.setCurrentStep(4)
        if feedback.isCanceled():
            return {}

        alg_params = {
            'test_ind': outputs['test_ind'],
            'df': outputs['df'],
            'OUT':parameters['folder']
        }
        SZ_utils.stamp_cv(alg_params)

        feedback.setCurrentStep(5)
        if feedback.isCanceled():
            return {}

        results['out'] = parameters['out']


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

        feedback.setCurrentStep(5)
        if feedback.isCanceled():
            return {}
    
    

        return results




