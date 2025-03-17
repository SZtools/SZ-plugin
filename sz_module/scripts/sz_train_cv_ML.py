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
from sz_module.scripts.algorithms import CV_utils
from sz_module.utils import log
from sz_module.test.utils import load_test_input

class CoreAlgorithm_cv():

    def init(self, config=None):
        if os.environ.get('DEBUG')=='True':
            data=load_test_input("ML_cv")
            self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Input layer'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=data[self.INPUT]))
            self.addParameter(QgsProcessingParameterField(self.STRING, 'Independent variables', parentLayerParameterName=self.INPUT, defaultValue=data[self.STRING], allowMultiple=True,type=QgsProcessingParameterField.Any))
            self.addParameter(QgsProcessingParameterField(self.STRING2, 'Field of dependent variable (0 for absence, > 0 for presence)', parentLayerParameterName=self.INPUT, defaultValue=data[self.STRING2]))
            self.addParameter(QgsProcessingParameterEnum(self.STRING5, 'ML algorithm', options=['SVM Classifier','DT Classifier','RF Classfier','SVM Regressor','DT Regressor','RF Regressor'], allowMultiple=False, usesStaticStrings=False, defaultValue=data[self.STRING5]))
            self.addParameter(QgsProcessingParameterEnum(self.STRING6, 'Class weight (for RF classifier and DT classifier)', options=['Balanced','Not balanced'], allowMultiple=False, usesStaticStrings=False, defaultValue=data[self.STRING6]),optional=True)
            self.addParameter(QgsProcessingParameterNumber(self.NUMBER1, self.tr('Estimators (for RF classifier and RF regressor)'), minValue=1,type=QgsProcessingParameterNumber.Integer,optional=True,defaultValue=data[self.NUMBER1]))
            self.addParameter(QgsProcessingParameterEnum(self.STRING3, 'CV method', options=['random CV','spatial CV','temporal CV (Time Series Split)','temporal CV (Leave One Out)', 'space-time CV (Leave One Out)'], allowMultiple=False, usesStaticStrings=False, defaultValue=data[self.STRING3]))
            self.addParameter(QgsProcessingParameterField(self.STRING4, 'Time field (for temporal CV only)', parentLayerParameterName=self.INPUT, defaultValue=data[self.STRING4], allowMultiple=False,type=QgsProcessingParameterField.Any, optional=True ))
            self.addParameter(QgsProcessingParameterNumber(self.NUMBER, self.tr('K-fold CV: K=1 to fit, k>1 to cross-validate for spatial CV only'), minValue=1,type=QgsProcessingParameterNumber.Integer,optional=True,defaultValue=data[self.NUMBER]))
            self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, 'Output test/fit',fileFilter='GeoPackage (*.gpkg *.GPKG)', defaultValue=data[self.OUTPUT]))
            self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT3, 'Outputs folder destination', defaultValue=data[self.OUTPUT3], createByDefault = True))
        else:
            self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Input layer'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
            self.addParameter(QgsProcessingParameterField(self.STRING, 'Independent variables', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=True,type=QgsProcessingParameterField.Any))
            self.addParameter(QgsProcessingParameterField(self.STRING2, 'Field of dependent variable (0 for absence, > 0 for presence)', parentLayerParameterName=self.INPUT, defaultValue=None))
            self.addParameter(QgsProcessingParameterEnum(self.STRING5, 'ML algorithm', options=['SVM Classifier','DT Classifier','RF Classfier','SVM Regressor','DT Regressor','RF Regressor'], allowMultiple=False, usesStaticStrings=False, defaultValue=[]))
            self.addParameter(QgsProcessingParameterEnum(self.STRING6, 'Class weight (for RF classifier and DT classifier)', options=['Balanced','Not balanced'], allowMultiple=False, usesStaticStrings=False, defaultValue=[]),optional=True)
            self.addParameter(QgsProcessingParameterNumber(self.NUMBER1, self.tr('Estimators (for RF classifier and RF regressor)'), minValue=1,type=QgsProcessingParameterNumber.Integer,optional=True,defaultValue=10000))
            self.addParameter(QgsProcessingParameterEnum(self.STRING3, 'CV method', options=['random CV','spatial CV','temporal CV (Time Series Split)','temporal CV (Leave One Out)', 'space-time CV (Leave One Out)'], allowMultiple=False, usesStaticStrings=False, defaultValue=[]))
            self.addParameter(QgsProcessingParameterField(self.STRING4, 'Time field (for temporal CV only)', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=False,type=QgsProcessingParameterField.Any, optional=True ))
            self.addParameter(QgsProcessingParameterNumber(self.NUMBER, self.tr('K-fold CV: K=1 to fit, k>1 to cross-validate for spatial CV only'), minValue=1,type=QgsProcessingParameterNumber.Integer,defaultValue=2,optional=True))
            self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, 'Output test/fit',fileFilter='GeoPackage (*.gpkg *.GPKG)', defaultValue=None))
            self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT3, 'Outputs folder destination', defaultValue=None, createByDefault = True))
            
    def process(self, parameters, context, feedback, algorithm=None, classifier=None):
        
        self.f=tempfile.gettempdir()
        feedback = QgsProcessingMultiStepFeedback(1, feedback)
        results = {}
        outputs = {}

        cv_method={'0':'random','1':'spatial','2':'temporal_TSS','3':'temporal_LOO','4':'spacetime_LOO'}
        ML={'0':'SVM_classifier','1':'DT_classifier','2':'RF_classifier','3':'SVM_regressor','4':'DT_regressor','5':'RF_regressor'}
        weight={'0':'balanced','1':None}

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
        
        parameters['estimators'] = self.parameterAsInt(parameters, self.NUMBER1, context)
        if parameters['estimators'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER1))
            
        parameters['weight'] = self.parameterAsString(parameters, self.STRING6, context)
        if parameters['weight'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING6))
        
        parameters['cv_method'] = self.parameterAsString(parameters, self.STRING3, context)
        if parameters['cv_method'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING3))
        
        parameters['time'] = self.parameterAsString(parameters, self.STRING4, context)
        if parameters['time'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING4))

        parameters['testN'] = self.parameterAsInt(parameters, self.NUMBER, context)
        if parameters['testN'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER))
 
        parameters['out'] = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        if parameters['out'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))

        parameters['folder'] = self.parameterAsString(parameters, self.OUTPUT3, context)
        if parameters['folder'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT3))
        
        SZ_utils.make_directory({'path':parameters['folder']})

        if cv_method[parameters['cv_method']]=='random' or cv_method[parameters['cv_method']]=='spatial':
            parameters['time']=None
        else:
            if parameters['time']=='':
                log(f"Time field is missing for temporal CV")
                raise RuntimeError("Time field is missing for temporal CV")
            
        if ML[parameters['family']]=='RF_classifier' or ML[parameters['family']]=='RF_regressor':
            classifier.set_params(n_estimators=parameters['estimators'])
        if ML[parameters['family']]=='RF_classifier' or ML[parameters['family']]=='DT_classifier':
            classifier.set_params(class_weight=weight[parameters['weight']])

        alg_params = {
            'INPUT_VECTOR_LAYER': parameters['covariates'],
            'nomi': parameters['field1'],
            'lsd' : parameters['fieldlsd'],
            'time':parameters['time'],
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
            'cv_method':cv_method[parameters['cv_method']],
            'time':parameters['time'],
            'family':ML[parameters['family']],
        }

        outputs['prob'],outputs['test_ind'],outputs['gam']=CV_utils.cross_validation(alg_params,algorithm,classifier[ML[parameters['family']]])

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        if parameters['testN']>0:
            alg_params = {
                'df': outputs['df'],
                'crs': outputs['crs'],
                'OUT': parameters['out']
            }
            SZ_utils.save(alg_params)

        feedback.setCurrentStep(3)
        if feedback.isCanceled():
            return {}
        
        if ML[parameters['family']]=='SVM_classifier' or ML[parameters['family']]=='DT_classifier' or ML[parameters['family']]=='RF_classifier':
            alg_params = {
                'test_ind': outputs['test_ind'],
                'df': outputs['df'],
                'OUT':parameters['folder']
            }
            SZ_utils.stamp_cv(alg_params)

        else:
            alg_params = {
                'test_ind': outputs['test_ind'],
                'df': outputs['df'],
                'OUT':parameters['folder']
            }
            outputs['error_train']=SZ_utils.stamp_qq(alg_params)

            alg_params = {
                'df': outputs['df'],                
                'OUT':parameters['folder']
            }
            outputs['error_train']=SZ_utils.stamp_qq_fit(alg_params)

        feedback.setCurrentStep(4)
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

        feedback.setCurrentStep(5)
        if feedback.isCanceled():
            return {}

        return results



