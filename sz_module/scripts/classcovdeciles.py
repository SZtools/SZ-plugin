# -*- coding: utf-8 -*-

"""
/***************************************************************************
    02 FR Fitting/CrossValid
        begin                : 2021-11
        copyright            : (C) 2021 by Giacomo Titti,
                               Padova, November 2021
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
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
__date__ = '2021-07-01'
__copyright__ = '(C) 2021 by Giacomo Titti'

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = '$Format:%H$'

from qgis.PyQt.QtCore import QCoreApplication,QVariant
from qgis.core import (QgsProcessing,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingMultiStepFeedback,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterVectorLayer,
                       QgsVectorLayer,
                       QgsProject,
                       QgsField,
                       QgsProcessingParameterField,
                       QgsProcessingParameterString,
                       QgsProcessingParameterField,
)
import matplotlib.pyplot as plt

from qgis import processing
from osgeo import gdal,ogr,osr
import numpy as np
from qgis import *
# ##############################
import matplotlib.pyplot as plt
from processing.algs.gdal.GdalUtils import GdalUtils

class classcovdecAlgorithm(QgsProcessingAlgorithm):
    # INPUT = 'INPUT'
    # STRING = 'STRING'
    # FILE = 'FILE'
    # STRING3 = 'STRING3'
    # OUTPUT = 'OUTPUT'
    # NUMBER = 'NUMBER'

    # def tr(self, string):
    #     return QCoreApplication.translate('Processing', string)

    # def createInstance(self):
    #     return classcovdecAlgorithm()

    # def name(self):
    #     return 'classy filed in quantiles'

    # def displayName(self):
    #     return self.tr('07 Classify field in quantiles')

    # def group(self):
    #     return self.tr('01 Data preparation')

    # def groupId(self):
    #     return '01 Data preparation'

    # def shortHelpString(self):
    #     return self.tr("Apply classification to field in quantiles")

    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('covariates'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING, 'field', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterString(self.STRING3, 'new field name', defaultValue=None))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER, self.tr('number of percentile (4=quartiles, 10=deciles)'), type=QgsProcessingParameterNumber.Integer, defaultValue = 10,  minValue=1))

    def process(self, parameters, context, model_feedback):

        feedback = QgsProcessingMultiStepFeedback(1, model_feedback)
        results = {}
        outputs = {}


        source = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        parameters['covariates']=source.source()
        if parameters['covariates'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        parameters['field'] = self.parameterAsString(parameters, self.STRING, context)
        if parameters['field'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING))

        parameters['nome'] = self.parameterAsString(parameters, self.STRING3, context)
        if parameters['nome'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING3))

        parameters['num'] = self.parameterAsInt(parameters, self.NUMBER, context)
        if parameters['num'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER))

        alg_params = {
        'INPUT_VECTOR_LAYER': parameters['covariates'],
        'field': parameters['field'],
        'nome' : parameters['nome'],
        'num' : parameters['num']
            }

        outputs['crs']=self.classify(alg_params)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}
        return results

    def classify(self,parameters):###############classify causes according to txt classes
        layer = QgsVectorLayer(parameters['INPUT_VECTOR_LAYER'], '', 'ogr')
        crs=layer.crs()
        features = layer.getFeatures()

        field=np.array([])
        for feature in features:
            field=np.append(field,feature.attribute(parameters['field']))
        deciles=np.percentile(field, np.arange(100/parameters['num'], 100, 100/parameters['num'])) # deciles
        deciles=np.hstack((np.min(field)-0.1,deciles,np.max(field)+0.1))
        print(deciles,'classes')
        Min={}
        Max={}
        clas={}
        countr=1
        for cond in range(len(deciles)-1):
            Min[countr]=deciles[cond].astype(np.float32)
            Max[countr]=deciles[cond+1].astype(np.float32)
            clas[countr]=cond+1#.astype(int)
            countr+=1
        key_max=None
        key_min=None
        key_max = max(Max.keys(), key=(lambda k: Max[k]))
        key_min = min(Min.keys(), key=(lambda k: Min[k]))

        layer = QgsVectorLayer(parameters['INPUT_VECTOR_LAYER'], '', 'ogr')
        crs=layer.crs()
        layer.dataProvider().addAttributes([QgsField(parameters['nome'], QVariant.Int)])
        layer.updateFields()
        features = layer.getFeatures()
        layer.startEditing()
        count=0
        feat=[]

        for feature in features:
            ff=feature.attribute(parameters['field'])
            for i in range(1,countr):
                if ff>=Min[i] and ff<Max[i]:
                    feature[parameters['nome']]=int(clas[i])
                    layer.updateFeature(feature)
        layer.commitChanges()
        QgsProject.instance().reloadAllLayers()

        return(crs)