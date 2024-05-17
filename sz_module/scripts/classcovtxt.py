# -*- coding: utf-8 -*-

"""
/***************************************************************************
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
                       QgsProcessingParameterVectorLayer,
                       QgsVectorLayer,
                       QgsProject,
                       QgsField,
                       QgsProcessingParameterField,
                       QgsProcessingParameterString,
                       QgsProcessingParameterField,
                       QgsProcessingParameterFile)
from qgis import processing
from osgeo import gdal,ogr,osr
import numpy as np
from qgis import *
# ##############################
import csv
from processing.algs.gdal.GdalUtils import GdalUtils

class classcovtxtAlgorithm(QgsProcessingAlgorithm):
    # INPUT = 'INPUT'
    # STRING = 'STRING'
    # FILE = 'FILE'
    # STRING3 = 'STRING3'
    # OUTPUT = 'OUTPUT'

    # def tr(self, string):
    #     return QCoreApplication.translate('Processing', string)

    # def createInstance(self):
    #     return classcovtxtAlgorithm()

    # def name(self):
    #     return 'classy filed by file.txt'

    # def displayName(self):
    #     return self.tr('06 Classify field by file.txt')

    # def group(self):
    #     return self.tr('01 Data preparation')

    # def groupId(self):
    #     return '01 Data preparation'

    # def shortHelpString(self):
    #     return self.tr("Apply classification to field from file.txt i.e value_1 value_2 class_1")

    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('covariates'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterFile(self.FILE, 'Txt classes', QgsProcessingParameterFile.File, '', defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING, 'field', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterString(self.STRING3, 'new field name', defaultValue=None))

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

        parameters['txt'] = self.parameterAsFile(parameters, self.FILE, context)#.source()
        if parameters['txt'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.FILE))
        print(parameters['txt'])

        parameters['nome'] = self.parameterAsString(parameters, self.STRING3, context)
        if parameters['nome'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING3))

        alg_params = {
        'INPUT_VECTOR_LAYER': parameters['covariates'],
        'field': parameters['field'],
        'txt' : parameters['txt'],
        'nome' : parameters['nome']
            }

        outputs['crs']=self.classify(alg_params)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}
        return results

    def classify(self,parameters):###############classify causes according to txt classes
        Min={}
        Max={}
        clas={}
        with open(parameters['txt'], 'r') as f:
            c = csv.reader(f,delimiter=' ')
            countr=1
            for cond in c:
                b=np.array([])
                b=np.asarray(cond)
                Min[countr]=b[0].astype(np.float32)
                Max[countr]=b[1].astype(np.float32)
                clas[countr]=b[2]#.astype(int)
                countr+=1
        key_max=None
        key_min=None
        key_max = max(Max.keys(), key=(lambda k: Max[k]))
        key_min = min(Min.keys(), key=(lambda k: Min[k]))

        layer = QgsVectorLayer(parameters['INPUT_VECTOR_LAYER'], '', 'ogr')
        crs=layer.crs()

        layer.dataProvider().addAttributes([QgsField(parameters['nome'], QVariant.Int)])
        layer.updateFields()
        layer.startEditing()
        features = layer.getFeatures()
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

