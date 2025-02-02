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
                       QgsMessageLog,
                       Qgis,
                       QgsProcessingMultiStepFeedback,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterField,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterField,
)
from sklearn.metrics import roc_curve, f1_score, cohen_kappa_score, roc_auc_score
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
from qgis import *
import matplotlib.pyplot as plt
import tempfile
import os
from sz_module.scripts.utils import SZ_utils

class rocGenerator(QgsProcessingAlgorithm):

    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Input layer'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING, 'Index', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING2, 'Field of dependent variable (0 for absence, > 0 for presence)', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT3, 'Folder destination', defaultValue=None, createByDefault = True))

    def process(self, parameters, context, model_feedback):
        self.f=tempfile.gettempdir()
        feedback = QgsProcessingMultiStepFeedback(1, model_feedback)
        results = {}
        outputs = {}

        self.f=tempfile.gettempdir()

        source = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        parameters['covariates']=source.source()
        if parameters['covariates'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))
        parameters['field1'] = self.parameterAsString(parameters, self.STRING, context)
        if parameters['field1'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING))
        parameters['fieldlsd'] = self.parameterAsString(parameters, self.STRING2, context)
        if parameters['fieldlsd'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING2))
        parameters['fold'] = self.parameterAsString(parameters, self.OUTPUT3, context)
        if parameters['fold'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT3))

        SZ_utils.make_directory({'path':parameters['fold']})

        alg_params = {
            'INPUT_VECTOR_LAYER': parameters['covariates'],
            'nomi': parameters['field1'],
            'lsd' : parameters['fieldlsd']
        }
        outputs['gdp'],outputs['crs']=SZ_utils.load_cv(self.f,alg_params)

        self.list_of_values=outputs['gdp']['SI']
        QgsMessageLog.logMessage(str(len(self.list_of_values)), 'MyPlugin', level=Qgis.Info)

        alg_params = {
            'df': outputs['gdp'],
            'OUT': parameters['fold']
        }
        Functions.roc(alg_params)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}
        return results

class Functions():
    def roc(parameters):
        df=parameters['df']
        y_true=df['y']
        scores=df['SI']
        ################################figure
        fpr1, tpr1, tresh1 = roc_curve(y_true,scores)
        norm=(scores-scores.min())/(scores.max()-scores.min())
        r=roc_auc_score(y_true, scores)
        print(r,'!!!!!!')
        idx = np.argmax(tpr1 - fpr1)  # x YOUDEN INDEX
        suscept01 = copy(scores)
        suscept01[scores > tresh1[idx]] = 1
        suscept01[scores <= tresh1[idx]] = 0
        f1_tot = f1_score(y_true, suscept01)
        ck_tot = cohen_kappa_score(y_true, suscept01)
        fig=plt.figure()
        lw = 2
        plt.plot(fpr1, tpr1, color='green',lw=lw, label= 'AUC = %0.2f, F1 = %0.2f, K = %0.2f' %(r, f1_tot,ck_tot))
        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        try:
            fig.savefig(parameters['OUT']+'/roc.pdf')
        except:
            os.mkdir(parameters['OUT'])
            fig.savefig(parameters['OUT']+'/roc.pdf')