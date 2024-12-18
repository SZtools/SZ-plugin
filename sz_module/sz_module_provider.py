
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

from qgis.core import QgsProcessingAlgorithm
from qgis.core import QgsProcessingProvider,QgsProcessingAlgorithm
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from sz_module.images.cqp_resources_rc import qInitResources
qInitResources()
import os
from .scripts.selfroc import rocGenerator
from .scripts.lsdanalysis import statistic
from .scripts.cleaning import cleankernelAlgorithm
from .scripts.graphs_lsdstats_kernel import statistickernel
from .scripts.randomsampler3 import samplerAlgorithm
from .scripts.stat31 import rasterstatkernelAlgorithm
from .scripts.classvector import classvAlgorithm
from .scripts.classvectorw import classvAlgorithmW
from .scripts.tptn import FPAlgorithm
from .scripts.classcovtxt import classcovtxtAlgorithm
from .scripts.classcovdeciles import classcovdecAlgorithm
from .scripts.corrplot import CorrAlgorithm
from .scripts.sz_train_cv_ML import CoreAlgorithm_cv
from .scripts.sz_train_cv_GAM import CoreAlgorithmGAM_cv
from .scripts.sz_train_cv_NN import CoreAlgorithmNN_cv
from .scripts.sz_trans_GAM import CoreAlgorithmGAM_trans
from .scripts.sz_trans_ML import CoreAlgorithmML_trans
from .scripts.sz_trans_NN import CoreAlgorithmNN_trans
from .scripts.algorithms import Algorithms
from sz_module.scripts.segmentation_aspect import segmentationAspectAlgorithm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier,MLPRegressor
from pygam import LogisticGAM,LinearGAM
from .utils import log

class classeProvider(QgsProcessingProvider):

    def __init__(self):
        QgsProcessingProvider.__init__(self)

    def unload(self):
        pass

    def loadAlgorithms(self):

        self.active={
            'classcovtxt':False,
            'classcovdeciles':False,
            'statistic':True,
            'rocGenerator':True,
            'cleankernel':False,
            'statistickernel':False,
            'sampler':False,
            'rasterstatkernel':False,
            'Corr':True,
            'classv':True,
            'classvW':True,
            'FP':True,
            'ML_cv':True,
            'GAM_cv':True,
            'GAM_trans':True,
            'ML_trans':True,
            'SegAsp':False,
            'NN_trans':True,
            'NN_cv':True,
        }

        dict_of_scripts={
            'alg': 'ML_cv',
            'function': CoreAlgorithm_cv,
            'name':'Fit-CV_SVCcv',
            'displayName':'01 Machine Learning tools',
            'group':'02 Modelling',
            'groupId':'02 Modelling',
            'shortHelpString':"This function uses Machine Learning algorithms to model. It allows to cross-validate data by many methods. If you want just do fitting put k-fold equal to one",
        }
        self.addAlgorithm(Instance(dict_of_scripts)) if self.active[dict_of_scripts['alg']] else print(dict_of_scripts['alg']+' is inactive')

        dict_of_scripts={
            'alg': 'GAM_cv',
            'function': CoreAlgorithmGAM_cv,
            'name':'Fit-CV_GAMcv',
            'displayName':'02 Statistical tools',
            'group':'02 Modelling',
            'groupId':'02 Modelling',
            'shortHelpString':"This function uses Generalized Additive Model to model. It allows to cross-validate data by many methods. If you want just do fitting put k-fold equal to one.",
        }
        self.addAlgorithm(Instance(dict_of_scripts)) if self.active[dict_of_scripts['alg']] else print(dict_of_scripts['alg']+' is inactive')

        dict_of_scripts={
            'alg': 'GAM_trans',
            'function': CoreAlgorithmGAM_trans,
            'name':'Transfer_GAM',
            'displayName':'02 Predict Generalized Additive Model',
             'group':'03 Transfer learning',
            'groupId':'03 Transfer learning',
            'shortHelpString':"This function uses Generalized Additive Model to transfer learning.",
        }
        self.addAlgorithm(Instance(dict_of_scripts)) if self.active[dict_of_scripts['alg']] else print(dict_of_scripts['alg']+' is inactive')

        dict_of_scripts={
            'alg': 'ML_trans',
            'function': CoreAlgorithmML_trans,
            'name':'Transfer_ML',
            'displayName':'01 Predict Machine Learning',
             'group':'03 Transfer learning',
            'groupId':'03 Transfer learning',
            'shortHelpString':"This function uses Machine Learning algorithms to transfer learning.",
        }
        self.addAlgorithm(Instance(dict_of_scripts)) if self.active[dict_of_scripts['alg']] else print(dict_of_scripts['alg']+' is inactive')

        dict_of_scripts={
            'alg': 'NN_cv',
            'function': CoreAlgorithmNN_cv,
            'name':'Fit-CV_NNcv',
            'displayName':'03 Neural Network tools',
            'group':'02 Modelling',
            'groupId':'02 Modelling',
            'shortHelpString':"This function uses Neural Networks algorithms to model. It allows to cross-validate data by many methods. If you want just do fitting put k-fold equal to one.",
        }
        self.addAlgorithm(Instance(dict_of_scripts)) if self.active[dict_of_scripts['alg']] else print(dict_of_scripts['alg']+' is inactive')

        dict_of_scripts={
            'alg': 'NN_trans',
            'function': CoreAlgorithmNN_trans,
            'name':'Transfer_NN',
            'displayName':'03 Predict Neural Network',
            'group':'03 Transfer learning',
            'groupId':'03 Transfer learning',
            'shortHelpString':"This function uses Neural Networks algorithm to transfer learning.",
        }
        self.addAlgorithm(Instance(dict_of_scripts)) if self.active[dict_of_scripts['alg']] else print(dict_of_scripts['alg']+' is inactive')

        dict_of_scripts={
            'alg': 'classcovtxt',
            'function': classcovtxtAlgorithm,
            'name':'classy filed by file.txt',
            'displayName':'06 Classify field by file.txt',
            'group':'01 Data preparation',
            'groupId':'01 Data preparation',
            'shortHelpString':"Apply classification to field from file.txt i.e value_1 value_2 class_1",
        }
        self.addAlgorithm(Instance(dict_of_scripts)) if self.active[dict_of_scripts['alg']] else print(dict_of_scripts['alg']+' is inactive')

        dict_of_scripts={
            'alg': 'classcovdeciles',
            'function': classcovdecAlgorithm,
            'name':'classy filed in quantiles',
            'displayName':'07 Classify field in quantiles',
            'group':'01 Data preparation',
            'groupId':'01 Data preparation',
            'shortHelpString':"Apply classification to field in quantiles",
        }
        self.addAlgorithm(Instance(dict_of_scripts)) if self.active[dict_of_scripts['alg']] else print(dict_of_scripts['alg']+' is inactive')
    
        dict_of_scripts={
            'alg': 'statistic',
            'function': statistic,
            'name':'attributes analysis',
            'displayName':'01 Attribute Table Statistics',
            'group':'01 Data preparation',
            'groupId':'01 Data preparation',
            'shortHelpString':"Distribution plot of the fields value density.",
        }
        self.addAlgorithm(Instance(dict_of_scripts)) if self.active[dict_of_scripts['alg']] else print(dict_of_scripts['alg']+' is inactive')

         
        dict_of_scripts={
            'alg': 'rocGenerator',
            'function': rocGenerator,
            'name':'ROC',
            'displayName':'04 ROC',
            'group':'04 Result interpretation',
            'groupId':'04 Result interpretation',
            'shortHelpString':"ROC curve creator. It calculates AUC, F1 score and Choen's Kappa coefficient.",
        }
        self.addAlgorithm(Instance(dict_of_scripts)) if self.active[dict_of_scripts['alg']] else print(dict_of_scripts['alg']+' is inactive')

        dict_of_scripts={
            'alg': 'cleankernel',
            'function': cleankernelAlgorithm,
            'name':'clean points',
            'displayName':'01 Clean Points By Raster Kernel Value',
            'group':'01 Data preparation',
            'groupId':'01 Data preparation',
            'shortHelpString':"It selects and remove features from point vector by a kernel raster condition",
        }
        self.addAlgorithm(Instance(dict_of_scripts)) if self.active[dict_of_scripts['alg']] else print(dict_of_scripts['alg']+' is inactive')

        dict_of_scripts={
            'alg': 'statistickernel',
            'function': statistickernel,
            'name':'points kernel graphs',
            'displayName':'04 Points kernel graphs',
            'group':'01 Data preparation',
            'groupId':'01 Data preparation',
            'shortHelpString':"It creates graphs of '03 Points Kernel Statistics' output",
        }
        self.addAlgorithm(Instance(dict_of_scripts)) if self.active[dict_of_scripts['alg']] else print(dict_of_scripts['alg']+' is inactive')

        dict_of_scripts={
            'alg': 'sampler',
            'function': samplerAlgorithm,
            'name':'points sampler',
            'displayName':'05 Points Sampler',
            'group':'01 Data preparation',
            'groupId':'01 Data preparation',
            'shortHelpString':"Sample randomly training and validating datasets with the contraint to have only training or validating points per pixel",
        }
        self.addAlgorithm(Instance(dict_of_scripts)) if self.active[dict_of_scripts['alg']] else print(dict_of_scripts['alg']+' is inactive')

        dict_of_scripts={
            'alg': 'rasterstatkernel',
            'function': rasterstatkernelAlgorithm,
            'name':'kernel stat',
            'displayName':'03 Points Kernel Statistics',
            'group':'01 Data preparation',
            'groupId':'01 Data preparation',
            'shortHelpString':"It calculates kernel statistic from raster around points: real, max, min, std, sum, average, range",
        }
        self.addAlgorithm(Instance(dict_of_scripts)) if self.active[dict_of_scripts['alg']] else print(dict_of_scripts['alg']+' is inactive')

        dict_of_scripts={
            'alg': 'Corr',
            'function': CorrAlgorithm,
            'name':'Correlation plot',
            'displayName':'02 Correlation plot',
            'group':'01 Data preparation',
            'groupId':'01 Data preparation',
            'shortHelpString':"This function calculate the correlation plot between continuous variables",
        }
        self.addAlgorithm(Instance(dict_of_scripts)) if self.active[dict_of_scripts['alg']] else print(dict_of_scripts['alg']+' is inactive')

        dict_of_scripts={
            'alg': 'classv',
            'function': classvAlgorithm,
            'name':'classy vector by ROC',
            'displayName':'01 Classify vector by ROC',
            'group':'04 Result interpretation',
            'groupId':'04 Result interpretation',
            'shortHelpString':"This function classifies a index (SI) maximizing the AUC of the relative ROC curve.",
        }
        self.addAlgorithm(Instance(dict_of_scripts)) if self.active[dict_of_scripts['alg']] else print(dict_of_scripts['alg']+' is inactive')

        dict_of_scripts={
            'alg': 'classvW',
            'function': classvAlgorithmW,
            'name':'classy vector by wROC',
            'displayName':'02 Classify vector by weighted ROC',
            'group':'04 Result interpretation',
            'groupId':'04 Result interpretation',
            'shortHelpString':"This function classifies a index (SI) maximizing the AUC of the relative weighted ROC curve",
        }
        self.addAlgorithm(Instance(dict_of_scripts)) if self.active[dict_of_scripts['alg']] else print(dict_of_scripts['alg']+' is inactive')

        dict_of_scripts={
            'alg': 'FP',
            'function': FPAlgorithm,
            'name':'Conf matrix',
            'displayName':'03 Confusion Matrix',
            'group':'04 Result interpretation',
            'groupId':'04 Result interpretation',
            'shortHelpString':"This function labels each feature as True Positive (0), True Negative (1), False Positive (2), False Negative (3) based on presence/absence.",
        }
        self.addAlgorithm(Instance(dict_of_scripts)) if self.active[dict_of_scripts['alg']] else print(dict_of_scripts['alg']+' is inactive')

        dict_of_scripts={
            'alg': 'SegAsp',
            'function': segmentationAspectAlgorithm,
            'name':'Segmentation aspect',
            'displayName':'09 Segmentation aspect',
            'group':'01 Data preparation',
            'groupId':'01 Data preparation',
            'shortHelpString':"Segmentation aspect metric proposed for SU by Alvioli et al (2016). For more details, please refer to the paper.",
        }
        self.addAlgorithm(Instance(dict_of_scripts)) if self.active[dict_of_scripts['alg']] else print(dict_of_scripts['alg']+' is inactive')

        # add additional algorithms here
        # self.addAlgorithm(MyOtherAlgorithm())

    def id(self):
        return 'SZ'

    def name(self):
        return self.tr('SZ')

    def icon(self):
        return QIcon(':/icon')

    def longName(self):
        return self.name()

class Instance(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    INPUT1 = 'INPUT1'
    STRING = 'STRING'
    STRING1 = 'STRING1'
    STRING2 = 'STRING2'
    STRING3 = 'STRING3'
    STRING4 = 'STRING4'
    STRING5 = 'STRING5'
    STRING6 = 'STRING6'
    STRING7 = 'STRING7'
    STRING8 = 'STRING8'
    STRING9 = 'STRING9'
    NUMBER = 'NUMBER'
    NUMBER1 = 'NUMBER1'
    NUMBER2 = 'NUMBER2'
    OUTPUT = 'OUTPUT'
    OUTPUT1 = 'OUTPUT1'
    OUTPUT2 = 'OUTPUT2'
    OUTPUT3 = 'OUTPUT3'
    FILE = 'FILE'
    EXTENT = 'EXTENT'
    FOLDER = 'FOLDER'
    MASK = 'MASK'
    
    def __init__(self, dict_of_scripts):
        super().__init__()
        self.dict_of_scripts = dict_of_scripts

        self.algorithms={
            'ML_cv':Algorithms.alg_MLrun,
            'ML_trans':Algorithms.alg_MLrun,
            'GAM_cv':Algorithms.alg_GAMrun,
            'GAM_trans':Algorithms.alg_GAMrun,
            'NN_trans':Algorithms.alg_NNrun,
            'NN_cv':Algorithms.alg_NNrun,
        }

        self.classifier={
            'ML_cv':{
                'SVC':SVC(kernel = 'linear', random_state = 0,probability=True),
                'RF':RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0),
                'DT':DecisionTreeClassifier(criterion = 'entropy', random_state = 0),
            },
            'ML_trans':{
                'SVC':SVC(kernel = 'linear', random_state = 0,probability=True),
                'RF':RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0),
                'DT':DecisionTreeClassifier(criterion = 'entropy', random_state = 0),
            },
            'GAM_cv':{'binomial':LogisticGAM,'gaussian':LinearGAM},
            'GAM_trans':{'binomial':LogisticGAM,'gaussian':LinearGAM},
            'NN_trans':{
                'MLP_classifier':MLPClassifier(hidden_layer_sizes=(16, 32, 64, 128, 64, 32, 16, 8), random_state=42, max_iter=2000, validation_fraction=0.1, early_stopping=True),
                'MLP_regressor':MLPRegressor(hidden_layer_sizes=(16, 32, 64, 128, 32, 16, 8), random_state=42, max_iter=2000, validation_fraction=0.1, early_stopping=True),
            },
            'NN_cv':{
                'MLP_classifier':MLPClassifier(hidden_layer_sizes=(16, 32, 64, 128, 64, 32, 16, 8), random_state=42, max_iter=2000, validation_fraction=0.1, early_stopping=True),
                'MLP_regressor':MLPRegressor(hidden_layer_sizes=(16, 32, 64, 128, 32, 16, 8), random_state=42, max_iter=2000, validation_fraction=0.1, early_stopping=True),
            },
        }

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return Instance(self.dict_of_scripts)

    def name(self):
        return self.dict_of_scripts['name']

    def displayName(self):
        return self.tr(self.dict_of_scripts['displayName'])

    def group(self):
        return self.tr(self.dict_of_scripts['group'])

    def groupId(self):
        return self.dict_of_scripts['groupId']

    def shortHelpString(self):
        return self.tr(self.dict_of_scripts['shortHelpString'])

    def initAlgorithm(self, config=None):
        self.dict_of_scripts['function'].init(self, config=None)

    def processAlgorithm(self, parameters, context, feedback):
        result={}

        if self.dict_of_scripts['alg'] in self.algorithms:
            if os.environ.get('DEBUG')=='False':
                try:
                    result=self.dict_of_scripts['function'].process(self,parameters, context, feedback, algorithm=self.algorithms[self.dict_of_scripts['alg']], classifier=self.classifier[self.dict_of_scripts['alg']])
                except Exception as e:
                    log(f"An error occurred: {e}")
            else:
                result=self.dict_of_scripts['function'].process(self,parameters, context, feedback, algorithm=self.algorithms[self.dict_of_scripts['alg']], classifier=self.classifier[self.dict_of_scripts['alg']])
        else:
            if os.environ.get('DEBUG')=='False':
                try:
                    result=self.dict_of_scripts['function'].process(self,parameters, context, feedback)
                except Exception as e:
                    log(f"An error occurred: {e}")
            else:
                result=self.dict_of_scripts['function'].process(self,parameters, context, feedback)
        
        return result