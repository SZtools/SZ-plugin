# -*- coding: utf-8 -*-

"""
/***************************************************************************
 classe
                                 A QGIS plugin
 susceptibility
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2021-07-01
        copyright            : (C) 2021 by Giacomo Titti
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

__author__ = 'Giacomo Titti'
__date__ = '2021-07-01'
__copyright__ = '(C) 2021 by Giacomo Titti'

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = '$Format:%H$'


#####################
from qgis.core import (QgsProcessingException,
                       QgsProcessingAlgorithm,
                       )

from qgis.core import QgsProcessingProvider,QgsProcessingAlgorithm
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from sz_module.images.cqp_resources_rc import qInitResources
qInitResources()  # necessary to be able to access your images

from .scripts.roc import rocAlgorithm
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
from sz_module.scripts.corrplot import CorrAlgorithm
from sz_module.scripts.sz_train_cv import CoreAlgorithm_cv
from sz_module.scripts.sz_train_cv_GAM import CoreAlgorithmGAM_cv
from sz_module.scripts.sz_trans_GAM import CoreAlgorithmGAM_trans
from sz_module.scripts.algorithms import Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from pygam import LogisticGAM

class classeProvider(QgsProcessingProvider):

    def __init__(self):
        """
        Default constructor.
        """
        QgsProcessingProvider.__init__(self)

    def unload(self):
        """
        Unloads the provider. Any tear-down steps required by the provider
        should be implemented here.
        """
        pass

    def loadAlgorithms(self):
        """
        Loads all algorithms belonging to this provider.
        """

        # dict_of_scripts={
        #     'alg': 'woe_simple',
        #     'function': CoreAlgorithm,
        #     'name':'Fit-CV_WOE',
        #     'displayName':'01 WoE',
        #     'group':'02 SI',
        #     'groupId':'02 SI',
        #     'shortHelpString':"This function apply Weight of Evidence to calculate susceptibility. It allows to cross-validate the analysis selecting the sample percentage test/training. If you want just do fitting put the test percentage equal to zero",
        # }
        # self.addAlgorithm(Instance(dict_of_scripts))

        # dict_of_scripts={
        #     'alg': 'woe_cv',
        #     'function': CoreAlgorithm_cv,
        #     'name':'Fit-CV_WOEcv',
        #     'displayName':'01 WoE',
        #     'group':'03 SI k-fold',
        #     'groupId':'03 SI k-fold',
        #     'shortHelpString':"This function apply Weight of Evidence to calculate susceptibility. It allows to cross-validate the analysis by k-fold cross-validation method. If you want just do fitting put k-fold equal to one",
        # }
        # self.addAlgorithm(Instance(dict_of_scripts))

        # dict_of_scripts={
        #     'alg': 'SVC_simple',
        #     'function': CoreAlgorithm,
        #     'name':'Fit-CV_SVC',
        #     'displayName':'05 SVM',
        #     'group':'02 SI',
        #     'groupId':'02 SI',
        #     'shortHelpString':"This function apply Support Vector Machine to calculate susceptibility. It allows to cross-validate the analysis selecting the sample percentage test/training. If you want just do fitting put the test percentage equal to zero",
        # }
        # self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'SVC_cv',
            'function': CoreAlgorithm_cv,
            'name':'Fit-CV_SVCcv',
            'displayName':'05 SVM',
            'group':'03 SI k-fold',
            'groupId':'03 SI k-fold',
            'shortHelpString':"This function apply Support Vector Machine to calculate susceptibility. It allows to cross-validate the analysis by k-fold cross-validation method. If you want just do fitting put k-fold equal to one",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        # dict_of_scripts={
        #     'alg': 'RF_simple',
        #     'function': CoreAlgorithm,
        #     'name':'Fit-CV_RF',
        #     'displayName':'04 RF',
        #     'group':'02 SI',
        #     'groupId':'02 SI',
        #     'shortHelpString':"This function apply Random Forest to calculate susceptibility. It allows to cross-validate the analysis selecting the sample percentage test/training. If you want just do fitting put the test percentage equal to zero",
        # }
        # self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'RF_cv',
            'function': CoreAlgorithm_cv,
            'name':'Fit-CV_RFcv',
            'displayName':'04 RF',
            'group':'03 SI k-fold',
            'groupId':'03 SI k-fold',
            'shortHelpString':"This function apply Random Forest to calculate susceptibility. It allows to cross-validate the analysis by k-fold cross-validation method. If you want just do fitting put k-fold equal to one",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        # dict_of_scripts={
        #     'alg': 'LR_simple',
        #     'function': CoreAlgorithm,
        #     'name':'Fit-CV_LR',
        #     'displayName':'03 LR',
        #     'group':'02 SI',
        #     'groupId':'02 SI',
        #     'shortHelpString':"This function apply Logistic Regression to calculate susceptibility. It allows to cross-validate the analysis selecting the sample percentage test/training. If you want just do fitting put the test percentage equal to zero",
        # }
        # self.addAlgorithm(Instance(dict_of_scripts))

        # dict_of_scripts={
        #     'alg': 'LR_cv',
        #     'function': CoreAlgorithm_cv,
        #     'name':'Fit-CV_LRcv',
        #     'displayName':'03 LR',
        #     'group':'03 SI k-fold',
        #     'groupId':'03 SI k-fold',
        #     'shortHelpString':"This function apply Logistic Regression to calculate susceptibility. It allows to cross-validate the analysis by k-fold cross-validation method. If you want just do fitting put k-fold equal to one",
        # }
        # self.addAlgorithm(Instance(dict_of_scripts))

        # dict_of_scripts={
        #     'alg': 'fr_simple',
        #     'function': CoreAlgorithm,
        #     'name':'Fit-CV_FR',
        #     'displayName':'02 FR',
        #     'group':'02 SI',
        #     'groupId':'02 SI',
        #     'shortHelpString':"This function apply Frequency Ratio to calculate susceptibility. It allows to cross-validate the analysis selecting the sample percentage test/training. If you want just do fitting put the test percentage equal to zero",
        # }
        # self.addAlgorithm(Instance(dict_of_scripts))

        # dict_of_scripts={
        #     'alg': 'fr_cv',
        #     'function': CoreAlgorithm_cv,
        #     'name':'Fit-CV_FRcv',
        #     'displayName':'02 FR',
        #     'group':'03 SI k-fold',
        #     'groupId':'03 SI k-fold',
        #     'shortHelpString':"This function apply Frequency Ratio to calculate susceptibility. It allows to cross-validate the analysis by k-fold cross-validation method. If you want just do fitting put k-fold equal to one",
        # }
        # self.addAlgorithm(Instance(dict_of_scripts))

        # dict_of_scripts={
        #     'alg': 'DT_simple',
        #     'function': CoreAlgorithm,
        #     'name':'Fit-CV_DT',
        #     'displayName':'06 DT',
        #     'group':'02 SI',
        #     'groupId':'02 SI',
        #     'shortHelpString':"This function apply Decision Tree to calculate susceptibility. It allows to cross-validate the analysis selecting the sample percentage test/training. If you want just do fitting put the test percentage equal to zero",
        # }
        # self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'DT_cv',
            'function': CoreAlgorithm_cv,
            'name':'Fit-CV_DTcv',
            'displayName':'06 DT',
            'group':'03 SI k-fold',
            'groupId':'03 SI k-fold',
            'shortHelpString':"This function apply Decision Tree to calculate susceptibility. It allows to cross-validate the analysis by k-fold cross-validation method. If you want just do fitting put k-fold equal to one",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        # dict_of_scripts={
        #     'alg': 'GAM_simple',
        #     'function': CoreAlgorithmGAM,
        #     'name':'Fit-CV_GAM',
        #     'displayName':'07 GAM',
        #     'group':'02 SI',
        #     'groupId':'02 SI',
        #     'shortHelpString':"This function apply Generalized Additive Model to calculate susceptibility. It allows to cross-validate the analysis selecting the sample percentage test/training. If you want just do fitting put the test percentage equal to zero",
        # }
        # self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'GAM_cv',
            'function': CoreAlgorithmGAM_cv,
            'name':'Fit-CV_GAMcv',
            'displayName':'07 GAM',
            'group':'03 SI k-fold',
            'groupId':'03 SI k-fold',
            'shortHelpString':"This function apply Generalized Additive Model to calculate susceptibility. It allows to cross-validate the analysis by k-fold cross-validation method. If you want just do fitting put k-fold equal to one",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'GAM_trans',
            'function': CoreAlgorithmGAM_trans,
            'name':'Transfer_GAM',
            'displayName':'01 GAM',
            'group':'04 SI Transfer',
            'groupId':'04 SI Transfer',
            'shortHelpString':"This function apply Generalized Additive Model to transfer susceptibility",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        ##############
        dict_of_scripts={
            'alg': 'classcovtxt',
            'function': classcovtxtAlgorithm,
            'name':'classy filed by file.txt',
            'displayName':'06 Classify field by file.txt',
            'group':'01 Data preparation',
            'groupId':'01 Data preparation',
            'shortHelpString':"Apply classification to field from file.txt i.e value_1 value_2 class_1",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'classcovdec',
            'function': classcovdecAlgorithm,
            'name':'classy filed in quantiles',
            'displayName':'07 Classify field in quantiles',
            'group':'01 Data preparation',
            'groupId':'01 Data preparation',
            'shortHelpString':"Apply classification to field in quantiles",
        }
        self.addAlgorithm(Instance(dict_of_scripts))
    

        ##self.addAlgorithm(polytogridAlgorithm())
        #self.addAlgorithm(pointtogridAlgorithm())
        dict_of_scripts={
            'alg': 'statistic',
            'function': statistic,
            'name':'attributes analysis',
            'displayName':'02 Attribute Table Statistics',
            'group':'01 Data preparation',
            'groupId':'01 Data preparation',
            'shortHelpString':"analysis of the points density distribution by attribute fields",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        #self.addAlgorithm(classAlgorithm())
        #self.addAlgorithm(rocAlgorithm())
        #self.addAlgorithm(matrixAlgorithm())
        dict_of_scripts={
            'alg': 'rocGenerator',
            'function': rocGenerator,
            'name':'ROC',
            'displayName':'04 ROC',
            'group':'05 Classify SI',
            'groupId':'05 Classify SI',
            'shortHelpString':"ROC curve creator",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'cleankernel',
            'function': cleankernelAlgorithm,
            'name':'clean points',
            'displayName':'01 Clean Points By Raster Kernel Value',
            'group':'01 Data preparation',
            'groupId':'01 Data preparation',
            'shortHelpString':"It selects and remove features from point vector by a kernel raster condition",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'statistickernel',
            'function': statistickernel,
            'name':'points kernel graphs',
            'displayName':'04 Points kernel graphs',
            'group':'01 Data preparation',
            'groupId':'01 Data preparation',
            'shortHelpString':"It creates graphs of '03 Points Kernel Statistics' output",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'sampler',
            'function': samplerAlgorithm,
            'name':'points sampler',
            'displayName':'05 Points Sampler',
            'group':'01 Data preparation',
            'groupId':'01 Data preparation',
            'shortHelpString':"Sample randomly training and validating datasets with the contraint to have only training or validating points per pixel",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'rasterstatkernel',
            'function': rasterstatkernelAlgorithm,
            'name':'kernel stat',
            'displayName':'03 Points Kernel Statistics',
            'group':'01 Data preparation',
            'groupId':'01 Data preparation',
            'shortHelpString':"It calculates kernel statistic from raster around points: real, max, min, std, sum, average, range",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'Corr',
            'function': CorrAlgorithm,
            'name':'Correlation plot',
            'displayName':'08 Correlation plot',
            'group':'01 Data preparation',
            'groupId':'01 Data preparation',
            'shortHelpString':"This function calculate the correlation plot between continuous variables",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'classv',
            'function': classvAlgorithm,
            'name':'classy vector by ROC',
            'displayName':'01 Classify vector by ROC',
            'group':'05 Classify SI',
            'groupId':'05 Classify SI',
            'shortHelpString':"Classifies a index (SI) maximizing the AUC of the relative ROC curve",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'classvW',
            'function': classvAlgorithmW,
            'name':'classy vector by wROC',
            'displayName':'02 Classify vector by weighted ROC',
            'group':'05 Classify SI',
            'groupId':'05 Classify SI',
            'shortHelpString':"Classifies a index (SI) maximizing the AUC of the relative weighted ROC curve",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'FP',
            'function': FPAlgorithm,
            'name':'Conf matrix',
            'displayName':'03 Confusion Matrix',
            'group':'05 Classify SI',
            'groupId':'05 Classify SI',
            'shortHelpString':"This function labels each feature as True Positive (0), True Negative (1), False Positive (2), False Negative (3)",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        # add additional algorithms here
        # self.addAlgorithm(MyOtherAlgorithm())

    def id(self):
        """
        Returns the unique provider id, used for identifying the provider. This
        string should be a unique, short, character only string, eg "qgis" or
        "gdal". This string should not be localised.
        """
        return 'SZ'

    def name(self):
        """
        Returns the provider name, which is used to describe the provider
        within the GUI.

        This string should be short (e.g. "Lastools") and localised.
        """
        return self.tr('SZ')

    def icon(self):
        """
        Should return a QIcon which is used for your provider inside
        the Processing toolbox.
        """
        return QIcon(':/icon')

    def longName(self):
        """
        Returns the a longer version of the provider name, which can include
        extra details such as version numbers. E.g. "Lastools LIDAR tools
        (version 2.2.1)". This string should be localised. The default
        implementation returns the same string as name().
        """
        return self.name()

class Instance(QgsProcessingAlgorithm):
    INPUT = 'INPUT'#'covariates'
    INPUT1 = 'INPUT1'#'input1'
    STRING = 'STRING'#'field1'
    STRING1 = 'STRING1'#'field2'
    STRING2 = 'STRING2'#'fieldlsd'
    STRING3 = 'STRING3'#'field3'
    STRING4 = 'STRING4'#'string4'
    STRING5 = 'STRING5'
    NUMBER = 'NUMBER'#'testN'
    NUMBER1 = 'NUMBER1'#'num1'
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
        #self.class_function=self.dict_of_scripts['function']()
        self.algorithms={
            'SVC_cv':Algorithms.alg_MLrun,
            'RF_cv':Algorithms.alg_MLrun,
            'LR_cv':Algorithms.alg_MLrun,
            'fr_cv':Algorithms.alg_MLrun,
            'DT_cv':Algorithms.alg_MLrun,
            'GAM_cv':Algorithms.alg_GAMrun,
            'GAM_trans':Algorithms.GAM_simple,
        }

        self.classifier={
            'SVC_cv':SVC(kernel = 'linear', random_state = 0,probability=True),
            'RF_cv':RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0),
            'LR_cv':LogisticRegression(),
            'DT_cv':DecisionTreeClassifier(criterion = 'entropy', random_state = 0),
            'GAM_cv':LogisticGAM,
            'GAM_simple':None,
            'GAM_trans':None,
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
        print(self.dict_of_scripts['alg'])
        print(self.algorithms)
        print(self.classifier)
        if self.algorithms[self.dict_of_scripts['alg']] or self.classifier[self.dict_of_scripts['alg']]:  
            result=self.dict_of_scripts['function'].process(self,parameters, context, feedback, algorithm=self.algorithms[self.dict_of_scripts['alg']], classifier=self.classifier[self.dict_of_scripts['alg']])
        else: 
            result=self.dict_of_scripts['function'].process(self,parameters, context, feedback)
        return result
        
