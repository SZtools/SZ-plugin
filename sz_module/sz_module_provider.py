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
from sz_module.scripts.sz_train_simple import CoreAlgorithm
from sz_module.scripts.sz_train_cv import CoreAlgorithm_cv
from sz_module.scripts.sz_train_simple_GAM import CoreAlgorithmGAM
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

        dict_of_scripts={
            'alg': 'woe_simple',
            'function': CoreAlgorithm,
            'name':'Fit-CV_WOE',
            'displayName':'01 WoE',
            'group':'SI',
            'groupId':'SI',
            'shortHelpString':"This function apply Weight of Evidence to calculate susceptibility. It allows to cross-validate the analysis selecting the sample percentage test/training. If you want just do fitting put the test percentage equal to zero",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'woe_cv',
            'function': CoreAlgorithm_cv,
            'name':'Fit-CV_WOEcv',
            'displayName':'01 WoE',
            'group':'SI k-fold',
            'groupId':'SI k-fold',
            'shortHelpString':"This function apply Weight of Evidence to calculate susceptibility. It allows to cross-validate the analysis by k-fold cross-validation method. If you want just do fitting put k-fold equal to one",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'SVC_simple',
            'function': CoreAlgorithm,
            'name':'Fit-CV_SVC',
            'displayName':'05 SVM',
            'group':'SI',
            'groupId':'SI',
            'shortHelpString':"This function apply Support Vector Machine to calculate susceptibility. It allows to cross-validate the analysis selecting the sample percentage test/training. If you want just do fitting put the test percentage equal to zero",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'SVC_cv',
            'function': CoreAlgorithm_cv,
            'name':'Fit-CV_SVCcv',
            'displayName':'05 SVM',
            'group':'SI k-fold',
            'groupId':'SI k-fold',
            'shortHelpString':"This function apply Support Vector Machine to calculate susceptibility. It allows to cross-validate the analysis by k-fold cross-validation method. If you want just do fitting put k-fold equal to one",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'RF_simple',
            'function': CoreAlgorithm,
            'name':'Fit-CV_RF',
            'displayName':'04 RF',
            'group':'SI',
            'groupId':'SI',
            'shortHelpString':"This function apply Random Forest to calculate susceptibility. It allows to cross-validate the analysis selecting the sample percentage test/training. If you want just do fitting put the test percentage equal to zero",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'RF_cv',
            'function': CoreAlgorithm_cv,
            'name':'Fit-CV_RFcv',
            'displayName':'04 RF',
            'group':'SI k-fold',
            'groupId':'SI k-fold',
            'shortHelpString':"This function apply Random Forest to calculate susceptibility. It allows to cross-validate the analysis by k-fold cross-validation method. If you want just do fitting put k-fold equal to one",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'LR_simple',
            'function': CoreAlgorithm,
            'name':'Fit-CV_LR',
            'displayName':'03 LR',
            'group':'SI',
            'groupId':'SI',
            'shortHelpString':"This function apply Logistic Regression to calculate susceptibility. It allows to cross-validate the analysis selecting the sample percentage test/training. If you want just do fitting put the test percentage equal to zero",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'LR_cv',
            'function': CoreAlgorithm_cv,
            'name':'Fit-CV_LRcv',
            'displayName':'03 LR',
            'group':'SI k-fold',
            'groupId':'SI k-fold',
            'shortHelpString':"This function apply Logistic Regression to calculate susceptibility. It allows to cross-validate the analysis by k-fold cross-validation method. If you want just do fitting put k-fold equal to one",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'fr_simple',
            'function': CoreAlgorithm,
            'name':'Fit-CV_FR',
            'displayName':'02 FR',
            'group':'SI',
            'groupId':'SI',
            'shortHelpString':"This function apply Frequency Ratio to calculate susceptibility. It allows to cross-validate the analysis selecting the sample percentage test/training. If you want just do fitting put the test percentage equal to zero",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'fr_cv',
            'function': CoreAlgorithm_cv,
            'name':'Fit-CV_FRcv',
            'displayName':'02 FR',
            'group':'SI k-fold',
            'groupId':'SI k-fold',
            'shortHelpString':"This function apply Frequency Ratio to calculate susceptibility. It allows to cross-validate the analysis by k-fold cross-validation method. If you want just do fitting put k-fold equal to one",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'DT_simple',
            'function': CoreAlgorithm,
            'name':'Fit-CV_DT',
            'displayName':'06 DT',
            'group':'SI',
            'groupId':'SI',
            'shortHelpString':"This function apply Decision Tree to calculate susceptibility. It allows to cross-validate the analysis selecting the sample percentage test/training. If you want just do fitting put the test percentage equal to zero",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'DT_cv',
            'function': CoreAlgorithm_cv,
            'name':'Fit-CV_DTcv',
            'displayName':'06 DT',
            'group':'SI k-fold',
            'groupId':'SI k-fold',
            'shortHelpString':"This function apply Decision Tree to calculate susceptibility. It allows to cross-validate the analysis by k-fold cross-validation method. If you want just do fitting put k-fold equal to one",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'GAM_simple',
            'function': CoreAlgorithmGAM,
            'name':'Fit-CV_GAM',
            'displayName':'07 GAM',
            'group':'SI',
            'groupId':'SI',
            'shortHelpString':"This function apply Generalized Additive Model to calculate susceptibility. It allows to cross-validate the analysis selecting the sample percentage test/training. If you want just do fitting put the test percentage equal to zero",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'GAM_cv',
            'function': CoreAlgorithmGAM_cv,
            'name':'Fit-CV_GAMcv',
            'displayName':'07 GAM',
            'group':'SI k-fold',
            'groupId':'SI k-fold',
            'shortHelpString':"This function apply Generalized Additive Model to calculate susceptibility. It allows to cross-validate the analysis by k-fold cross-validation method. If you want just do fitting put k-fold equal to one",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        dict_of_scripts={
            'alg': 'GAM_trans',
            'function': CoreAlgorithmGAM_trans,
            'name':'Transfer_GAM',
            'displayName':'01 GAM',
            'group':'SI Transfer',
            'groupId':'SI Transfer',
            'shortHelpString':"This function apply Generalized Additive Model to transfer susceptibility",
        }
        self.addAlgorithm(Instance(dict_of_scripts))

        self.addAlgorithm(classcovtxtAlgorithm())
        self.addAlgorithm(classcovdecAlgorithm())
        ##self.addAlgorithm(polytogridAlgorithm())
        #self.addAlgorithm(pointtogridAlgorithm())
        self.addAlgorithm(statistic())

        #self.addAlgorithm(classAlgorithm())
        #self.addAlgorithm(rocAlgorithm())
        #self.addAlgorithm(matrixAlgorithm())

        self.addAlgorithm(cleankernelAlgorithm())
        self.addAlgorithm(statistickernel())
        self.addAlgorithm(samplerAlgorithm())
        self.addAlgorithm(rasterstatkernelAlgorithm())
        self.addAlgorithm(CorrAlgorithm())

        self.addAlgorithm(classvAlgorithm())
        self.addAlgorithm(classvAlgorithmW())
        self.addAlgorithm(FPAlgorithm())

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
    INPUT = 'covariates'
    INPUT1 = 'input1'
    STRING = 'field1'
    STRING1 = 'field2'
    STRING2 = 'fieldlsd'
    STRING3 = 'field3'
    NUMBER = 'testN'
    NUMBER1 = 'num1'
    OUTPUT = 'OUTPUT'
    OUTPUT1 = 'OUTPUT1'
    OUTPUT2 = 'OUTPUT2'
    OUTPUT3 = 'OUTPUT3'
    
    def __init__(self, dict_of_scripts):
        super().__init__()
        self.dict_of_scripts = dict_of_scripts
        #self.class_function=self.dict_of_scripts['function']()
        self.algorithms={
            'woe_simple':Algorithms.woe_simple,
            'woe_cv':Algorithms.woe_cv,
            'SCV_simple':Algorithms.SVC_simple,
            'SVC_cv':Algorithms.SVC_cv,
            'RF_simple':Algorithms.RF_simple,
            'RF_cv':Algorithms.RF_cv,
            'LR_simple':Algorithms.LR_simple,
            'LR_cv':Algorithms.LR_cv,
            'fr_simple':Algorithms.fr_simple,
            'fr_cv':Algorithms.fr_cv,
            'DT_simple':Algorithms.DT_simple,
            'DT_cv':Algorithms.DT_cv,
            'GAM_simple':Algorithms.GAM_simple,
            'GAM_cv':Algorithms.GAM_cv,
            'GAM_trans':Algorithms.GAM_simple,
        }

        self.classifier={
            'woe_cv':None,
            'SVC_cv':SVC(kernel = 'linear', random_state = 0,probability=True),
            'RF_cv':RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0),
            'LR_cv':LogisticRegression(),
            'fr_cv':None,
            'DT_cv':DecisionTreeClassifier(criterion = 'entropy', random_state = 0),
            'GAM_cv':LogisticGAM,
            'woe_simple':None,
            'SCV_simple':None,
            'RF_simple':None,
            'LR_simple':None,
            'fr_simple':None,
            'DT_simple':None,
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
        result=self.dict_of_scripts['function'].process(self,parameters, context, feedback, algorithm=self.algorithms[self.dict_of_scripts['alg']], classifier=self.classifier[self.dict_of_scripts['alg']])
        return result
        
