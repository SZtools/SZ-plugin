#!/usr/bin/python
#coding=utf-8
"""
/***************************************************************************
    CleanPointsByRasterKernelValue
        begin                : 2020-03
        copyright            : (C) 2020 by Giacomo Titti,
                               Padova, March 2020
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
    CleanPointsByRasterKernelValue
    Copyright (C) 2020 by Giacomo Titti, Padova, March 2020

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
import sys
sys.setrecursionlimit(10000)
from qgis.PyQt.QtCore import QCoreApplication,QVariant
from qgis.core import (QgsProcessing,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingMultiStepFeedback,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFileDestination,
                       QgsProcessingParameterVectorLayer,
                       QgsVectorLayer,
                       QgsProject,
                       QgsField,
                       QgsFields,
                       QgsVectorFileWriter,
                       QgsWkbTypes,
                       QgsFeature,
                       QgsGeometry,
                       QgsProcessingParameterField,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterField,
                       QgsProcessingContext
                       )
from qgis.core import *
from qgis.utils import iface
import numpy as np
from qgis import *
# ##############################
import matplotlib.pyplot as plt
from processing.algs.gdal.GdalUtils import GdalUtils
import plotly.graph_objs as go
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import tempfile
from sz_module.utils import SZ_utils


class DTAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'covariates'
    STRING = 'field1'
    STRING2 = 'fieldlsd'
    NUMBER = 'testN'
    OUTPUT = 'OUTPUT'
    OUTPUT1 = 'OUTPUT1'
    OUTPUT3 = 'OUTPUT3'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return DTAlgorithm()

    def name(self):
        return 'Fit-CV_DT'

    def displayName(self):
        return self.tr('06 DT Fitting/CrossValid')

    def group(self):
        return self.tr('SI')

    def groupId(self):
        return 'SI'

    def shortHelpString(self):
        return self.tr("This function apply Decision Tree to calculate susceptibility. It allows to cross-validate the analysis selecting the sample percentage test/training. If you want just do fitting put the test percentage equal to zero")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Input layer'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING, 'Independent variables', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=True,type=QgsProcessingParameterField.Any))
        self.addParameter(QgsProcessingParameterField(self.STRING2, 'Field of dependent variable (0 for absence, > 0 for presence)', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER, self.tr('Percentage of test sample (0 to fit, > 0 to cross-validate)'), type=QgsProcessingParameterNumber.Integer,defaultValue=30))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, 'Output test [mandatory if Test percentage > 0]',fileFilter='GeoPackage (*.gpkg *.GPKG)', defaultValue=None))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT1, 'Output train/fit',fileFilter='GeoPackage (*.gpkg *.GPKG)', defaultValue=None))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT3, 'Outputs folder destination', defaultValue=None, createByDefault = True))

    def processAlgorithm(self, parameters, context, feedback):
        self.f=tempfile.gettempdir()

        feedback = QgsProcessingMultiStepFeedback(1, feedback)
        results = {}
        outputs = {}

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

        parameters['testN'] = self.parameterAsInt(parameters, self.NUMBER, context)
        if parameters['testN'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER))
       
        parameters['out'] = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        if parameters['out'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))

        parameters['out1'] = self.parameterAsFileOutput(parameters, self.OUTPUT1, context)
        if parameters['out1'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT1))

        parameters['folder'] = self.parameterAsString(parameters, self.OUTPUT3, context)
        if parameters['folder'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT3))

        # Intersectionpoly
        alg_params = {
            'INPUT_VECTOR_LAYER': parameters['covariates'],
            'field1': parameters['field1'],
            'lsd' : parameters['fieldlsd'],
            'testN':parameters['testN']
        }
        outputs['train'],outputs['testy'],outputs['nomes'],outputs['crs']=SZ_utils.load_simple(self.f,alg_params)

        alg_params = {
            'train': outputs['train'],
            'testy': outputs['testy'],
            'nomi':outputs['nomes'],
            'testN':parameters['testN']
        }
        outputs['trainsi'],outputs['testsi']=self.DT(alg_params)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        if parameters['testN']>0:
            alg_params = {
                'df': outputs['testsi'],
                'crs': outputs['crs'],
                'OUT': parameters['out']
            }
            self.save(alg_params)

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        alg_params = {
            'df': outputs['trainsi'],
            'crs': outputs['crs'],
            'OUT': parameters['out1']
        }
        self.save(alg_params)

        if parameters['testN']==0:
            alg_params = {
                'df': outputs['trainsi'],
                'OUT':parameters['folder']

            }
            self.stampfit(alg_params)
        else:
            alg_params = {
                'train': outputs['trainsi'],
                'test': outputs['testsi'],
                'OUT':parameters['folder']
            }
            self.stampcv(alg_params)

        feedback.setCurrentStep(3)
        if feedback.isCanceled():
            return {}
       
        results['out'] = parameters['out']
        results['out1'] = parameters['out1']

        if parameters['testN']>0:
            fileName = parameters['out1']
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

        else:
            fileName = parameters['out1']
            layer = QgsVectorLayer(fileName,"fitting","ogr")
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
                context.addLayerToLoadOnCompletion(sub_vlayer.id(), QgsProcessingContext.LayerDetails('fitting', context.project(),'LAYER'))

        feedback.setCurrentStep(3)
        if feedback.isCanceled():
            return {}

        return results

    # def load(self,parameters):
    #     layer = QgsVectorLayer(parameters['INPUT_VECTOR_LAYER'], '', 'ogr')
    #     crs=layer.crs()
    #     campi=[]
    #     for field in layer.fields():
    #         campi.append(field.name())
    #     campi.append('geom')
    #     gdp=pd.DataFrame(columns=campi,dtype=float)
    #     features = layer.getFeatures()
    #     count=0
    #     feat=[]
    #     for feature in features:
    #         attr=feature.attributes()
    #         geom = feature.geometry()
    #         feat=attr+[geom.asWkt()]
    #         gdp.loc[len(gdp)] = feat
    #         count=+ 1
    #     gdp.to_csv(self.f+'/file.csv')
    #     del gdp
    #     gdp=pd.read_csv(self.f+'/file.csv')
    #     gdp['ID']=np.arange(1,len(gdp.iloc[:,0])+1)
    #     df=gdp[parameters['field1']]
    #     nomi=list(df.head())
    #     lsd=gdp[parameters['lsd']]
    #     lsd[lsd>0]=1
    #     df['y']=lsd#.astype(int)
    #     df['ID']=gdp['ID']
    #     df['geom']=gdp['geom']
    #     df=df.dropna(how='any',axis=0)
    #     X=[parameters['field1']]
    #     if parameters['testN']==0:
    #         train=df
    #         test=pd.DataFrame(columns=nomi,dtype=float)
    #     else:
    #         # split the data into train and test set
    #         per=int(np.ceil(df.shape[0]*parameters['testN']/100))
    #         train, test = train_test_split(df, test_size=per, random_state=42, shuffle=True)
    #     return train, test, nomi,crs

    def DT(self,parameters):
        sc = StandardScaler()
        nomi=parameters['nomi']
        train=parameters['train']
        test=parameters['testy']
        X_train = sc.fit_transform(train[nomi])
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train,train['y'])
        prob_fit=classifier.predict_proba(X_train)[::,1]
        if parameters['testN']>0:
            X_test = sc.transform(test[nomi])
            predictions = classifier.predict(X_test)
            prob_predic=classifier.predict_proba(X_test)[::,1]
            test['SI']=prob_predic
        train['SI']=prob_fit
        return(train,test)

    # def stampfit(self,parameters):
    #     df=parameters['df']
    #     y_true=df['y']
    #     scores=df['SI']
    #     ################################figure
    #     fpr1, tpr1, tresh1 = roc_curve(y_true,scores)
    #     norm=(scores-scores.min())/(scores.max()-scores.min())
    #     r=roc_auc_score(y_true, scores)

    #     fig=plt.figure()
    #     lw = 2
    #     plt.plot(fpr1, tpr1, color='green',lw=lw, label= 'Complete dataset (AUC = %0.2f)' %r)
    #     plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('ROC')
    #     plt.legend(loc="lower right")
    #     try:
    #         fig.savefig(parameters['OUT']+'/fig01.png')
    #     except:
    #         os.mkdir(parameters['OUT'])
    #         fig.savefig(parameters['OUT']+'/fig01.png')
    
    # def stampcv(self,parameters):
    #     train=parameters['train']
    #     y_t=train['y']
    #     scores_t=train['SI']

    #     test=parameters['test']
    #     y_v=test['y']
    #     scores_v=test['SI']
    #     lw = 2
        
    #     fprv, tprv, treshv = roc_curve(y_v,scores_v)
    #     fprt, tprt, tresht = roc_curve(y_t,scores_t)

    #     aucv=roc_auc_score(y_v, scores_v)
    #     auct=roc_auc_score(y_t, scores_t)
    #     normt=(scores_t-scores_t.min())/(scores_t.max()-scores_t.min())
    #     normv=(scores_v-scores_v.min())/(scores_v.max()-scores_v.min())

    #     fig=plt.figure()
    #     plt.plot(fprv, tprv, color='green',lw=lw, label= 'Prediction performance (AUC = %0.2f)' %aucv)
    #     plt.plot(fprt, tprt, color='red',lw=lw, label= 'Success performance (AUC = %0.2f)' %auct)
    #     plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('ROC')
    #     plt.legend(loc="lower right")
    #     #plt.show()
    #     try:
    #         fig.savefig(parameters['OUT']+'/fig02.pdf')
    #     except:
    #         os.mkdir(parameters['OUT'])
    #         fig.savefig(parameters['OUT']+'/fig02.pdf')

    # def save(self,parameters):

    #     df=parameters['df']
    #     nomi=list(df.head())
    #     fields = QgsFields()

    #     for field in nomi:
    #         if field=='ID':
    #             fields.append(QgsField(field, QVariant.Int))
    #         if field=='geom':
    #             continue
    #         if field=='y':
    #             fields.append(QgsField(field, QVariant.Int))
    #         else:
    #             fields.append(QgsField(field, QVariant.Double))

    #     transform_context = QgsProject.instance().transformContext()
    #     save_options = QgsVectorFileWriter.SaveVectorOptions()
    #     save_options.driverName = 'GPKG'
    #     save_options.fileEncoding = 'UTF-8'

    #     writer = QgsVectorFileWriter.create(
    #       parameters['OUT'],
    #       fields,
    #       QgsWkbTypes.Polygon,
    #       parameters['crs'],
    #       transform_context,
    #       save_options
    #     )

    #     if writer.hasError() != QgsVectorFileWriter.NoError:
    #         print("Error when creating shapefile: ",  writer.errorMessage())
    #     for i, row in df.iterrows():
    #         fet = QgsFeature()
    #         fet.setGeometry(QgsGeometry.fromWkt(row['geom']))
    #         fet.setAttributes(list(map(float,list(df.loc[ i, df.columns != 'geom']))))
    #         writer.addFeature(fet)

    #     del writer

    # def addmap(self,parameters):
    #     context=parameters()
    #     fileName = parameters['trainout']
    #     layer = QgsVectorLayer(fileName,"train","ogr")
    #     subLayers =layer.dataProvider().subLayers()

    #     for subLayer in subLayers:
    #         name = subLayer.split('!!::!!')[1]
    #         print(name,'name')
    #         uri = "%s|layername=%s" % (fileName, name,)
    #         print(uri,'uri')
    #         # Create layer
    #         sub_vlayer = QgsVectorLayer(uri, name, 'ogr')
    #         if not sub_vlayer.isValid():
    #             print('layer failed to load')
    #         # Add layer to map
    #         context.temporaryLayerStore().addMapLayer(sub_vlayer)
    #         context.addLayerToLoadOnCompletion(sub_vlayer.id(), QgsProcessingContext.LayerDetails('layer', context.project(),'LAYER'))
