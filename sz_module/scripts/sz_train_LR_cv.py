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
from qgis.PyQt.QtCore import QCoreApplication,QVariant
from qgis.core import (QgsProcessing,
                       QgsFeatureSink,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterRasterLayer,
                       QgsMessageLog,
                       Qgis,
                       QgsProcessingMultiStepFeedback,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFileDestination,
                       QgsProcessingParameterVectorLayer,
                       QgsVectorLayer,
                       QgsRasterLayer,
                       QgsProject,
                       QgsField,
                       QgsFields,
                       QgsVectorFileWriter,
                       QgsWkbTypes,
                       QgsFeature,
                       QgsGeometry,
                       QgsPointXY,
                       QgsProcessingParameterField,
                       QgsProcessingParameterString,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterField,
                       QgsProcessingParameterVectorDestination,
                       QgsProcessingContext
                       )
from qgis.core import *
from qgis.utils import iface
from qgis import processing
from osgeo import gdal,ogr,osr
import numpy as np
import math
import operator
import random
from qgis import *
# ##############################
import matplotlib.pyplot as plt
import csv
from processing.algs.gdal.GdalUtils import GdalUtils
#import plotly.express as px
#import chart_studio
import plotly.offline
import plotly.graph_objs as go
#import geopandas as gd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from scipy import interpolate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import tempfile
from sz_module.utils import SZ_utils


class LRcvAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'covariates'
    STRING = 'field1'
    STRING2 = 'fieldlsd'
    NUMBER = 'testN'
    OUTPUT = 'OUTPUT'
    OUTPUT3 = 'OUTPUT3'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return LRcvAlgorithm()

    def name(self):
        return 'Fit-CV_LRcv'

    def displayName(self):
        return self.tr('03 LR Fitting/CrossValid')

    def group(self):
        return self.tr('SI k-fold')

    def groupId(self):
        return 'SI_k-fold'

    def shortHelpString(self):
        return self.tr("This function apply Logistic Regression to calculate susceptibility. It allows to cross-validate the analysis by k-fold cross-validation method. If you want just do fitting put k-fold equal to one")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Input layer'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))

        self.addParameter(QgsProcessingParameterField(self.STRING, 'Independent variables', parentLayerParameterName=self.INPUT, defaultValue=None, allowMultiple=True,type=QgsProcessingParameterField.Any))
        self.addParameter(QgsProcessingParameterField(self.STRING2, 'Field of dependent variable (0 for absence, > 0 for presence)', parentLayerParameterName=self.INPUT, defaultValue=None))

        self.addParameter(QgsProcessingParameterNumber(self.NUMBER, self.tr('K-fold CV (1 to fit or > 1 to cross-validate)'), minValue=1,type=QgsProcessingParameterNumber.Integer,defaultValue=2))

        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, 'Output test/fit',fileFilter='GeoPackage (*.gpkg *.GPKG)', defaultValue=None))
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

        parameters['folder'] = self.parameterAsString(parameters, self.OUTPUT3, context)
        if parameters['folder'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT3))

        alg_params = {
            'INPUT_VECTOR_LAYER': parameters['covariates'],
            'field1': parameters['field1'],
            'lsd' : parameters['fieldlsd'],
            'testN':parameters['testN'],
            'fold':parameters['folder']
        }

        outputs['prob'],outputs['test_ind'],outputs['df'],outputs['nomi'],outputs['crs']=SZ_utils.load_cv(self.f,alg_params)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        if parameters['testN']>0:
            alg_params = {
                'df': outputs['df'],
                'crs': outputs['crs'],
                'OUT': parameters['out']
            }
            self.save(alg_params)

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        alg_params = {
            'test_ind': outputs['test_ind'],
            'df': outputs['df'],
            'OUT':parameters['folder']
        }
        SZ_utils.stamp_cv(alg_params)

        feedback.setCurrentStep(3)
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

        feedback.setCurrentStep(4)
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
    #         #print(attr)
    #         geom = feature.geometry()
    #         #print(type(geom.asWkt()))
    #         feat=attr+[geom.asWkt()]
    #         #print(feat)
    #         gdp.loc[len(gdp)] = feat
    #         #gdp = gdp.append(feat, ignore_index=True)
    #         count=+ 1
    #     gdp.to_csv(self.f+'/file.csv')
    #     del gdp
    #     gdp=pd.read_csv(self.f+'/file.csv')
    #     #print(feat)
    #     #print(gdp['S'].dtypes)
    #     gdp['ID']=np.arange(1,len(gdp.iloc[:,0])+1)
    #     df=gdp[parameters['field1']]
    #     nomi=list(df.head())
    #     #print(list(df['Sf']),'1')
    #     lsd=gdp[parameters['lsd']]
    #     lsd[lsd>0]=1
    #     df['y']=lsd#.astype(int)
    #     df['ID']=gdp['ID']
    #     df['geom']=gdp['geom']
    #     df=df.dropna(how='any',axis=0)
    #     return(df,nomi,crs)

    def cross_validation(parameters,df,nomi):
        x=df[parameters['field1']]
        y=df['y']
        sc = StandardScaler()#####scaler
        X = sc.fit_transform(x)
        classifier=LogisticRegression()
        train_ind={}
        test_ind={}
        prob={}
        cofl=[]
        df["SI"] = np.nan
        if parameters['testN']>1:
            cv = StratifiedKFold(n_splits=parameters['testN'])
            for i, (train, test) in enumerate(cv.split(X, y)):
                train_ind[i]=train
                test_ind[i]=test
                prob[i],coeff=self.LR(classifier,X,y,train,test)
                df.loc[test,'SI']=prob[i]
                cofl.append(coeff)
        elif parameters['testN']==1:
            train=np.arange(len(y))
            test=np.arange(len(y))
            prob[0],coeff=self.LR(classifier,X,y,train,test)
            df.loc[test,'SI']=prob[0]
            test_ind[0]=test
            cofl.append(coeff)
        if not os.path.exists(parameters['fold']):
            os.mkdir(parameters['fold'])
        with open(parameters['fold']+'/r_coeffs.csv', 'w') as f:
            write = csv.writer(f)
            ll=['intercept']
            lll=ll+nomi
            write.writerow(lll)
            write.writerows(cofl)
        return prob,test_ind,df,nomi

    def LR(self,classifier,X,y,train,test):
        classifier.fit(X[train], y[train])
        prob_predic=classifier.predict_proba(X[test])[::,1]
        regression_coeff=classifier.coef_
        regression_intercept=classifier.intercept_
        coeff=np.hstack((regression_intercept,regression_coeff[0]))
        print(coeff,'regression coeff')
        #prob_fit=classifier.predict_proba(X[train])[::,1]
        return prob_predic,coeff


        # from sklearn.model_selection import cross_val_predict
        # sc = StandardScaler()
        # nomi=parameters['nomi']
        # train=parameters['train']
        # test=parameters['testy']
        # X_train = sc.fit_transform(train[nomi])
        # logistic_regression = LogisticRegression()
        # logistic_regression.fit(X_train,train['y'])
        # prob_fit=logistic_regression.predict_proba(X_train)[::,1]
        # if parameters['testN']>0:
        #     X_test = sc.transform(test[nomi])
        #     #predictions = logistic_regression.predict(X_test)
        #     predictions = cross_val_predict(logistic_regression, X_test, test['y'], cv=3)
        #     prob_predic = cross_val_predict(logistic_regression, X_test, test['y'], cv=3, method='predict_proba')[::,1]
        #     #prob_predic=logistic_regression.predict_proba(X_test)[::,1]
        #     print(predictions)
        #     print(prob_predic)
        #     test['SI']=prob_predic
        # train['SI']=prob_fit
        # return(train,test)

    # def stampfit(self,parameters):
    #     df=parameters['df']
    #     y_true=df['y']
    #     scores=df['SI']
    #     #W=df['w']
    #     ################################figure
    #     #fpr1, tpr1, tresh1 = roc_curve(y_true,scores,sample_weight=W)
    #     fpr1, tpr1, tresh1 = roc_curve(y_true,scores)
    #     norm=(scores-scores.min())/(scores.max()-scores.min())
    #
    #     #fpr2, tpr2, tresh2 = roc_curve(self.y_true,self.norm)
    #     #print(tresh1)
    #
    #     #fprv, tprv, treshv = roc_curve(self.y_v,self.scores_v)
    #     #fprt, tprt, tresht = roc_curve(self.y_t,self.scores_t)
    #
    #     #print self.fpr
    #     #print self.tpr
    #     #print self.classes
    #     #aucv=roc_auc_score(self.y_v, self.scores_v, None)
    #     #auct=roc_auc_score(self.y_t, self.scores_t, None)
    #     r=roc_auc_score(y_true, scores, None)
    #
    #     fig=plt.figure()
    #     lw = 2
    #     plt.plot(fpr1, tpr1, color='green',lw=lw, label= 'Complete dataset (AUC = %0.2f)' %r)
    #     #plt.plot(self.fpr, self.tpr, 'ro')
    #     #plt.plot(self.fpr, self.tpr, color='darkorange',lw=lw, label='Classified dataset (AUC = %0.2f)' % self.fitness)
    #     plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('ROC')
    #     plt.legend(loc="lower right")
    #     #plt.show()
    #     try:
    #         fig.savefig(parameters['OUT']+'/fig01.png')
    #     except:
    #         os.mkdir(parameters['OUT'])
    #         fig.savefig(parameters['OUT']+'/fig01.png')
    #     # fig=plt.figure()
    #     # frequency, bins = np.histogram(norm)
    #     # frequency=(frequency/len(norm))*100
    #     # bincenters = 0.5*(bins[1:]+bins[:-1])
    #     # plt.hist(bins[:-1], bins, weights=frequency,color='blue',alpha = 0.8)
    #     # #plt.plot(bincenters,frequency,'-')#segmented curve
    #     # #print(bincenters,frequency)
    #     #
    #     # #xnew = interpolate.splrep(bincenters, frequency, s=0)
    #     # xnew = np.linspace(bincenters.min(),bincenters.max())
    #     # #print(bincenters, xnew)
    #     # power_smooth=interpolate.splev(bincenters, xnew, der=0)
    #     # #power_smooth = spline(bincenters,frequency,xnew)
    #     # plt.plot(xnew,power_smooth,color='black',lw=lw, label= 'LSI')
    #     # plt.xlabel('Standardized Susceptibility Index')
    #     # plt.ylabel('Area %')
    #     # plt.title('')
    #     # plt.legend(loc="upper right")
    #     # #plt.show()
    #     # fig.savefig(parameters['OUT']+'/fig02.png')

    def stampcv(self,parameters):
        df=parameters['df']
        test_ind=parameters['test_ind']

        #test=parameters['test']
        y_v=df['y']
        scores_v=df['SI']
        lw = 2
        #W=df['w']
        ################################figure
        #fpr1, tpr1, tresh1 = roc_curve(y_true,scores,sample_weight=W)
        #fpr1, tpr1, tresh1 = roc_curve(y_true,scores)
        #fpr2, tpr2, tresh2 = roc_curve(self.y_true,self.norm)
        #print(tresh1)
        fig=plt.figure()
        #print(len(test_ind))
        #print(test_ind)
        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        for i in range(len(test_ind)):

            fprv, tprv, treshv = roc_curve(y_v[test_ind[i]],scores_v[test_ind[i]])
            #fprt, tprt, tresht = roc_curve(y_t,scores_t)
            #print self.tpr
            #print self.classes
            aucv=roc_auc_score(y_v[test_ind[i]],scores_v[test_ind[i]])
            print('ROC '+ str(i) +' AUC=',aucv)
            #auct=roc_auc_score(y_t, scores_t, None)
            #r=roc_auc_score(y_true, scores, None)
            #normt=(scores_t-scores_t.min())/(scores_t.max()-scores_t.min())
            #normv=(scores_v-scores_v.min())/(scores_v.max()-scores_v.min())



            plt.plot(fprv, tprv,lw=lw, alpha=0.5, label='ROC fold '+str(i+1)+' (AUC = %0.2f)' %aucv)
            #plt.plot(fprt, tprt, color='red',lw=lw, label= 'Success performance (AUC = %0.2f)' %auct)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.title('ROC')
        plt.legend(loc="lower right")
        #plt.show()
        print('ROC curve figure = ',parameters['OUT']+'/fig02.pdf')
        try:
            fig.savefig(parameters['OUT']+'/fig02.pdf')

        except:
            os.mkdir(parameters['OUT'])
            fig.savefig(parameters['OUT']+'/fig02.pdf')

        # fig=plt.figure()
        # frequency, bins = np.histogram(normt)



        # fig1=plt.figure()
        # #print(len(test_ind))
        # #print(test_ind)
        # plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        # for i in range(len(test_ind)):
        #
        #     #fprv, tprv, treshv = roc_curve(y_v[test_ind[i]],scores_v[test_ind[i]])
        #     #fprt, tprt, tresht = roc_curve(y_t,scores_t)
        #     from sklearn.metrics import precision_recall_curve
        #     from sklearn.metrics import PrecisionRecallDisplay
        #
        #     prec, recall, _ = precision_recall_curve(y_v[test_ind[i]], scores_v[test_ind[i]])
        #     pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)
        #     print(pr_display,'disp')
        #     print(prec,recall)
        #     #print self.tpr
        #     #print self.classes
        #     #aucv=roc_auc_score(y_v[test_ind[i]],scores_v[test_ind[i]], None)
        #     #print('ROC '+ str(i) +' AUC=',aucv)
        #     #auct=roc_auc_score(y_t, scores_t, None)
        #     #r=roc_auc_score(y_true, scores, None)
        #     #normt=(scores_t-scores_t.min())/(scores_t.max()-scores_t.min())
        #     #normv=(scores_v-scores_v.min())/(scores_v.max()-scores_v.min())
        #
        #
        #
        #     plt.plot(fprv, tprv,lw=lw, alpha=0.5)
        #     #plt.plot(fprt, tprt, color='red',lw=lw, label= 'Success performance (AUC = %0.2f)' %auct)
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # #plt.title('ROC')
        # plt.legend(loc="lower right")
        # #plt.show()
        # print('PRC curve figure = ',parameters['OUT']+'/fig03.pdf')
        # try:
        #     fig1.savefig(parameters['OUT']+'/fig03.pdf')
        #
        # except:
        #     os.mkdir(parameters['OUT'])
        #     fig.savefig(parameters['OUT']+'/fig03.pdf')

        # fig=plt.figure()
        # frequency, bins = np.histogram(normt)
        # bincenters = 0.5*(bins[1:]+bins[:-1])
        # plt.hist(bins[:-1], bins, weights=frequency,color='blue',alpha = 0.8)
        # #plt.plot(bincenters,frequency,'-')#segmented curve
        # xnew = np.linspace(bincenters.min(),bincenters.max())
        # power_smooth=interpolate.splev(bincenters, xnew, der=0)
        # #power_smooth = spline(bincenters,frequency,xnew)
        # plt.plot(xnew,power_smooth,color='black',lw=lw, label= 'Train SI')
        # plt.xlabel('Standardized Susceptibility Index')
        # plt.ylabel('Area %')
        # plt.title('')
        # plt.legend(loc="upper right")
        # fig.savefig(parameters['OUT']+'/fig02.png') # Use fig. here
        # #plt.show()
        #
        # fig=plt.figure()
        # frequency, bins = np.histogram(normv)
        # frequency=(frequency/len(normv))*100
        # bincenters = 0.5*(bins[1:]+bins[:-1])
        # plt.hist(bins[:-1], bins, weights=frequency,color='blue',alpha = 0.8)
        # #plt.plot(bincenters,frequency,'-')#segmented curve
        # xnew = np.linspace(bincenters.min(),bincenters.max())
        # power_smooth=interpolate.splev(bincenters, xnew, der=0)
        # #power_smooth = spline(bincenters,frequency,xnew)
        # plt.plot(xnew,power_smooth,color='black',lw=lw, label= 'Test SI')
        # plt.xlabel('Standardized Susceptibility Index')
        # plt.ylabel('Area %')
        # plt.title('')
        # plt.legend(loc="upper right")
        # fig.savefig(parameters['OUT']+'/fig03.png') # Use fig. here

    def save(self,parameters):

        #print(parameters['nomi'])
        df=parameters['df']
        nomi=list(df.head())
        # define fields for feature attributes. A QgsFields object is needed
        fields = QgsFields()

        #fields.append(QgsField('ID', QVariant.Int))

        for field in nomi:
            if field=='ID':
                fields.append(QgsField(field, QVariant.Int))
            if field=='geom':
                continue
            if field=='y':
                fields.append(QgsField(field, QVariant.Int))
            else:
                fields.append(QgsField(field, QVariant.Double))

        #crs = QgsProject.instance().crs()
        transform_context = QgsProject.instance().transformContext()
        save_options = QgsVectorFileWriter.SaveVectorOptions()
        save_options.driverName = 'GPKG'
        save_options.fileEncoding = 'UTF-8'

        writer = QgsVectorFileWriter.create(
          parameters['OUT'],
          fields,
          QgsWkbTypes.Polygon,
          parameters['crs'],
          transform_context,
          save_options
        )

        if writer.hasError() != QgsVectorFileWriter.NoError:
            print("Error when creating shapefile: ",  writer.errorMessage())
        for i, row in df.iterrows():
            fet = QgsFeature()
            fet.setGeometry(QgsGeometry.fromWkt(row['geom']))
            fet.setAttributes(list(map(float,list(df.loc[ i, df.columns != 'geom']))))
            writer.addFeature(fet)

        # delete the writer to flush features to disk
        del writer

    def addmap(self,parameters):
        context=parameters()
        fileName = parameters['trainout']
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
            context.addLayerToLoadOnCompletion(sub_vlayer.id(), QgsProcessingContext.LayerDetails('layer', context.project(),'LAYER'))

            #QgsProject.instance().addMapLayer(sub_vlayer)
            #iface.mapCanvas().refresh()


        # fileName = parameters['out']
        # layer = QgsVectorLayer(fileName,"test","ogr")
        # subLayers =layer.dataProvider().subLayers()
        #
        # for subLayer in subLayers:
        #     name = subLayer.split('!!::!!')[1]
        #     uri = "%s|layername=%s" % (fileName, name,)
        #     # Create layer
        #     sub_vlayer = QgsVectorLayer(uri, name, 'ogr')
        #     if not sub_vlayer.isValid():
        #         print('layer failed to load')
        #     # Add layer to map
        #     QgsProject.instance().addMapLayer(sub_vlayer)
