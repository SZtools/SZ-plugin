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
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterVectorLayer,
                       QgsVectorLayer,
                       QgsProcessingParameterField,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterField,
                       )
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import operator
import os
import numpy as np
import math
import operator
from qgis import *
# ##############################
import pandas as pd
import tempfile
from sz_module.scripts.utils import SZ_utils

class classvAlgorithm(QgsProcessingAlgorithm):
    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Input layer'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING, 'Index', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING2, 'Field of dependent variable (0 for absence, > 0 for presence)', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER, self.tr('Number of classes'), type=QgsProcessingParameterNumber.Integer, defaultValue = None,  minValue=1))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT3, 'Folder destination', defaultValue=None, createByDefault = True))

    def process(self, parameters, context, model_feedback):
        self.f=tempfile.gettempdir()
        feedback = QgsProcessingMultiStepFeedback(1, model_feedback)
        results = {}
        outputs = {}
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
        parameters['edgesGA'] = self.parameterAsString(parameters, self.OUTPUT3, context)
        if parameters['edgesGA'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT3))
        parameters['classes'] = self.parameterAsEnum(parameters, self.NUMBER, context)
        if parameters['classes'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER))

        SZ_utils.make_directory({'path':parameters['edgesGA']})

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
            'NUMBER': parameters['classes'],
            'OUTPUT': parameters['edgesGA']
        }
        outputs['ga']=Functions.classy(alg_params)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}
        return results

class Functions():

    def load(parameters):
        layer = QgsVectorLayer(parameters['INPUT_VECTOR_LAYER'], '', 'ogr')
        crs=layer.crs()
        campi=[]
        for field in layer.fields():
            campi.append(field.name())
        campi.append('geom')
        gdp=pd.DataFrame(columns=campi,dtype=float)
        df=pd.DataFrame(dtype=float)
        features = layer.getFeatures()
        count=0
        feat=[]
        for feature in features:
            attr=feature.attributes()
            geom = feature.geometry()
            feat=attr+[geom.asWkt()]
            gdp.loc[len(gdp)] = feat
            count=+ 1
        gdp.to_csv(self.f+'/file.csv')
        del gdp
        gdp=pd.read_csv(self.f+'/file.csv')
        gdp['ID']=np.arange(1,len(gdp.iloc[:,0])+1)
        df['SI']=gdp.loc[:,parameters['field1']]
        nomi=list(df.head())
        lsd=gdp[parameters['lsd']]
        lsd[lsd>0]=1
        df['y']=lsd#.astype(int)
        df['ID']=gdp['ID']
        df['geom']=gdp['geom']
        df=df.dropna(how='any',axis=0)
        return df,crs

    def classy(parameters):
        df=parameters['df']
        y_true=np.array(df['y']).reshape(-1,1)
        scores=np.array(df['SI']).reshape(-1,1)
        y_scores=np.array(df['SI']).reshape(-1,1)
        r=roc_auc_score(y_true, scores)
        print('AUC =',r)
        giri=20*parameters['NUMBER']
        numOff=giri#divisibile per 5
        Off=giri
        nclasses=parameters['NUMBER']
        M=np.max(scores)
        m=np.min(scores)
        count=0
        ran=np.array([])
        fitness=0
        values=np.array([])
        classes=([])
        c={}
        ran=np.array([])
        summ=0
        while count<Off:
            weight={}
            fpr={}
            tpr={}
            tresh={}
            roc_auc={}
            ran=np.array([])
            FPR={}
            TPR={}
            mm=None
            if count==0:
                c={}
                for pop in range(numOff):
                    ran=np.sort(np.random.random_sample(nclasses-1)*(M-m))
                    c[pop]=np.hstack((m,m+ran,M))
            else:
                c=file
            for k in range(numOff):
                weight[k]=y_scores
                for i in range(nclasses):
                    index=np.array([])
                    index=np.where((scores>=c[k][i]) & (scores<c[k][i+1]))
                    weight[k][index]=float(i+1)
                FPR[k],TPR[k]=Functions.rok(y_true,scores,nclasses,c[k])
                roc_auc[k]=np.trapz(TPR[k],FPR[k])
            mm=None
            mm=max(roc_auc, key=roc_auc.get)
            if roc_auc[mm]>fitness:#############################fitness
                print('fit!')
                fitness=None
                classes=np.array([])
                values=np.array([])
                ttpr=np.array([])
                ffpr=np.array([])
                fitness=roc_auc[mm]
                print(fitness,'fitness')
                classes=c[mm]
                values=weight[mm]
                print(classes,'classes')
                ttpr=TPR[mm]
                ffpr=FPR[mm]
                summ=1
            else:
                summ+=1
            ##########################PASS
            count+=1
            #########################GA
            file={}
            qq=0
            for q in range(0,numOff,nclasses):
                a=np.array([])
                bb={}
                cc=[]
                cc=list(roc_auc.items())
                bb=dict(cc[q:q+nclasses])
                a=sorted(bb.items(), key=operator.itemgetter(1),reverse=True)
                file[q]=c[a[0][0]]
                for b in range(1,nclasses):
                    file[q+b]=np.hstack((file[q][:b],file[q][b-1]+(np.sort(np.random.random_sample(1)*(file[q][b+1]-file[q][b-1]))),file[q][b+1:]))
                qq+=nclasses
        fitness=None
        tpr1=np.array([])
        fpr1=np.array([])
        values1=np.array([])
        classes1=np.array([])
        fitness1=fitness
        values1=values
        classes1=classes
        tpr1=ttpr
        fpr1=ffpr
        try:
            file = open(parameters['OUTPUT']+'/plotROC.txt','w')#################save txt
        except:
            os.mkdir(parameters['OUTPUT'])
            file = open(parameters['OUTPUT']+'/plotROC.txt','w')#################save txt
        var=[fpr1,tpr1]
        file.write('false positive, true positive: %s\n' %var)#################save fp,tp
        np.savetxt(parameters['OUTPUT']+'/SIclasses.txt', classes1, delimiter=',')

    def rok(y,w,nclasses,c):
        fpra,tpra,t=roc_curve(y,w)
        xx=np.array([])
        yy=np.array([])
        for i in range(nclasses):
            index = (np.abs(t-c[i])).argmin()
            xx=np.append(fpra[index],xx)
            yy=np.append(tpra[index],yy)
        xx=np.append(np.array([0]),xx)
        yy=np.append(np.array([0]),yy)
        return(xx,yy)