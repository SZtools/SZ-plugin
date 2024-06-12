# -*- coding: utf-8 -*-

"""
/***************************************************************************
    classvAlgorithmW
        begin                : 2021-11
        copyright            : (C) 2021 by Giacomo Titti,
                               Padova, November 2021
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
    classvAlgorithmW
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

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = '$Format:%H$'

from qgis.PyQt.QtCore import QCoreApplication
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
                       QgsProcessingParameterVectorDestination)
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import math
import operator
import matplotlib.pyplot as plt
import os
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
import tempfile
from sz_module.scripts.utils import SZ_utils


class classvAlgorithmW(QgsProcessingAlgorithm):
    # INPUT = 'INPUT'
    # STRING = 'STRING'
    # STRING2 = 'STRING2'
    # STRING3 = 'STRING3'
    # NUMBER = 'classes'
    # OUTPUT1 = 'OUTPUT1'
    # OUTPUT2 = 'OUTPUT2'
    # OUTPUT3 = 'OUTPUT3'

    # def tr(self, string):
    #     return QCoreApplication.translate('Processing', string)

    # def createInstance(self):
    #     return classvAlgorithmW()

    # def name(self):
    #     return 'classy vector wROC'

    # def displayName(self):
    #     return self.tr('02 Classify vector by weighted ROC')

    # def group(self):
    #     return self.tr('04 Classify SI')

    # def groupId(self):
    #     return '04 Classify SI'

    # def shortHelpString(self):
    #     return self.tr("Classifies a index (SI) maximizing the AUC of the relative weighted ROC curve.")

    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Input layer'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING, 'SI field', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING2, 'Field of dependent variable (0 for absence, > 0 for presence)', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER, self.tr('Number of classes'), type=QgsProcessingParameterNumber.Integer, defaultValue = None,  minValue=1))
        self.addParameter(QgsProcessingParameterField(self.STRING3, 'Field of ROC weights', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT3, 'Folder destination', defaultValue=None, createByDefault = True))



    def process(self, parameters, context, model_feedback):
        self.f=tempfile.gettempdir()
        #parameters['classes']=5
        # Use a multi-step feedback, so that individual child algorithm progress reports are adjusted for the
        # overall progress through the model
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

        parameters['w'] = self.parameterAsString(parameters, self.STRING3, context)
        if parameters['w'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING3))

        # parameters['edgesJenks'] = self.parameterAsFileOutput(parameters, self.OUTPUT1, context)
        # if parameters['edgesJenks'] is None:
        #     raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT1))
        #
        # parameters['edgesEqual'] = self.parameterAsFileOutput(parameters, self.OUTPUT2, context)
        # if parameters['edgesEqual'] is None:
        #     raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT2))

        parameters['edgesGA'] = self.parameterAsString(parameters, self.OUTPUT3, context)
        if parameters['edgesGA'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT3))

        parameters['classes'] = self.parameterAsEnum(parameters, self.NUMBER, context)
        if parameters['classes'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER))


        #QgsMessageLog.logMessage(parameters['lsi'], 'MyPlugin', level=Qgis.Info)
        #QgsMessageLog.logMessage(parameters['lsi'], 'MyPlugin', level=Qgis.Info)


        alg_params = {
            #'INPUT_RASTER_LAYER': parameters['Slope'],
            #'INPUT_EXTENT': parameters['Extension'],
            'INPUT_VECTOR_LAYER': parameters['covariates'],
            'field1': parameters['field1'],
            'lsd' : parameters['fieldlsd'],
            'W':parameters['w']
            #'INPUT_INT': parameters['BufferRadiousInPxl'],
            #'INPUT_INT_1': parameters['minSlopeAcceptable'],
        }
        outputs['gdp'],outputs['crs']=SZ_utils.load_cv(alg_params)

        #list_of_values=list(np.arange(10))
        self.list_of_values=outputs['gdp']['SI']
        QgsMessageLog.logMessage(str(len(self.list_of_values)), 'MyPlugin', level=Qgis.Info)

        # alg_params = {
        #     'OUTPUT': parameters['edgesEqual'],
        #     'NUMBER': parameters['classes']
        # }
        #outputs['equal']=self.equal(alg_params)

        alg_params = {
            'df': outputs['gdp'],
            'NUMBER': parameters['classes'],
            'OUTPUT': parameters['edgesGA']
        }
        outputs['ga']=Functions.classy(alg_params)

        # alg_params = {
        #     'OUTPUT': parameters['edgesJenks'],
        #     'NUMBER': parameters['classes']
        # }
        # #outputs['jenk']=self.jenk(alg_params)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}
        return results


class Functions():
    # def load(self,parameters):
    #     layer = QgsVectorLayer(parameters['INPUT_VECTOR_LAYER'], '', 'ogr')
    #     crs=layer.crs()
    #     campi=[]
    #     for field in layer.fields():
    #         campi.append(field.name())
    #     campi.append('geom')
    #     gdp=pd.DataFrame(columns=campi,dtype=float)
    #     df=pd.DataFrame(dtype=float)
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
    #     df['SI']=gdp.loc[:,parameters['field1']]
    #     df['w']=gdp.loc[:,parameters['W']]
    #     nomi=list(df.head())
    #     #print(list(df['Sf']),'1')
    #     lsd=gdp[parameters['lsd']]
    #     lsd[lsd>0]=1
    #     df['y']=lsd#.astype(int)
    #     df['ID']=gdp['ID']
    #     df['geom']=gdp['geom']
    #     df=df.dropna(how='any',axis=0)
    #     #df['ID']=df['ID'].astype('Int32')
    #     return df,crs


    # def raster2array(self,parameters):
    #     self.ds22 = gdal.Open(parameters['INPUT'])
    #     if self.ds22 is None:#####################verify empty row input
    #         #QgsMessageLog.logMessage("ERROR: can't open raster input", tag="WoE")
    #         raise ValueError  # can't open raster input, see 'WoE' Log Messages Panel
    #     self.gt=self.ds22.GetGeoTransform()
    #     self.xsize = self.ds22.RasterXSize
    #     self.ysize = self.ds22.RasterYSize
    #     #print(w,h,xmin,xmax,ymin,ymax,self.xsize,self.ysize)
    #     aa=self.ds22.GetRasterBand(1)
    #     NoData=aa.GetNoDataValue()
    #     matrix = np.array(aa.ReadAsArray())
    #     bands = self.ds22.RasterCount
    #     if bands>1:#####################verify bands
    #         #QgsMessageLog.logMessage("ERROR: input rasters shoud be 1-band raster", tag="WoE")
    #         raise ValueError  # input rasters shoud be 1-band raster, see 'WoE' Log Messages Panel
    #     return matrix

    # def jenk(self,parameters):
    #     breaks = jenkspy.jenks_breaks(self.list_of_values, nb_class=parameters['NUMBER'])
    #     QgsMessageLog.logMessage(str(breaks), 'ClassyLSI', level=Qgis.Info)
    #     np.savetxt(parameters['OUTPUT'], breaks, delimiter=",")
    #
    # def equal(self,parameters):
    #     interval=(np.max(self.list_of_values)-np.min(self.list_of_values))/parameters['NUMBER']
    #     QgsMessageLog.logMessage(str(interval), 'ClassyLSI', level=Qgis.Info)
    #     edges=[]
    #     for i in range(parameters['NUMBER']):
    #         QgsMessageLog.logMessage(str(i), 'ClassyLSI', level=Qgis.Info)
    #         edges=np.append(edges,np.min(self.list_of_values)+(i*interval))
    #     edges=np.append(edges,np.max(self.list_of_values))
    #     np.savetxt(parameters['OUTPUT'], edges, delimiter=",")


    def classy(parameters):

        df=parameters['df']
        y_true=np.array(df['y']).reshape(-1,1)
        scores=np.array(df['SI']).reshape(-1,1)
        y_scores=np.array(df['SI']).reshape(-1,1)#.to_numpy()
        W=np.array(df['w']).reshape(-1,1)
        #print(y_scores,'scores')
        #print(y_true,'true')
        #print(y_scores)
        #print(y_true)
        ################################figure
        fpr1, tpr1, tresh1 = roc_curve(y_true,scores,sample_weight=W)
        #fprv, tprv, treshv = roc_curve(self.y_v,self.scores_v)
        #fprt, tprt, tresht = roc_curve(self.y_t,self.scores_t)
        #aucv=roc_auc_score(self.y_v, self.scores_v, None)
        #auct=roc_auc_score(self.y_t, self.scores_t, None)
        r=roc_auc_score(y_true, scores)
        print('AUC =',r)

        giri=20*parameters['NUMBER']

        numOff=giri#divisibile per 5
        Off=giri
        # l=self.xsize*self.ysize
        # self.matrix=np.reshape(parameters['INPUT1'],-1)
        # self.inventory=np.reshape(parameters['INPUT2'],-1)
        # idx=np.where(self.matrix==-9999.)
        # self.scores = np.delete(self.matrix,idx)
        # self.y_scores=np.delete(self.matrix,idx)
        # self.y_true = np.delete(parameters['INPUT2'],idx)
        # #self.y_v = np.delete(self.validation,idx)
        # #self.y_t = np.delete(self.training,idx)
        nclasses=parameters['NUMBER']
        M=np.max(scores)
        #QgsMessageLog.logMessage(str(M), 'ClassyLSI', level=Qgis.Info)
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
                    #c[pop]=np.hstack((m,m+ran,M+1))############
                    c[pop]=np.hstack((m,m+ran,M))
                    #print(c,'primo')
                    #print ciao
            else:
                c=file
            for k in range(numOff):
                #print weight,'weight'
                weight[k]=y_scores
                for i in range(nclasses):
                    index=np.array([])
                    index=np.where((scores>=c[k][i]) & (scores<c[k][i+1]))
                    weight[k][index]=float(i+1)
                #################################
                #FPR[k],TPR[k]=curve(self,W,y_true,weight[k],nclasses)
                FPR[k],TPR[k]=Functions.rok(W,y_true,scores,nclasses,c[k])
                #######################
                roc_auc[k]=np.trapz(TPR[k],FPR[k])
            #print(roc_auc[k],'area')
            ###############################################
            #print(TPR[k],FPR[k])
            mm=None
            #print(roc_auc)
            mm=max(roc_auc, key=roc_auc.get)
            #print(mm)
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
            #print(count)
            count+=1

            #########################GA
            file={}
            qq=0
            #print(file)
            for q in range(0,numOff,nclasses):
                #print q,'qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq'
                a=np.array([])
                bb={}
                cc=[]
                cc=list(roc_auc.items())
                bb=dict(cc[q:q+nclasses])
                a=sorted(bb.items(), key=operator.itemgetter(1),reverse=True)
                file[q]=c[a[0][0]]
                for b in range(1,nclasses):
                    file[q+b]=np.hstack((file[q][:b],file[q][b-1]+(np.sort(np.random.random_sample(1)*(file[q][b+1]-file[q][b-1]))),file[q][b+1:]))
                # file[q+1]=np.hstack((file[q][:1],file[q][0]+(np.sort(np.random.random_sample(1)*(file[q][2]-file[q][0]))),file[q][2:]))
                # file[q+2]=np.hstack((file[q][:2],file[q][1]+(np.sort(np.random.random_sample(1)*(file[q][3]-file[q][1]))),file[q][3:]))
                # file[q+3]=np.hstack((file[q][:3],file[q][2]+(np.sort(np.random.random_sample(1)*(file[q][4]-file[q][2]))),file[q][4:]))
                # file[q+4]=np.hstack((file[q][:4],file[q][3]+(np.sort(np.random.random_sample(1)*(file[q][5]-file[q][3]))),file[q][5:]))
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
            file = open(parameters['OUTPUT']+'/plotROCW.txt','w')#################save txt
        except:
            os.mkdir(parameters['OUTPUT'])
            file = open(parameters['OUTPUT']+'/plotROCW.txt','w')#################save txt
        #file = open(parameters['OUTPUT']+'/plotROCW.txt','w')#################save txt
        var=[fpr1,tpr1]
        file.write('false positive, true positive: %s\n' %var)#################save fp,tp
        np.savetxt(parameters['OUTPUT']+'/SIclassesW.txt', classes1, delimiter=',')

    # def vector2array(self,parameters):
    #     inn=parameters['INPUT']
    #     w=self.gt[1]
    #     h=self.gt[5]
    #     xmin=self.gt[0]
    #     ymax=self.gt[3]
    #     xmax=xmin+(self.xsize*w)
    #     ymin=ymax+(self.ysize*h)
    #
    #     pxlw=w
    #     pxlh=h
    #     xm=xmin
    #     ym=ymin
    #     xM=xmax
    #     yM=ymax
    #     sizex=self.xsize
    #     sizey=self.ysize
    #
    #     driverd = ogr.GetDriverByName('ESRI Shapefile')
    #     ds9 = driverd.Open(inn)
    #     layer = ds9.GetLayer()
    #     count=0
    #     for feature in layer:
    #         count+=1
    #         geom = feature.GetGeometryRef()
    #         xy=np.array([geom.GetX(),geom.GetY()])
    #         try:
    #             XY=np.vstack((XY,xy))
    #         except:
    #             XY=xy
    #     size=np.array([pxlw,pxlh])
    #     OS=np.array([xm,yM])
    #     NumPxl=(np.ceil(abs((XY-OS)/size)-1))#from 0 first cell
    #     valuess=np.zeros((sizey,sizex),dtype='int16')
    #     try:
    #         for i in range(count):
    #             #print(i,'i')
    #             if XY[i,1]<yM and XY[i,1]>ym and XY[i,0]<xM and XY[i,0]>xm:
    #                 valuess[NumPxl[i,1].astype(int),NumPxl[i,0].astype(int)]=1
    #     except:#only 1 feature
    #         if XY[1]<yM and XY[1]>ym and XY[0]<xM and XY[0]>xm:
    #             valuess[NumPxl[1].astype(int),NumPxl[0].astype(int)]=1
    #     fuori = valuess.astype('float32')
    #     return fuori

def curve(x,y,w,nclasses):
    #x Area
    #y 0 1
    #w lsi
    d={'x':list(x),'y':list(y),'w':list(w)}
    df=pd.DataFrame(data=d)
    #print(w,'w')
    sortdf=df.sort_values(by='w', ascending=False)
    sortdf['ysum']=sortdf['x'].cumsum()
    m1=sum(sortdf['ysum'])
    sortdf1=sortdf.loc[sortdf['y']==1,:]
    #sortdf=sortdf[sortdf['y']==1,]
    sortdf1['xsum']=sortdf1['x'].cumsum()
    m2=sum(sortdf1['xsum'])
    xx=np.array([0])
    yy=np.array([0])
    for i in range(nclasses):
        sortdf1n=sortdf1.loc[sortdf1['w']>=nclasses-i,:]
        xn=sortdf1n['xsum'].sum()
        sortdfn=sortdf.loc[sortdf['w']>=nclasses-i,:]
        yn=sortdfn['ysum'].sum()
        xx=np.append(xx,xn)
        yy=np.append(yy,yn)
    xx=xx/m2
    yy=yy/m1
    #print(x,'x')
    #print(y,'y')
    #print(np.array([0,1,2]))
    #print(ciao)

    return(xx,yy)

def rok(x,y,w,nclasses,c):
    #fpra,tpra,t=roc_curve(y, w, None)
    fpra,tpra,t=roc_curve(y,w,sample_weight=x)
    xx=np.array([])
    yy=np.array([])
    for i in range(nclasses):
        index = (np.abs(t-c[i])).argmin()
        xx=np.append(fpra[index],xx)
        yy=np.append(tpra[index],yy)
    xx=np.append(np.array([0]),xx)
    yy=np.append(np.array([0]),yy)


    # print(x[index])
    # fpr = {}
    # tpr = {}
    # #print(x)
    # #print(x[x==0.])
    # P=float(len(y[y==1.]))#tp+fn
    # N=float(len(y[y==0.]))#tn+fp
    # #print(N)
    # for i in range(nclasses):
    #     index=np.array([])
    #     fptp=np.array([])
    #     index=np.where(w<=i+1)
    #     fptp=y[index]
    #     tp=float(len(np.argwhere(fptp==1)))
    #     fp=float(len(np.argwhere(fptp==0)))
    #     fpr[i]=float(fp/N)
    #     tpr[i]=float(tp/P)
    # xx=np.array([0])
    # yy=np.array([0])
    # for aa in range(nclasses):
    #     xx=np.append(xx,xx[aa]+fpr[nclasses-aa-1])
    #     yy=np.append(yy,yy[aa]+tpr[nclasses-aa-1])
    #print(xx,'x')
    #print(yy,'y')
    return(xx,yy)
