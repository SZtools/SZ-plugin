# -*- coding: utf-8 -*-

"""
/***************************************************************************
    rocAlgorithm
        begin                : 2021-11
        copyright            : (C) 2021 by Giacomo Titti,
                               Padova, November 2021
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
    rocAlgorithm
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

__author__ = 'Giacomo Titti'
__date__ = '2021-07-01'
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
                       QgsProcessingParameterVectorLayer)
from qgis import processing
#import jenkspy
from osgeo import gdal,ogr
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import math
import operator
import matplotlib.pyplot as plt
import tempfile

class rocAlgorithm(QgsProcessingAlgorithm):
    INPUT1 = 'lsi'
    INPUT2 = 'lsd'
    NUMBER = 'classes'
    OUTPUT1 = 'OUTPUT1'
    OUTPUT2 = 'OUTPUT2'
    OUTPUT3 = 'OUTPUT3'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return rocAlgorithm()

    def name(self):
        return 'classy raster'

    def displayName(self):
        return self.tr('02 Classify Raster')

    def group(self):
        return self.tr('Raster analysis')

    def groupId(self):
        return 'Raster analysis'

    def shortHelpString(self):
        return self.tr("Apply different kind of classificator to raster: Jenks Natural Breaks, Equal Interval")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT1, self.tr('LSI'), defaultValue=None))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT1, 'edgesJenks', '*.txt', defaultValue=None))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT2, 'edgesEqual', '*.txt', defaultValue=None))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT3, 'edgesGA', '*.txt', defaultValue=None))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER, self.tr('number of classes'), type=QgsProcessingParameterNumber.Integer, defaultValue = None,  minValue=0))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT2, self.tr('Landslides'), types=[QgsProcessing.TypeVectorPoint], defaultValue=None))

    def processAlgorithm(self, parameters, context, model_feedback):
        self.f=tempfile.gettempdir()
        #parameters['classes']=5
        # Use a multi-step feedback, so that individual child algorithm progress reports are adjusted for the
        # overall progress through the model
        feedback = QgsProcessingMultiStepFeedback(1, model_feedback)
        results = {}
        outputs = {}

        parameters['lsi'] = self.parameterAsRasterLayer(parameters, self.INPUT1, context).source()
        if parameters['lsi'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT1))

        source= self.parameterAsVectorLayer(parameters, self.INPUT2, context)
        parameters['lsd']=source.source()
        if parameters['lsd'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT2))

        parameters['edgesJenks'] = self.parameterAsFileOutput(parameters, self.OUTPUT1, context)
        if parameters['edgesJenks'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT1))

        parameters['edgesEqual'] = self.parameterAsFileOutput(parameters, self.OUTPUT2, context)
        if parameters['edgesEqual'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT2))

        parameters['edgesGA'] = self.parameterAsFileOutput(parameters, self.OUTPUT3, context)
        if parameters['edgesGA'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT3))

        parameters['classes'] = self.parameterAsEnum(parameters, self.NUMBER, context)
        if parameters['classes'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER))


        #QgsMessageLog.logMessage(parameters['lsi'], 'MyPlugin', level=Qgis.Info)
        #QgsMessageLog.logMessage(parameters['lsi'], 'MyPlugin', level=Qgis.Info)
        # Input
        alg_params = {
            'INPUT': parameters['lsi']
        }
        outputs['open']=self.raster2array(alg_params)
        #list_of_values=list(np.arange(10))
        self.list_of_values=outputs['open'][outputs['open']>-9999].reshape(-1)
        QgsMessageLog.logMessage(str(len(self.list_of_values)), 'MyPlugin', level=Qgis.Info)

        alg_params = {
            'OUTPUT': parameters['edgesEqual'],
            'NUMBER': parameters['classes']
        }
        #outputs['equal']=self.equal(alg_params)

        alg_params = {
            'INPUT': parameters['lsd']
        }
        b=self.vector2array(alg_params)
        outputs['inv'] = b.astype(int)

        alg_params = {
            'INPUT1': outputs['open'],
            'INPUT2': outputs['inv'],
            'NUMBER': parameters['classes'],
            'OUTPUT': parameters['edgesGA']
        }
        outputs['ga']=self.classy(alg_params)

        alg_params = {
            'OUTPUT': parameters['edgesJenks'],
            'NUMBER': parameters['classes']
        }
        #outputs['jenk']=self.jenk(alg_params)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}
        return results

    def raster2array(self,parameters):
        self.ds22 = gdal.Open(parameters['INPUT'])
        if self.ds22 is None:#####################verify empty row input
            #QgsMessageLog.logMessage("ERROR: can't open raster input", tag="WoE")
            raise ValueError  # can't open raster input, see 'WoE' Log Messages Panel
        self.gt=self.ds22.GetGeoTransform()
        self.xsize = self.ds22.RasterXSize
        self.ysize = self.ds22.RasterYSize
        #print(w,h,xmin,xmax,ymin,ymax,self.xsize,self.ysize)
        aa=self.ds22.GetRasterBand(1)
        NoData=aa.GetNoDataValue()
        matrix = np.array(aa.ReadAsArray())
        bands = self.ds22.RasterCount
        if bands>1:#####################verify bands
            #QgsMessageLog.logMessage("ERROR: input rasters shoud be 1-band raster", tag="WoE")
            raise ValueError  # input rasters shoud be 1-band raster, see 'WoE' Log Messages Panel
        return matrix

    def jenk(self,parameters):
        breaks = jenkspy.jenks_breaks(self.list_of_values, nb_class=parameters['NUMBER'])
        QgsMessageLog.logMessage(str(breaks), 'ClassyLSI', level=Qgis.Info)
        np.savetxt(parameters['OUTPUT'], breaks, delimiter=",")

    def equal(self,parameters):
        interval=(np.max(self.list_of_values)-np.min(self.list_of_values))/parameters['NUMBER']
        QgsMessageLog.logMessage(str(interval), 'ClassyLSI', level=Qgis.Info)
        edges=[]
        for i in range(parameters['NUMBER']):
            QgsMessageLog.logMessage(str(i), 'ClassyLSI', level=Qgis.Info)
            edges=np.append(edges,np.min(self.list_of_values)+(i*interval))
        edges=np.append(edges,np.max(self.list_of_values))
        np.savetxt(parameters['OUTPUT'], edges, delimiter=",")


    def classy(self,parameters):
        self.numOff=100#divisibile per 5
        self.Off=100
        l=self.xsize*self.ysize
        self.matrix=np.reshape(parameters['INPUT1'],-1)
        self.inventory=np.reshape(parameters['INPUT2'],-1)
        idx=np.where(self.matrix==-9999.)
        self.scores = np.delete(self.matrix,idx)
        self.y_scores=np.delete(self.matrix,idx)
        self.y_true = np.delete(parameters['INPUT2'],idx)
        #self.y_v = np.delete(self.validation,idx)
        #self.y_t = np.delete(self.training,idx)
        nclasses=parameters['NUMBER']
        M=np.max(self.scores)
        #QgsMessageLog.logMessage(str(M), 'ClassyLSI', level=Qgis.Info)
        m=np.min(self.scores)
        count=0
        ran=np.array([])
        fitness=0
        values=np.array([])
        classes=([])
        c={}
        ran=np.array([])
        summ=0
        while count<self.Off:
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
                for pop in range(self.numOff):
                    ran=np.sort(np.random.random_sample(nclasses-1)*(M-m))
                    c[pop]=np.hstack((m,m+ran,M+1))############
                    #c[pop]=np.hstack((m,m+ran,M))
                    #print c
                    #print ciao
            else:
                c=file
            for k in range(self.numOff):
                #print weight,'weight'
                weight[k]=self.y_scores
                for i in range(nclasses):
                    index=np.array([])
                    index=np.where((self.scores>=c[k][i]) & (self.scores<c[k][i+1]))
                    weight[k][index]=float(i+1)
                #roc_auc[k]=roc_auc_score(self.y_true, weight[k], None)
                ####################################
                fpr = {}
                tpr = {}
                P=float(len(np.argwhere(self.y_true==1)))#tp+fn
                N=float(len(np.argwhere(self.y_true==0)))#tn+fp
                for i in range(nclasses):
                    index=np.array([])
                    fptp=np.array([])
                    index=np.where(weight[k]==i+1)
                    fptp=self.y_true[index]
                    tp=float(len(np.argwhere(fptp==1)))
                    fp=float(len(np.argwhere(fptp==0)))
                    #print tp,':tp',fp,':fp'
                    #print P,':P',N,':N'
                    fpr[i]=float(fp/N)
                    tpr[i]=float(tp/P)
                    #FPR[i]=fpr[i]
                    #TPR[i]=tpr[i]
                FPR[k]=np.array([0,fpr[4],fpr[4]+fpr[3],fpr[4]+fpr[3]+fpr[2],fpr[4]+fpr[3]+fpr[2]+fpr[1],fpr[4]+fpr[3]+fpr[2]+fpr[1]+fpr[0]])
                #TPR=np.array([tpr[0],tpr[0]+tpr[1],tpr[0]+tpr[1]+tpr[2],tpr[0]+tpr[1]+tpr[2]+tpr[3],tpr[0]+tpr[1]+tpr[2]+tpr[3]+tpr[4]])
                TPR[k]=np.array([0,tpr[4],tpr[4]+tpr[3],tpr[4]+tpr[3]+tpr[2],tpr[4]+tpr[3]+tpr[2]+tpr[1],tpr[4]+tpr[3]+tpr[2]+tpr[1]+tpr[0]])
                roc_auc[k]=np.trapz(TPR[k],FPR[k])
            ###############################################
            mm=None
            mm=max(roc_auc, key=roc_auc.get)
            #print fitness
            if roc_auc[mm]>fitness:#############################fitness
                fitness=None
                classes=np.array([])
                values=np.array([])
                ttpr=np.array([])
                ffpr=np.array([])
                fitness=roc_auc[mm]
                #print(fitness)
                classes=c[mm]
                values=weight[mm]
                #print(classes)
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
            for q in range(0,self.numOff,5):
                #print q,'qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq'
                a=np.array([])
                bb={}
                cc=[]
                cc=list(roc_auc.items())
                bb=dict(cc[q:q+5])
                a=sorted(bb.items(), key=operator.itemgetter(1),reverse=True)
                file[q]=c[a[0][0]]
                file[q+1]=np.hstack((file[q][0],file[q][0]+(np.sort(np.random.random_sample(1)*(file[q][2]-file[q][0]))),file[q][2:]))
                file[q+2]=np.hstack((file[q][:2],file[q][1]+(np.sort(np.random.random_sample(1)*(file[q][3]-file[q][1]))),file[q][3:]))
                file[q+3]=np.hstack((file[q][:3],file[q][2]+(np.sort(np.random.random_sample(1)*(file[q][4]-file[q][2]))),file[q][4:]))
                file[q+4]=np.hstack((file[q][:4],file[q][3]+(np.sort(np.random.random_sample(1)*(file[q][5]-file[q][3]))),file[q][5:]))
                qq+=5
        self.fitness=None
        self.tpr=np.array([])
        self.fpr=np.array([])
        self.values=np.array([])
        self.classes=np.array([])
        self.fitness=fitness
        self.values=values
        self.classes=classes
        self.tpr=ttpr
        self.fpr=ffpr
        file = open(self.f+'/plotROC.txt','w')#################save txt
        var=[self.fpr,self.tpr]
        file.write('false positive, true positive: %s\n' %var)#################save fp,tp
        np.savetxt(parameters['OUTPUT'], self.classes, delimiter=',')

    def vector2array(self,parameters):
        inn=parameters['INPUT']
        w=self.gt[1]
        h=self.gt[5]
        xmin=self.gt[0]
        ymax=self.gt[3]
        xmax=xmin+(self.xsize*w)
        ymin=ymax+(self.ysize*h)

        pxlw=w
        pxlh=h
        xm=xmin
        ym=ymin
        xM=xmax
        yM=ymax
        sizex=self.xsize
        sizey=self.ysize

        driverd = ogr.GetDriverByName('ESRI Shapefile')
        ds9 = driverd.Open(inn)
        layer = ds9.GetLayer()
        count=0
        for feature in layer:
            count+=1
            geom = feature.GetGeometryRef()
            xy=np.array([geom.GetX(),geom.GetY()])
            try:
                XY=np.vstack((XY,xy))
            except:
                XY=xy
        size=np.array([pxlw,pxlh])
        OS=np.array([xm,yM])
        NumPxl=(np.ceil(abs((XY-OS)/size)-1))#from 0 first cell
        valuess=np.zeros((sizey,sizex),dtype='int16')
        try:
            for i in range(count):
                #print(i,'i')
                if XY[i,1]<yM and XY[i,1]>ym and XY[i,0]<xM and XY[i,0]>xm:
                    valuess[NumPxl[i,1].astype(int),NumPxl[i,0].astype(int)]=1
        except:#only 1 feature
            if XY[1]<yM and XY[1]>ym and XY[0]<xM and XY[0]>xm:
                valuess[NumPxl[1].astype(int),NumPxl[0].astype(int)]=1
        fuori = valuess.astype('float32')
        return fuori
