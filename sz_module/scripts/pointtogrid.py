#!/usr/bin/python
#coding=utf-8
"""
/***************************************************************************
    pointtogridAlgorithm
        begin                : 2021-11
        copyright            : (C) 2021 by Giacomo Titti,
                               Padova, November 2021
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
    pointtogridAlgorithm
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

from PyQt5.QtCore import QCoreApplication
from qgis.core import (QgsProcessing,
                       QgsFeatureSink,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingMultiStepFeedback,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterRasterDestination,
                       QgsProcessingParameterExtent,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterField,
                       QgsProcessingParameterFileDestination,
                       QgsVectorLayer)
import processing
import numpy as np
from osgeo import gdal,osr,ogr
import sys
import math
import csv
from qgis.core import QgsMessageLog
import os
#from scipy.ndimage import generic_filter
from qgis.core import Qgis
#from processing.algs.gdal.GdalAlgorithm import GdalAlgorithm
from processing.algs.gdal.GdalUtils import GdalUtils


class pointtogridAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'points'
    INPUT1 = 'grid'
    EXTENT = 'Extension'
    OUTPUT = 'OUTPUT'
    STRING = 'STRING'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Points'), types=[QgsProcessing.TypeVectorPoint], defaultValue=None))

        #self.addParameter(
        #QgsProcessingParameterFeatureSource(
        #    self.INPUT,
    #        self.tr('Input layer'),
#            [QgsProcessing.TypeVectorPoint]))

        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT1, self.tr('Raster'), defaultValue=None))

        self.addParameter(QgsProcessingParameterExtent(self.EXTENT, self.tr('Extension'), defaultValue=None))

        #self.addParameter(QgsProcessingParameterField(self.STRING, 'Field', parentLayerParameterName=self.INPUT, defaultValue=None))

        #self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, 'XYZ grid', '*.txt', defaultValue=None))

        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT, self.tr('Output raster'), createByDefault=True, defaultValue=None))

    def processAlgorithm(self, parameters, context, feedback):

        feedback = QgsProcessingMultiStepFeedback(1, feedback)
        results = {}
        outputs = {}

        parameters['grid'] = self.parameterAsRasterLayer(parameters, self.INPUT1, context).source()
        if parameters['grid'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT1))

        source= self.parameterAsVectorLayer(parameters, self.INPUT, context)
        parameters['points']=source.source()
        #parameters['points']=self.asPythonString(parameters,self.INPUT, context)
        #print(parameters['points'])
        if parameters['points'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        parameters['poly'] = self.parameterAsExtent(parameters, self.EXTENT, context)
        if parameters['poly'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.EXTENT))

        # parameters['field'] = self.parameterAsString(parameters, self.STRING, context)
        # if parameters['field'] is None:
        #     raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING))

        outFile = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        parameters['out'], outputFormat = GdalUtils.ogrConnectionStringAndFormat(outFile, context)
        if parameters['out'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))

        # parameters['out'] = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        # if parameters['out'] is None:
        #     raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))


        # Intersectionpoly
        alg_params = {
            'STRING':'',#parameters['field'],
            'INPUT_RASTER_LAYER': parameters['grid'],
            'INPUT_EXTENT': parameters['poly'],
            'INPUT_VECTOR_LAYER': parameters['points'],
            'OUTPUT': parameters['out']
            }
        self.extent(alg_params)
        outputs['cleaninventory']=self.importingandcounting(alg_params)
        #del self.raster
        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        return results

    def extent(self,parameters):
        #print(parameters['INPUT_EXTENT'],'ext')
        #limits=np.fromstring(parameters['INPUT_EXTENT'], dtype=float, sep=',')
        limits=parameters['INPUT_EXTENT']
        self.xmin=limits.xMinimum()
        #print(self.xmin)
        self.xmax=limits.xMaximum()
        self.ymin=limits.yMinimum()
        self.ymax=limits.yMaximum()

    def importingandcounting(self,parameters):
        #QgsMessageLog.logMessage(str(type(parameters['OUTPUT'])), 'MyPlugin', level=Qgis.Info)
        #ciao

        ds=gdal.Open(parameters['INPUT_RASTER_LAYER'])
        prj=ds.GetProjection()
        xc = ds.RasterXSize
        yc = ds.RasterYSize
        geot=ds.GetGeoTransform()
        #print(geot)
        pw = geot[1]
        ph = geot[5]

        size=np.array([abs(geot[1]),abs(geot[5])])
        OS=np.array([geot[0],geot[3]])
        print('start reading vector...')
        layer=QgsVectorLayer(parameters['INPUT_VECTOR_LAYER'], '', 'ogr')
        features=layer.getFeatures()


        count=0
        for feature in features:
            count +=1
            geom = feature.geometry().asPoint()
            #print(geom)
            #print(geom.asMultiPoint())
            xyz=np.array([geom[0],geom[0],0])#,feature.attribute(parameters['STRING']))
        # driverd = ogr.GetDriverByName('ESRI Shapefile')
        # print(parameters['INPUT_VECTOR_LAYER'])
        # ds9 = ogr.Open(parameters['INPUT_VECTOR_LAYER'])
        # layer = ds9.GetLayer()
        # count=0

        #
        # for feature in layer:
        #     print(count)
        #     count +=1
        #     geom = feature.GetGeometryRef()
        #     xyz=np.array([geom.GetX(),geom.GetY(),feature.GetField(parameters['STRING'])])
            try:
                self.XYZ=np.vstack((self.XYZ,xyz))
            except:
                self.XYZ=xyz

        print(self.XYZ[:,:-1],'self')
        NumPxl=(np.ceil((abs(self.XYZ[:,:-1]-OS)/size)-1))#from 0 first cell
        values=np.full((yc,xc), -9999, dtype='float32')
        del ds
        #del ds9

        print('start matrix...')
        #try:
        for i in range(count):
            #print(i,'i')
            if self.XYZ[i,1]<self.ymax and self.XYZ[i,1]>self.ymin and self.XYZ[i,0]<self.xmax and self.XYZ[i,0]>self.xmin:
                if values[NumPxl[i,1].astype(int),NumPxl[i,0].astype(int)]>0:
                    values[NumPxl[i,1].astype(int),NumPxl[i,0].astype(int)]+=1
                else:
                    values[NumPxl[i,1].astype(int),NumPxl[i,0].astype(int)]=1#self.XYZ[i,2]######posso anche scegliere un campo da darlgi
        #print(values)
        #except:#only 1 feature
            #if self.XYZ[1]<self.ymax and self.XYZ[1]>self.ymin and self.XYZ[0]<self.xmax and self.XYZ[0]>self.xmin:
            #    values[NumPxl[1].astype(int),NumPxl[0].astype(int)]=self.XYZ[i,2]

        self.raster = values#.astype('float32')
        cols = self.raster.shape[1]
        rows = self.raster.shape[0]
        originX = OS[0]
        originY = OS[1]

        print('write matrix....')
        #np.savetxt(parameters['OUTPUT'], values, delimiter=',', fmt='%d',header='EPSG: '+prj+', PxlW: '+str(pw)+', PxlH: '+str(ph)+', cols: '+str(cols)+', rows: '+str(rows)+', OSx: '+str(OS[0])+', OSy: '+str(OS[1]))

        #np.savetxt(parameters['OUTPUT'], values, delimiter=' ', fmt='%d',newline='\n',header='PxlW '+str(pw)+'\nPxlH '+str(ph)+'\ncols '+str(cols)+'\nrows '+str(rows)+'\nOSx '+str(OS[0])+'\nOSy '+str(OS[1])+'\nnodata -9999',comments='')

        del values

        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.Create(parameters['OUTPUT'], cols, rows, 1, gdal.GDT_Float32)
        outRaster.SetGeoTransform((originX, pw, 0, originY, 0, ph))
        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(self.raster)
        outband.SetNoDataValue(-9999.)
        outRasterSRS = osr.SpatialReference()
        #outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(prj)
        outband.FlushCache()
        return parameters['OUTPUT']

    def createInstance(self):
        return pointtogridAlgorithm()

    def name(self):
        return 'PointsToGrid'

    def displayName(self):
        return self.tr('06 PointsToGrid')

    def group(self):
        return self.tr('01 Data preparation')

    def groupId(self):
        return '01 Data preparation'

    def shortHelpString(self):
        return self.tr("PointsToGrid")
