#!/usr/bin/python
#coding=utf-8
"""
/***************************************************************************
    polytogridAlgorithm
        begin                : 2021-11
        copyright            : (C) 2021 by Giacomo Titti,
                               Padova, November 2021
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
    polytogridAlgorithm
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
                       QgsProcessingParameterFileDestination)
import processing
import numpy as np
from osgeo import gdal,osr,ogr
import sys
import math
import csv
from qgis.core import QgsMessageLog,QgsVectorLayer
import os
#from scipy.ndimage import generic_filter
from qgis.core import Qgis
#from processing.algs.gdal.GdalAlgorithm import GdalAlgorithm
from processing.algs.gdal.GdalUtils import GdalUtils


class polytogridAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
    NUM1 = 'w'
    NUM2 = 'h'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Grid'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))

        #self.addParameter(
        #QgsProcessingParameterFeatureSource(
        #    self.INPUT,
    #        self.tr('Input layer'),
#            [QgsProcessing.TypeVectorPoint]))

        #self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT1, self.tr('Raster'), defaultValue=None))

        #self.addParameter(QgsProcessingParameterExtent(self.EXTENT, self.tr('Extension'), defaultValue=None))

        #self.addParameter(QgsProcessingParameterField(self.STRING, 'Field', parentLayerParameterName=self.INPUT, defaultValue=None))

        #self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, 'XYZ grid', '*.txt', defaultValue=None))
        self.addParameter(QgsProcessingParameterNumber(self.NUM1, 'w', type=QgsProcessingParameterNumber.Integer, defaultValue = 10,  minValue=1))
        self.addParameter(QgsProcessingParameterNumber(self.NUM2, 'h', type=QgsProcessingParameterNumber.Integer, defaultValue = 10,  minValue=1))

        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT, self.tr('Output raster'), createByDefault=True, defaultValue=None))

    def processAlgorithm(self, parameters, context, feedback):

        feedback = QgsProcessingMultiStepFeedback(1, feedback)
        results = {}
        outputs = {}

        source= self.parameterAsVectorLayer(parameters, self.INPUT, context)
        parameters['grid']=source.source()
        #parameters['points']=self.asPythonString(parameters,self.INPUT, context)
        #print(parameters['points'])
        if parameters['grid'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        #parameters['poly'] = self.parameterAsExtent(parameters, self.EXTENT, context)
        #if parameters['poly'] is None:
        #    raise QgsProcessingException(self.invalidSourceError(parameters, self.EXTENT))
        parameters['w'] = self.parameterAsInt(parameters, self.NUM1, context)
        if parameters['w'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUM1))

        parameters['h'] = self.parameterAsInt(parameters, self.NUM2, context)
        if parameters['h'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUM2))


        outFile = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        parameters['out'], outputFormat = GdalUtils.ogrConnectionStringAndFormat(outFile, context)
        if parameters['out'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))

        #parameters['out'] = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        #if parameters['out'] is None:
        #    raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))


        # Intersectionpoly
        alg_params = {
            'INPUT_VECTOR_LAYER': parameters['grid'],
            'OUTPUT': parameters['out'],
            'W': parameters['w'],
            'H' : parameters['h']
            }
        #self.extent(alg_params)
        outputs['cleaninventory']=self.importingandcounting(alg_params)
        #del self.raster
        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        return results




    def importingandcounting(self,parameters):
        #QgsMessageLog.logMessage(str(type(parameters['OUTPUT'])), 'MyPlugin', level=Qgis.Info)

        layer = QgsVectorLayer(parameters['INPUT_VECTOR_LAYER'],'grid','ogr')
        limits= layer.extent()
        prj=layer.crs().toWkt()

        xmin=limits.xMinimum()
        xmax=limits.xMaximum()
        ymin=limits.yMinimum()
        ymax=limits.yMaximum()

        number=layer.featureCount()
        pw = parameters['W']#metri
        ph = parameters['H']*-1#metri

        print(xmax-xmin,ymax-ymin)
        xc = np.int((xmax-xmin)/abs(pw))
        yc = np.int((ymax-ymin)/abs(ph))
        print(xc,yc)

        size=np.array([abs(pw),abs(ph)])
        OS=np.array([xmin,ymax])

        self.raster = np.full((yc,xc),-10,dtype='float32')
        #self.raster = rastero.fill(-9999)#.astype('float32')
        cols = xc
        rows = yc
        originX = OS[0]
        originY = OS[1]

        print('write matrix....')
        #np.savetxt(parameters['OUTPUT'], values, delimiter=',', fmt='%d',header='EPSG: '+prj+', PxlW: '+str(pw)+', PxlH: '+str(ph)+', cols: '+str(cols)+', rows: '+str(rows)+', OSx: '+str(OS[0])+', OSy: '+str(OS[1]))
        #np.savetxt(parameters['OUTPUT'], values, delimiter=' ', fmt='%d',newline='\n',header='PxlW '+str(pw)+'\nPxlH '+str(ph)+'\ncols '+str(cols)+'\nrows '+str(rows)+'\nOSx '+str(OS[0])+'\nOSy '+str(OS[1])+'\nnodata -9999',comments='')
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
        return polytogridAlgorithm()

    def name(self):
        return 'PolyToGrid'

    def displayName(self):
        return self.tr('07 PolyToGrid')

    def group(self):
        return self.tr('01 Data preparation')

    def groupId(self):
        return '01 Data preparation'

    def shortHelpString(self):
        return self.tr("PolyToGrid")
