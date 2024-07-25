#!/usr/bin/python
#coding=utf-8
"""
/***************************************************************************
    cleankernelAlgorithm
        begin                : 2021-11
        copyright            : (C) 2021 by Giacomo Titti,
                               Padova, November 2021
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
    cleankernelAlgorithm
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
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingMultiStepFeedback,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterVectorDestination,
                       QgsProcessingParameterExtent,
                       QgsProcessingParameterNumber,
                       QgsVectorLayer)
import processing
import numpy as np
from osgeo import gdal,osr,ogr
import sys
from qgis.core import QgsMessageLog
import os
from scipy.ndimage import generic_filter
from qgis.core import Qgis
from processing.algs.gdal.GdalUtils import GdalUtils
import tempfile


class cleankernelAlgorithm(QgsProcessingAlgorithm):
    # INPUT = 'Inventory'
    # INPUT1 = 'Slope'
    # EXTENT = 'Extension'
    # NUMBER = 'BufferRadiousInPxl'
    # NUMBER1 = 'minSlopeAcceptable'
    # OUTPUT = 'OUTPUT'

    # def tr(self, string):
    #     return QCoreApplication.translate('Processing', string)

    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Points'), types=[QgsProcessing.TypeVectorPoint], defaultValue=None))

        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT1, self.tr('Raster'), defaultValue=None))

        self.addParameter(QgsProcessingParameterVectorDestination(self.OUTPUT, self.tr('Output layer'), type=QgsProcessing.TypeVectorPoint, createByDefault=True, defaultValue=None))

        self.addParameter(QgsProcessingParameterExtent(self.EXTENT, self.tr('Extension'), defaultValue=None))

        self.addParameter(QgsProcessingParameterNumber(self.NUMBER, self.tr('Buffer radius in pixels'), type=QgsProcessingParameterNumber.Integer))

        self.addParameter(QgsProcessingParameterNumber(self.NUMBER1, self.tr('Min value acceptable'), type=QgsProcessingParameterNumber.Integer))

    def process(self, parameters, context, feedback):
        self.f=tempfile.gettempdir()

        feedback = QgsProcessingMultiStepFeedback(1, feedback)
        results = {}
        outputs = {}

        parameters['Slope'] = self.parameterAsRasterLayer(parameters, self.INPUT1, context).source()
        if parameters['Slope'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT1))

        source = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        parameters['Inventory']=source.source()
        if parameters['Inventory'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        parameters['poly'] = self.parameterAsExtent(parameters, self.EXTENT, context)
        if parameters['poly'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.EXTENT))

        parameters['BufferRadiousInPxl'] = self.parameterAsInt(parameters, self.NUMBER, context)
        if parameters['BufferRadiousInPxl'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER))

        parameters['minSlopeAcceptable'] = self.parameterAsInt(parameters, self.NUMBER1, context)
        if parameters['minSlopeAcceptable'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER1))

        outFile = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        parameters['out'], outputFormat = GdalUtils.ogrConnectionStringAndFormat(outFile, context)

        if parameters['out'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))

        # Intersectionpoly
        alg_params = {
            'INPUT_RASTER_LAYER': parameters['Slope'],
            'INPUT_EXTENT': parameters['Extension'],
            'INPUT_VECTOR_LAYER': parameters['Inventory'],
            'INPUT_INT': parameters['BufferRadiousInPxl'],
            'INPUT_INT_1': parameters['minSlopeAcceptable'],
            'OUTPUT': parameters['out']
        }
        outputs['xmin'],outputs['xmax'],outputs['ymin'],outputs['ymax']=Functions.extent(alg_params)

        alg_params = {
            'INPUT_RASTER_LAYER': parameters['Slope'],
            'INPUT_EXTENT': parameters['Extension'],
            'INPUT_VECTOR_LAYER': parameters['Inventory'],
            'INPUT_INT': parameters['BufferRadiousInPxl'],
            'INPUT_INT_1': parameters['minSlopeAcceptable'],
            'OUTPUT': parameters['out'],
            'xmin':outputs['xmin'],
            'xmax':outputs['xmax'],
            'ymin':outputs['ymin'],
            'ymax':outputs['ymax'],
            'fold':self.f

        }
        outputs['raster'],outputs['ds1'],outputs['XY']=Functions.importingandcounting(alg_params)

        alg_params = {
            'INPUT_RASTER_LAYER': parameters['Slope'],
            'INPUT_EXTENT': parameters['Extension'],
            'INPUT_VECTOR_LAYER': parameters['Inventory'],
            'INPUT_INT': parameters['BufferRadiousInPxl'],
            'INPUT_INT_1': parameters['minSlopeAcceptable'],
            'OUTPUT': parameters['out'],
            'xmin':outputs['xmin'],
            'xmax':outputs['xmax'],
            'ymin':outputs['ymin'],
            'ymax':outputs['ymax'],
            'raster':outputs['raster'],
            'fold':self.f,
            'ds1':outputs['ds1'],
            'XY':outputs['XY']
        }
        outputs['oout']=Functions.indexing(alg_params)

        alg_params = {
            'INPUT_RASTER_LAYER': parameters['Slope'],
            'INPUT_EXTENT': parameters['Extension'],
            'INPUT_VECTOR_LAYER': parameters['Inventory'],
            'INPUT_INT': parameters['BufferRadiousInPxl'],
            'INPUT_INT_1': parameters['minSlopeAcceptable'],
            'OUTPUT': parameters['out'],
            'xmin':outputs['xmin'],
            'xmax':outputs['xmax'],
            'ymin':outputs['ymin'],
            'ymax':outputs['ymax'],
            'raster':outputs['raster'],
            'fold':self.f,
            'ds1':outputs['ds1'],
            'XY':outputs['XY'],
            'oout':outputs['oout']
        }
        outputs['XYcoord']=Functions.vector()
        del alg_params['oout']

        alg_params = {
            'INPUT_RASTER_LAYER': parameters['Slope'],
            'INPUT_EXTENT': parameters['Extension'],
            'INPUT_VECTOR_LAYER': parameters['Inventory'],
            'INPUT_INT': parameters['BufferRadiousInPxl'],
            'INPUT_INT_1': parameters['minSlopeAcceptable'],
            'OUTPUT': parameters['out'],
            'xmin':outputs['xmin'],
            'xmax':outputs['xmax'],
            'ymin':outputs['ymin'],
            'ymax':outputs['ymax'],
            'raster':outputs['raster'],
            'fold':self.f,
            'ds1':outputs['ds1'],
            'XY':outputs['XY'],
            'oout':outputs['oout'],
            'XYcoord':outputs['XYcoord']
        }
        outputs['cleaninventory']=Functions.saveV(alg_params)
        del self.raster

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        return results
class Functions():
    def extent(parameters):
        limits=np.fromstring(parameters['INPUT_EXTENT'], dtype=float, sep=',')
        xmin=limits[0]
        xmax=limits[1]
        ymin=limits[2]
        ymax=limits[3]
        return xmin,xmax,ymin,ymax

    def importingandcounting(parameters):
        f=parameters['fold']
        raster={}
        ds=gdal.Open(parameters['INPUT_RASTER_LAYER'])
        xc = ds.RasterXSize
        yc = ds.RasterYSize
        geot=ds.GetGeoTransform()
        newXNumPxl=np.round(abs(parameters['xmax']-parameters['xmin'])/(abs(geot[1]))).astype(int)
        newYNumPxl=np.round(abs(parameters['ymax']-parameters['ymin'])/(abs(geot[5]))).astype(int)
        try:
            os.system('gdal_translate -of GTiff -ot Float32 -strict -outsize ' + str(newXNumPxl) +' '+ str(newYNumPxl) +' -projwin ' +str(parameters['xmin'])+' '+str(parameters['ymax'])+' '+ str(parameters['xmax']) + ' ' + str(parameters['ymin']) +' -co COMPRESS=DEFLATE -co PREDICTOR=1 -co ZLEVEL=6 ' + parameters['INPUT_RASTER_LAYER'] +' '+ f+'/sizedslopexxx.tif')
        except:
            raise ValueError  # Failure to save sized cause, see 'WoE' Log Messages Panel
        del ds
        ds1=gdal.Open(f+'/sizedslopexxx.tif')
        if ds1 is None:
            print("ERROR: can't open raster input")
        nodata=ds1.GetRasterBand(1).GetNoDataValue()
        raster[0] = np.array(ds1.GetRasterBand(1).ReadAsArray())
        raster[0][raster[0]==nodata]=-9999
        x = ds1.RasterXSize
        y = ds1.RasterYSize

        gtdem= ds1.GetGeoTransform()
        size=np.array([abs(gtdem[1]),abs(gtdem[5])])
        OS=np.array([gtdem[0],gtdem[3]])
        xmin=OS[0]
        xmax=OS[0]+(size[0]*x)
        ymax=OS[1]
        ymin=OS[1]-(size[1]*y)
        layer=QgsVectorLayer(parameters['INPUT_VECTOR_LAYER'], '', 'ogr')
        features=layer.getFeatures()
        count=0
        for feature in features:
            count +=1
            geom = feature.geometry().asPoint()
            xy=np.array([geom[0],geom[1]])
            try:
                XY=np.vstack((XY,xy))
            except:
                XY=xy
        gtdem= ds1.GetGeoTransform()
        size=np.array([abs(gtdem[1]),abs(gtdem[5])])
        OS=np.array([gtdem[0],gtdem[3]])
        NumPxl=(np.ceil((abs(XY-OS)/size)-1))#from 0 first cell
        values=np.zeros((y,x), dtype=int)
        for i in range(count):
            if XY[i,1]<=ymax and XY[i,1]>=ymin and XY[i,0]<=xmax and XY[i,0]>=xmin:
                values[NumPxl[i,1].astype(int),NumPxl[i,0].astype(int)]=1
        raster[1] = values.astype('float32')
        return raster,ds1,XY

    def indexing(parameters):
        raster=parameters['raster']
        ggg=np.array([])
        ggg=raster[0].astype('float32')
        R=np.array([])
        R=raster[1].astype(np.int16)
        numbb=parameters['INPUT_INT']*2+1
        g = generic_filter(ggg, np.nanmax, size=(numbb,numbb))
        oout=np.array([])
        oout=R*g
        oout[(raster[0]==-9999)]=-9999
        oout[(raster[1]==0)]=-9999
        oout[(oout<parameters['INPUT_INT_1'])]=-9999
        oout[oout>=parameters['INPUT_INT_1']]=1
        g=None
        del ggg
        return oout

    def vector(parameters):
        oout=parameters['oout']
        ds1=parameters['ds1']
        XY=parameters['XY']
        row,col=np.where(oout==1)
        geo=ds1.GetGeoTransform()
        xsize=geo[1]
        ysize=geo[5]
        OOx=geo[0]
        OOy=geo[3]
        XYcoord=np.array([0,0])
        for i in range(len(col)):
            xmin=OOx+(xsize*col[i])
            xmax=OOx+(xsize*col[i])+(xsize)
            ymax=OOy+(ysize*row[i])
            ymin=OOy+(ysize*row[i])+(ysize)
            for ii in range(len(XY)):
                if (XY[ii,0]>=xmin and XY[ii,0]<=xmax and XY[ii,1]>=ymin and XY[ii,1]<=ymax):
                    XYcoord=np.vstack((XYcoord,XY[ii,:]))
        XYcoord=XYcoord[:]
        return XYcoord

    def saveV(parameters):
        ds1=parameters['ds1']
        XYcoord=parameters['XY_coord']

        driver = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(parameters['OUTPUT']):
            driver.DeleteDataSource(parameters['OUTPUT'])
        ds=driver.CreateDataSource(parameters['OUTPUT'])

        srs=osr.SpatialReference(wkt = s1.GetProjection())
        layer = ds.CreateLayer("inventory_cleaned", srs, ogr.wkbPoint)
        field_name = ogr.FieldDefn("id", ogr.OFTInteger)
        field_name.SetWidth(100)
        layer.CreateField(field_name)
        for i in range(len(XYcoord)):
            # create the feature
            feature = ogr.Feature(layer.GetLayerDefn())
            # Set the attributes using the values from the delimited text file
            feature.SetField("id", i)
            # create the WKT for the feature using Python string formatting
            wkt = "POINT(%f %f)" % (float(XYcoord[i,0]) , float(XYcoord[i,1]))
            # Create the point from the Well Known Txt
            point = ogr.CreateGeometryFromWkt(wkt)
            # Set the feature geometry using the point
            feature.SetGeometry(point)
            # Create the feature in the layer (shapefile)
            layer.CreateFeature(feature)
            # Dereference the feature
            feature = None
        # Save and close the data source
        ds = None
        return parameters['OUTPUT']

    # def createInstance(self):
    #     return cleankernelAlgorithm()

    # def name(self):
    #     return 'clean points'

    # def displayName(self):
    #     return self.tr('01 Clean Points By Raster Kernel Value')

    # def group(self):
    #     return self.tr('01 Data preparation')

    # def groupId(self):
    #     return '01 Data preparation'

    # def shortHelpString(self):
    #     return self.tr("It selects and remove features from point vector by a kernel raster condition")
