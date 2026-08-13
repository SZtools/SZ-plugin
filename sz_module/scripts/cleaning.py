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
                       QgsProcessingMultiStepFeedback,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterFileDestination,
                       QgsProcessingParameterExtent,
                       QgsProcessingParameterNumber,
                       QgsVectorLayer)
import processing
import numpy as np
from osgeo import gdal,osr,ogr
import sys
import os
from scipy.ndimage import generic_filter
import tempfile

class cleankernelAlgorithm(QgsProcessingAlgorithm):

    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Points'), types=[QgsProcessing.TypeVectorPoint], defaultValue=None))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT1, self.tr('Raster'), defaultValue=None))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, self.tr('Output layer'), defaultValue=None, fileFilter='GeoPackage (*.gpkg *.GPKG)'))
        self.addParameter(QgsProcessingParameterExtent(self.EXTENT, self.tr('Extension'), defaultValue=None))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER, self.tr('Buffer radius in pixels'), type=QgsProcessingParameterNumber.Integer))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER1, self.tr('Min value acceptable'), type=QgsProcessingParameterNumber.Integer))

    def process(self, parameters, context, feedback):
        self.f=tempfile.mkdtemp(prefix='SZ_cleaning_')
        feedback = QgsProcessingMultiStepFeedback(1, feedback)
        results = {}
        outputs = {}
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT1, context)
        if raster_layer is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT1))
        parameters['Slope'] = raster_layer.source()
        source = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))
        parameters['Inventory']=source.source()
        parameters['poly'] = self.parameterAsExtent(parameters, self.EXTENT, context)
        if parameters['poly'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.EXTENT))
        parameters['Extension'] = parameters['poly']
        parameters['BufferRadiousInPxl'] = self.parameterAsInt(parameters, self.NUMBER, context)
        if parameters['BufferRadiousInPxl'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER))
        parameters['minSlopeAcceptable'] = self.parameterAsInt(parameters, self.NUMBER1, context)
        if parameters['minSlopeAcceptable'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER1))
        parameters['out'] = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        if not parameters['out']:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))

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
        outputs['XYcoord']=Functions.vector(alg_params)
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

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}
        results[self.OUTPUT] = outputs['cleaninventory']
        return results
    
class Functions():
    def extent(parameters):
        extent=parameters['INPUT_EXTENT']
        return (
            extent.xMinimum(),
            extent.xMaximum(),
            extent.yMinimum(),
            extent.yMaximum(),
        )

    def importingandcounting(parameters):
        f=parameters['fold']
        raster={}
        ds=gdal.Open(parameters['INPUT_RASTER_LAYER'])
        xc = ds.RasterXSize
        yc = ds.RasterYSize
        geot=ds.GetGeoTransform()
        newXNumPxl=np.round(abs(parameters['xmax']-parameters['xmin'])/(abs(geot[1]))).astype(int)
        newYNumPxl=np.round(abs(parameters['ymax']-parameters['ymin'])/(abs(geot[5]))).astype(int)
        resized_raster = os.path.join(f,'sizedslopexxx.tif')
        translated = gdal.Translate(
            resized_raster,
            ds,
            format='GTiff',
            outputType=gdal.GDT_Float32,
            width=int(newXNumPxl),
            height=int(newYNumPxl),
            projWin=[
                parameters['xmin'],
                parameters['ymax'],
                parameters['xmax'],
                parameters['ymin'],
            ],
            creationOptions=['COMPRESS=DEFLATE', 'PREDICTOR=1', 'ZLEVEL=6'],
        )
        if translated is None:
            raise RuntimeError('gdal.Translate failed while resizing the raster')
        translated = None
        del ds
        ds1=gdal.Open(resized_raster)
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
        XY=None
        for feature in features:
            count +=1
            geom = feature.geometry().asPoint()
            xy=np.array([geom[0],geom[1]])
            XY=xy.reshape(1, -1) if XY is None else np.vstack((XY,xy))
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
        XYcoord=parameters['XYcoord']

        output_path = parameters['OUTPUT']
        driver = ogr.GetDriverByName("GPKG")
        if os.path.exists(output_path):
            driver.DeleteDataSource(output_path)
        ds=driver.CreateDataSource(output_path)
        if ds is None:
            raise QgsProcessingException(f'Unable to create {output_path}')
        srs=osr.SpatialReference(wkt=ds1.GetProjection())
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
        return output_path
