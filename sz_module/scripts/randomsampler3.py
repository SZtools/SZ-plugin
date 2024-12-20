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
                       QgsProcessingMultiStepFeedback,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFileDestination,
                       QgsProcessingParameterVectorLayer,
                       QgsVectorLayer,
                       QgsRasterLayer,
                       QgsProject,
                       QgsProcessingContext
                       )
from qgis import processing
import numpy as np
import random
from qgis import *
import tempfile
import os
from osgeo import gdal,ogr


class samplerAlgorithm(QgsProcessingAlgorithm):

    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Points'), types=[QgsProcessing.TypeVectorPoint], defaultValue=None))
        self.addParameter(QgsProcessingParameterVectorLayer(self.MASK, self.tr('Contour polygon'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER, 'Pixel width', type=QgsProcessingParameterNumber.Integer, defaultValue = 0,  minValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER1, 'Pixel height', type=QgsProcessingParameterNumber.Integer, defaultValue = 0,  minValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER2, 'Sample (%)', type=QgsProcessingParameterNumber.Integer, defaultValue = 0,  minValue=0))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT1, 'Layer of sample', defaultValue=None, fileFilter='ESRI Shapefile (*.shp *.SHP)'))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT2, 'Layer of 1-sample',  defaultValue=None, fileFilter='ESRI Shapefile (*.shp *.SHP)'))

    def process(self, parameters, context, model_feedback):
        self.f=tempfile.gettempdir()
        feedback = QgsProcessingMultiStepFeedback(1, model_feedback)
        results = {}
        outputs = {}
        parameters['lsd'] = self.parameterAsVectorLayer(parameters, self.INPUT, context).source()
        if parameters['lsd'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))
        parameters['poly'] = self.parameterAsVectorLayer(parameters, self.MASK, context).source()
        if parameters['poly'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.MASK))
        parameters['w'] = self.parameterAsInt(parameters, self.NUMBER, context)
        if parameters['w'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER))
        parameters['h'] = self.parameterAsInt(parameters, self.NUMBER1, context)
        if parameters['h'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER1))
        parameters['train'] = self.parameterAsInt(parameters, self.NUMBER2, context)
        if parameters['train'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER2))
        parameters['vout'] = self.parameterAsFileOutput(parameters, self.OUTPUT1, context)
        if parameters['vout'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT1))
        parameters['tout'] = self.parameterAsFileOutput(parameters, self.OUTPUT2, context)
        if parameters['tout'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT2))
        
        alg_params = {
            'INPUT': parameters['lsd'],
            'INPUT1': parameters['poly'],
            'w': parameters['w'],
            'h': parameters['h'],
            'train': parameters['train']
        }
        v,t,xy,ref=Functions.resampler(alg_params)
        outputs['V'] = v
        outputs['T'] = t
        outputs['xy'] = xy
        outputs['ref'] = ref

        alg_params = {
            'INPUT1': parameters['vout'],
            'INPUT2': outputs['V'],
            'INPUT3': outputs['xy'],
            'ref': outputs['ref']
        }
        Functions.save(alg_params)

        alg_params = {
            'INPUT1': parameters['tout'],
            'INPUT2': outputs['T'],
            'INPUT3': outputs['xy'],
            'ref': outputs['ref']
        }
        Functions.save(alg_params)

        vlayer = QgsVectorLayer(parameters['vout'], 'valid', "ogr")
        QgsProject.instance().addMapLayer(vlayer)

        tlayer1 = QgsVectorLayer(parameters['tout'], 'train', "ogr")
        QgsProject.instance().addMapLayer(tlayer1)

        fileName = parameters['vout']
        layer1 = QgsVectorLayer(fileName,"test","ogr")
        subLayers =layer1.dataProvider().subLayers()

        for subLayer in subLayers:
            name = subLayer.split('!!::!!')[1]
            uri = "%s|layername=%s" % (fileName, name,)
            # Create layer
            sub_vlayer = QgsVectorLayer(uri, name, 'ogr')
            if not sub_vlayer.isValid():
                print('layer failed to load')
            # Add layer to map
            context.temporaryLayerStore().addMapLayer(sub_vlayer)
            context.addLayerToLoadOnCompletion(sub_vlayer.id(), QgsProcessingContext.LayerDetails('sample', context.project(),'LAYER1'))

        fileName = parameters['tout']
        layer1 = QgsVectorLayer(fileName,"test","ogr")
        subLayers =layer1.dataProvider().subLayers()

        for subLayer in subLayers:
            name = subLayer.split('!!::!!')[1]
            uri = "%s|layername=%s" % (fileName, name,)
            # Create layer
            sub_vlayer = QgsVectorLayer(uri, name, 'ogr')
            if not sub_vlayer.isValid():
                print('layer failed to load')
            # Add layer to map
            context.temporaryLayerStore().addMapLayer(sub_vlayer)
            context.addLayerToLoadOnCompletion(sub_vlayer.id(), QgsProcessingContext.LayerDetails('1-sample', context.project(),'LAYER1'))

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}
        return results

class Functions():
    def resampler(parameters):
        f=parameters['fold']
        poly=parameters['INPUT1']
        vlayer = QgsVectorLayer(poly, "layer", "ogr")
        ext=vlayer.extent()#xmin
        xmin = ext.xMinimum()
        xmax = ext.xMaximum()
        ymin = ext.yMinimum()
        ymax = ext.yMaximum()
        newXNumPxl=(np.ceil(abs(xmax-xmin)/(parameters['w']))-1).astype(int)
        newYNumPxl=(np.ceil(abs(ymax-ymin)/(parameters['h']))-1).astype(int)
        xsize=newXNumPxl
        ysize=newYNumPxl
        origine=[xmin,ymax]
        dem_datas=np.zeros((ysize,xsize),dtype='int64')
        # write the data to output file
        rf1=f+'/inv_sampler.tif'
        dem_datas1=np.zeros(np.shape(dem_datas),dtype='float32')
        dem_datas1[:]=dem_datas[:]
        w1=parameters['w']
        h1=parameters['h']*(-1)
        Functions.array2raster(rf1,w1,h1,dem_datas1,origine,parameters['INPUT'])##########rasterize inventory
        del dem_datas
        del dem_datas1

        IN1a=rf1
        IN2a=f+'/invq_sampler.tif'
        IN3a=f+'/inventorynxn_sampler.tif'
        Functions.cut(IN1a,IN3a,poly)##########traslate inventory
        ds15=None
        ds15 = gdal.Open(IN3a)
        if ds15 is None:#####################verify empty row input
            QgsMessageLog.logMessage("ERROR: can't open raster input", tag="WoE")
            raise ValueError  # can't open raster input, see 'WoE' Log Messages Panel
        ap=ds15.GetRasterBand(1)
        NoData=ap.GetNoDataValue()
        invmatrix = np.array(ap.ReadAsArray()).astype(np.int64)
        bands = ds15.RasterCount
        if bands>1:#####################verify bands
            QgsMessageLog.logMessage("ERROR: input rasters shoud be 1-band raster", tag="WoE")
            raise ValueError  # input rasters shoud be 1-band raster, see 'WoE' Log Messages Panel
        ###########################################load inventory
        catalog0=np.zeros(np.shape(invmatrix),dtype='int64')
        print(np.shape(invmatrix),'shape catalog')
        catalog0[:]=invmatrix[:]
        del invmatrix
        #######################################inventory from shp to tif
        v,t,XY,ref=Functions.vector2arrayinv(IN3a,parameters['INPUT'],catalog0,parameters['train'])
        return v,t,XY,ref

    def array2raster(newRasterfn,pixelWidth,pixelHeight,array,oo,lsd):
        ds = ogr.Open(lsd)
        cr=np.shape(array)
        cols=cr[1]
        rows=cr[0]
        originX = oo[0]
        originY = oo[1]
        driver = gdal.GetDriverByName('GTiff')
        print(newRasterfn)
        print(int(cols), int(rows))
        gdal.UseExceptions()
        try:
            outRaster = driver.Create(newRasterfn, int(cols), int(rows), 1, gdal.GDT_Float32)
        except Exception as e:
            print(f"Error creating raster dataset: {e}")
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        outband = outRaster.GetRasterBand(1)
        outband.SetNoDataValue(-9999)
        outband.WriteArray(array)
        outRaster.SetProjection(ds.GetLayer().GetSpatialRef().ExportToWkt())
        outband.FlushCache()
        del array

    def cut(in1,in3,poly):
        try:
            if os.path.isfile(in3):
                os.remove(in3)
            processing.run('gdal:cliprasterbymasklayer', {'INPUT': in1,'MASK': poly, 'NODATA': -9999, 'ALPHA_BAND': False, 'CROP_TO_CUTLINE': True, 'KEEP_RESOLUTION': True, 'MULTITHREADING': True, 'OPTIONS': '', 'DATA_TYPE': 6,'OUTPUT': in3})
        except:
            QgsMessageLog.logMessage("Failure to save sized /tmp input", tag="WoE")
            raise ValueError  # Failure to save sized /tmp input Log Messages Panel

    def vector2arrayinv(raster,lsd,invzero,parameters):
        rlayer = QgsRasterLayer(raster, "layer")
        if not rlayer.isValid():
            print("Layer failed to load!")
        ext=rlayer.extent()#xmin
        xm = ext.xMinimum()
        xM = ext.xMaximum()
        ym = ext.yMinimum()
        yM = ext.yMaximum()
        pxlw=rlayer.rasterUnitsPerPixelX()
        pxlh=rlayer.rasterUnitsPerPixelY()
        newXNumPxl=(np.ceil(abs(xM-xm)/(rlayer.rasterUnitsPerPixelX()))-1).astype(int)
        newYNumPxl=(np.ceil(abs(yM-ym)/(rlayer.rasterUnitsPerPixelY()))-1).astype(int)
        sizex=newXNumPxl
        sizey=newYNumPxl
        origine=[xm,yM]
        driverd = ogr.GetDriverByName('ESRI Shapefile')
        ds9 = driverd.Open(lsd)
        layer = ds9.GetLayer()
        ref = layer.GetSpatialRef()
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
        NumPxl=(np.ceil(abs((XY-OS)/size)-1)).astype(int)#from 0 first cell
        valuess=np.zeros(np.shape(invzero),dtype='float32')
        for i in range(count):
            if XY[i,1]<=yM and XY[i,1]>=ym and XY[i,0]<=xM and XY[i,0]>=xm:
                valuess[NumPxl[i,1].astype(int),NumPxl[i,0].astype(int)]=1
        rows,cols=np.where(valuess==1)
        l=len(rows)
        vec=np.arange(l)
        tt=np.ceil((parameters/100.)*l).astype(int)
        tr=np.asarray(random.sample(range(0, l), tt))
        vec[tr]=-1
        va=vec[vec>-1]
        trow=rows[tr]
        tcol=cols[tr]
        traincells=np.array([trow,tcol]).T
        vrow=rows[va]
        vcol=cols[va]
        validcells=np.array([vrow,vcol]).T
        v=[]
        t=[]
        for i in range(len(traincells)):
            ttt=np.where((NumPxl[:,1]==traincells[i,0]) & (NumPxl[:,0]==traincells[i,1]))
            t=t+list(ttt[0])
        for i in range(len(validcells)):
            vv=np.where((NumPxl[:,1]==validcells[i,0]) & (NumPxl[:,0]==validcells[i,1]))
            v=v+list(vv[0])
        return v,t,XY,ref

    def save(parameters):
        ref=parameters['ref']
        XY=parameters['INPUT3']
        driver = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(parameters['INPUT1']):
            driver.DeleteDataSource(parameters['INPUT1'])
        # create the data source
        ds=driver.CreateDataSource(parameters['INPUT1'])
        # create the layer
        layer = ds.CreateLayer("vector", ref, ogr.wkbPoint)
        # Add the fields we're interested in
        field_name = ogr.FieldDefn("id", ogr.OFTInteger)
        field_name.SetWidth(100)
        layer.CreateField(field_name)
        for i in range(len(parameters['INPUT2'])):
            #print(i)
            # create the feature
            feature = ogr.Feature(layer.GetLayerDefn())
            # Set the attributes using the values from the delimited text file
            feature.SetField("id", i)
            # create the WKT for the feature using Python string formatting
            wkt = "POINT(%f %f)" % (float(XY[parameters['INPUT2'][i],0]) , float(XY[parameters['INPUT2'][i],1]))
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
        vlayer = QgsVectorLayer(parameters['INPUT1'], 'vector', "ogr")
        # add the layer to the registry
        QgsProject.instance().addMapLayer(vlayer)