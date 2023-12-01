#!/usr/bin/python
#coding=utf-8
"""
/***************************************************************************
    samplerAlgorithm
        begin                : 2021-11
        copyright            : (C) 2021 by Giacomo Titti,
                               Padova, November 2021
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
    samplerAlgorithm
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

from qgis.PyQt.QtCore import QCoreApplication,QVariant
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
from processing.algs.gdal.GdalUtils import GdalUtils
import tempfile

class samplerAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'lsd'
    OUTPUT1 = 'vout'
    OUTPUT2 = 'tout'
    MASK = 'poly'
    NUMBER = 'w'
    NUMBER1 = 'h'
    NUMBER2 = 'train'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return samplerAlgorithm()

    def name(self):
        return 'points sampler'

    def displayName(self):
        return self.tr('05 Points Sampler')

    def group(self):
        return self.tr('Data preparation')

    def groupId(self):
        return 'Data preparation'

    def shortHelpString(self):
        return self.tr("Sample randomly training and validating datasets with the contraint to have only training or validating points per pixel")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Points'), types=[QgsProcessing.TypeVectorPoint], defaultValue=None))
        self.addParameter(QgsProcessingParameterVectorLayer(self.MASK, self.tr('Contour polygon'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER, 'Pixel width', type=QgsProcessingParameterNumber.Integer, defaultValue = 0,  minValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER1, 'Pixel height', type=QgsProcessingParameterNumber.Integer, defaultValue = 0,  minValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER2, 'Sample (%)', type=QgsProcessingParameterNumber.Integer, defaultValue = 0,  minValue=0))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT1, 'Layer of sample', defaultValue=None, fileFilter='ESRI Shapefile (*.shp *.SHP)'))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT2, 'Layer of 1-sample',  defaultValue=None, fileFilter='ESRI Shapefile (*.shp *.SHP)'))

    def processAlgorithm(self, parameters, context, model_feedback):
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
        v,t,xy=self.resampler(alg_params)
        outputs['V'] = v
        outputs['T'] = t
        outputs['xy'] = xy

        alg_params = {
            'INPUT1': parameters['vout'],
            'INPUT2': outputs['V'],
            'INPUT3': outputs['xy']
        }
        self.save(alg_params)

        alg_params = {
            'INPUT1': parameters['tout'],
            'INPUT2': outputs['T'],
            'INPUT3': outputs['xy']
        }
        self.save(alg_params)


        vlayer = QgsVectorLayer(parameters['vout'], 'valid', "ogr")
        QgsProject.instance().addMapLayer(vlayer)


        vlayer1 = QgsVectorLayer(parameters['tout'], 'train', "ogr")
        QgsProject.instance().addMapLayer(vlayer1)

        fileName = parameters['vout']
        print(fileName)
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
            context.addLayerToLoadOnCompletion(sub_vlayer.id(), QgsProcessingContext.LayerDetails('sample', context.project(),'LAYER1'))

        fileName = parameters['tout']
        print(fileName)
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
            context.addLayerToLoadOnCompletion(sub_vlayer.id(), QgsProcessingContext.LayerDetails('1-sample', context.project(),'LAYER1'))

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}
        return results

    def resampler(self,parameters):
        self.poly=parameters['INPUT1']
        vlayer = QgsVectorLayer(self.poly, "layer", "ogr")
        ext=vlayer.extent()#xmin
        self.xmin = ext.xMinimum()
        self.xmax = ext.xMaximum()
        self.ymin = ext.yMinimum()
        self.ymax = ext.yMaximum()
        self.newXNumPxl=(np.ceil(abs(self.xmax-self.xmin)/(parameters['w']))-1).astype(int)
        self.newYNumPxl=(np.ceil(abs(self.ymax-self.ymin)/(parameters['h']))-1).astype(int)
        self.xsize=self.newXNumPxl
        self.ysize=self.newYNumPxl
        self.origine=[self.xmin,self.ymax]
        #########################################
        #try:
        dem_datas=np.zeros((self.ysize,self.xsize),dtype='int64')
        # write the data to output file
        rf1=self.f+'/inv_sampler.tif'
        dem_datas1=np.zeros(np.shape(dem_datas),dtype='float32')
        dem_datas1[:]=dem_datas[:]#[::-1]
        w1=parameters['w']
        h1=parameters['h']*(-1)
        self.array2raster(rf1,w1,h1,dem_datas1,self.origine,parameters['INPUT'])##########rasterize inventory
        del dem_datas
        del dem_datas1
        ##################################
        IN1a=rf1
        IN2a=self.f+'/invq_sampler.tif'
        IN3a=self.f+'/inventorynxn_sampler.tif'
        self.cut(IN1a,IN3a)##########traslate inventory
        self.ds15=None
        self.ds15 = gdal.Open(IN3a)
        if self.ds15 is None:#####################verify empty row input
            QgsMessageLog.logMessage("ERROR: can't open raster input", tag="WoE")
            raise ValueError  # can't open raster input, see 'WoE' Log Messages Panel
        ap=self.ds15.GetRasterBand(1)
        NoData=ap.GetNoDataValue()
        invmatrix = np.array(ap.ReadAsArray()).astype(np.int64)
        bands = self.ds15.RasterCount
        if bands>1:#####################verify bands
            QgsMessageLog.logMessage("ERROR: input rasters shoud be 1-band raster", tag="WoE")
            raise ValueError  # input rasters shoud be 1-band raster, see 'WoE' Log Messages Panel
        ###########################################load inventory
        self.catalog0=np.zeros(np.shape(invmatrix),dtype='int64')
        print(np.shape(invmatrix),'shape catalog')
        self.catalog0[:]=invmatrix[:]
        del invmatrix
        #######################################inventory from shp to tif
        v,t,XY=self.vector2arrayinv(IN3a,parameters['INPUT'],self.catalog0,parameters['train'])
        return v,t,XY

    def array2raster(self,newRasterfn,pixelWidth,pixelHeight,array,oo,lsd):
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
        #outRaster = driver.Create(newRasterfn, int(cols), int(rows), 1, gdal.GDT_Float32)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        outband = outRaster.GetRasterBand(1)
        outband.SetNoDataValue(-9999)
        outband.WriteArray(array)
        #outRasterSRS = osr.SpatialReference()
        #outRasterSRS.ImportFromEPSG(int(self.epsg[self.epsg.rfind(':')+1:]))
        outRaster.SetProjection(ds.GetLayer().GetSpatialRef().ExportToWkt())
        outband.FlushCache()
        print(cols,rows,originX, pixelWidth,originY, pixelHeight, 'array2raster')
        del array

    def cut(self,in1,in3):
        print(self.newYNumPxl,self.newXNumPxl,'cause dimensions')
        #if self.polynum==1:
        try:
            if os.path.isfile(in3):
                os.remove(in3)

            processing.run('gdal:cliprasterbymasklayer', {'INPUT': in1,'MASK': self.poly, 'NODATA': -9999, 'ALPHA_BAND': False, 'CROP_TO_CUTLINE': True, 'KEEP_RESOLUTION': True, 'MULTITHREADING': True, 'OPTIONS': '', 'DATA_TYPE': 6,'OUTPUT': in3})

        except:
            QgsMessageLog.logMessage("Failure to save sized /tmp input", tag="WoE")
            raise ValueError  # Failure to save sized /tmp input Log Messages Panel

    def vector2arrayinv(self,raster,lsd,invzero,parameters):
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
        self.ref = layer.GetSpatialRef()
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
        return v,t,XY

    def save(self,parameters):
     
        XY=parameters['INPUT3']
        driver = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(parameters['INPUT1']):
            driver.DeleteDataSource(parameters['INPUT1'])
        # create the data source
        ds=driver.CreateDataSource(parameters['INPUT1'])
        # create the layer
        layer = ds.CreateLayer("vector", self.ref, ogr.wkbPoint)
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
