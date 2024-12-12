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

from qgis.PyQt.QtCore import QVariant
from qgis.core import (QgsProcessing,
                        QgsProcessingException,
                        QgsProcessingAlgorithm,
                        QgsProcessingParameterRasterLayer,
                        QgsProcessingMultiStepFeedback,
                        QgsProcessingParameterNumber,
                        QgsProcessingParameterFileDestination,
                        QgsProcessingParameterVectorLayer,
                        QgsVectorLayer,
                        QgsField,
                        QgsFields,
                        QgsVectorFileWriter,
                        QgsWkbTypes,
                        QgsFeature,
                        QgsGeometry,
                        QgsPointXY,
                        )
from qgis.core import *
from osgeo import gdal
import numpy as np
from qgis import *
import scipy.ndimage
import tempfile
import os

class rasterstatkernelAlgorithm(QgsProcessingAlgorithm):
   
    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Points'), types=[QgsProcessing.TypeVectorPoint], defaultValue=None))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT1, self.tr('Raster'), defaultValue=None))
        self.addParameter(QgsProcessingParameterVectorLayer(self.EXTENT, self.tr('Contour polygon'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER, 'Buffer radious in pixels', type=QgsProcessingParameterNumber.Integer, defaultValue = 4,  minValue=1))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, self.tr('Output layer'), defaultValue=None,fileFilter='ESRI Shapefile (*.shp *.SHP)'))

    def process(self, parameters, context, model_feedback):
        self.f=tempfile.gettempdir()
        feedback = QgsProcessingMultiStepFeedback(1, model_feedback)
        results = {}
        outputs = {}
        parameters['Slope'] = self.parameterAsRasterLayer(parameters, self.INPUT1, context).source()
        if parameters['Slope'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT1))
        parameters['Inventory'] = self.parameterAsVectorLayer(parameters, self.INPUT, context).source()
        if parameters['Inventory'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))
        parameters['poly'] = self.parameterAsVectorLayer(parameters, self.EXTENT, context).source()
        if parameters['poly'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.EXTENT))
        parameters['BufferRadiousInPxl'] = self.parameterAsInt(parameters, self.RADIUS, context)
        if parameters['BufferRadiousInPxl'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.RADIUS))

        parameters['Out'] = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        if parameters['Out'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))

        print('importing')
        alg_params = {
            'INPUT': parameters['poly'],
            'INPUT2': parameters['Slope'],
            'INPUT3' : parameters['Inventory']
        }
        raster,ds1,XY,crs=Functions.importing(alg_params)
        outputs['raster'] = raster
        outputs['ds1'] = ds1
        outputs['XY'] = XY
        outputs['crs']= crs

        print('indexing')
        alg_params = {
            'INPUT': parameters['BufferRadiousInPxl'],
            'INPUT3': outputs['raster'],
            'INPUT2': outputs['XY'],
            'INPUT1': outputs['ds1'],
            'CRS': outputs['crs']
        }
        XYcoord,attributi=Functions.indexing(alg_params)
        outputs['XYcoord'] = XYcoord
        outputs['attributi'] = attributi

        print('save')
        alg_params = {
            'OUTPUT': parameters['Out'],
            'INPUT2': outputs['XYcoord'],
            'INPUT': outputs['ds1'],
            'INPUT3': outputs['attributi'],
            'CRS':outputs['crs']
        }
        Functions.saveV(alg_params)

        fileName = parameters['Out']
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
            context.addLayerToLoadOnCompletion(sub_vlayer.id(), QgsProcessingContext.LayerDetails('out', context.project(),'LAYER1'))

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}
        return results

class Functions():
    def importing(parameters):
        vlayer = QgsVectorLayer(parameters['INPUT'], "layer", "ogr")
        ext=vlayer.extent()#xmin
        xmin = ext.xMinimum()
        xmax = ext.xMaximum()
        ymin = ext.yMinimum()
        ymax = ext.yMaximum()
        raster={}
        ds1=gdal.Open(parameters['INPUT2'])
        if ds1 is None:
            print("ERROR: can't open raster input")
        nodata=ds1.GetRasterBand(1).GetNoDataValue()
        band1=ds1.GetRasterBand(1)
        raster[0] = band1.ReadAsArray()
        raster[0][raster[0]==nodata]=-9999
        x = ds1.RasterXSize
        y = ds1.RasterYSize
        layer=QgsVectorLayer(parameters['INPUT3'], '', 'ogr')
        crs=layer.crs()
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
        NumPxl[NumPxl==-1.]=0
        values=np.zeros((y,x), dtype=int)
        for i in range(len(NumPxl)):
            if XY[i,1]<ymax and XY[i,1]>ymin and XY[i,0]<xmax and XY[i,0]>xmin:
                values[NumPxl[i,1].astype(int),NumPxl[i,0].astype(int)]=1
        raster[1]=values[:]
        del values
        del layer
        return raster,ds1,XY,crs

    def indexing(parameters):
        ggg=np.zeros(np.shape(parameters['INPUT3'][0]),dtype=np.float32)
        ggg[:]=parameters['INPUT3'][0][:]
        ggg[(ggg==-9999)]=np.nan
        numbb=parameters['INPUT']*2+1
        row,col=np.where(parameters['INPUT3'][1]==1)
        geo=parameters['INPUT1'].GetGeoTransform()
        xsize=geo[1]
        ysize=geo[5]
        OOx=geo[0]
        OOy=geo[3]
        XYcoord=np.array([0,0])
        attributi={}
        print('filtering...')
        g={}
        for ix in range(7):
            lll=['real','max','min','std','sum','average','range']
            print(ix*15, '%')
            if ix == 0:
                g[ix] = ggg[:]
            if ix == 1:
                g[ix] = scipy.ndimage.generic_filter(ggg, np.nanmax, size=(numbb,numbb))
            if ix == 2:
                g[ix] = scipy.ndimage.generic_filter(ggg, np.nanmin, size=(numbb,numbb))
            if ix == 3:
                g[ix] = scipy.ndimage.generic_filter(ggg, np.std, size=(numbb,numbb))
            if ix == 4:
                g[ix] = scipy.ndimage.generic_filter(ggg, np.sum, size=(numbb,numbb))
            if ix == 5:
                g[ix] = scipy.ndimage.generic_filter(ggg, np.average, size=(numbb,numbb))
            if ix == 6:
                print(g)
                g[ix] = g[1]-g[2]
            count=0
            for i in range(len(col)):
                xmin=OOx+(xsize*col[i])
                xmax=OOx+(xsize*col[i])+(xsize)
                ymax=OOy+(ysize*row[i])
                ymin=OOy+(ysize*row[i])+(ysize)
                for ii in range(len(parameters['INPUT2'])):
                    if (parameters['INPUT2'][ii,0]>=xmin and parameters['INPUT2'][ii,0]<=xmax and parameters['INPUT2'][ii,1]>=ymin and parameters['INPUT2'][ii,1]<=ymax):
                        if ix==0:
                            XYcoord=np.vstack((XYcoord,parameters['INPUT2'][ii,:]))
                        try:
                            attributi[count]=attributi[count]+[float(g[ix][row[i],col[i]])]
                        except:
                            attributi[count]=[float(g[ix][row[i],col[i]])]
                        count+=1
            fn = self.f+'/stat'+str(lll[ix])+'.shp'
            if os.path.isfile(fn):
                os.remove(fn)
            layerFields = QgsFields()
            layerFields.append(QgsField('ID', QVariant.Int))
            layerFields.append(QgsField(lll[ix], QVariant.Double))
            writer = QgsVectorFileWriter(fn, 'UTF-8', layerFields, QgsWkbTypes.Point, parameters['CRS'], 'ESRI Shapefile')
            XYcoords=XYcoord[1:]
            for i in range(len(XYcoords)):
                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(XYcoords[i,0]) , float(XYcoords[i,1]))))
                l=[]
                l=[i]
                feat.setAttributes(l+[attributi[i][ix]])
                writer.addFeature(feat)
            del(writer)

        print('100 %...end filtering')
        del parameters['INPUT2']
        XYcoord=XYcoord[1:]
        del ggg
        del parameters['INPUT3']
        return XYcoord,attributi

    def saveV(parameters):
        if os.path.isfile(parameters['OUTPUT']):
            os.remove(parameters['OUTPUT'])
        layerFields = QgsFields()
        layerFields.append(QgsField('id', QVariant.Int))
        layerFields.append(QgsField('real', QVariant.Double))
        layerFields.append(QgsField('max', QVariant.Double))
        layerFields.append(QgsField('min', QVariant.Double))
        layerFields.append(QgsField('std', QVariant.Double))
        layerFields.append(QgsField('sum', QVariant.Double))
        layerFields.append(QgsField('average', QVariant.Double))
        layerFields.append(QgsField('range', QVariant.Double))
        fn = parameters['OUTPUT']
        writer = QgsVectorFileWriter(fn, 'UTF-8', layerFields, QgsWkbTypes.Point, parameters['CRS'], 'ESRI Shapefile')
        if writer.hasError() != QgsVectorFileWriter.NoError:
            print("Error when creating file: ",  writer.errorMessage())
        for i in range(len(parameters['INPUT2'])):
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(parameters['INPUT2'][i,0]) , float(parameters['INPUT2'][i,1]))))
            l=[]
            l=[i]
            feat.setAttributes(l+parameters['INPUT3'][i])
            writer.addFeature(feat)
        del writer

    def addmap(parameters):
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