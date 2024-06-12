#!/usr/bin/python
#coding=utf-8
"""
/***************************************************************************
    rasterstatkernelAlgorithm
        begin                : 2021-11
        copyright            : (C) 2021 by Giacomo Titti,
                               Padova, November 2021
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
    rasterstatkernelAlgorithm
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
                        QgsCoordinateReferenceSystem
                        )
from qgis.core import *
from qgis import processing
from osgeo import gdal,ogr,osr
import numpy as np
import math
import operator
import random
from qgis import *
import scipy.ndimage
from qgis.utils import iface
from processing.algs.gdal.GdalUtils import GdalUtils
import tempfile
import os

class rasterstatkernelAlgorithm(QgsProcessingAlgorithm):
    # INPUT = 'INPUT'
    # INPUT1 = 'INPUT1'
    # OUTPUT = 'OUTPUT'
    # EXTENT = 'POLY'
    # RADIUS = 'BufferRadiousInPxl'

    # def tr(self, string):
    #     return QCoreApplication.translate('Processing', string)

    # def createInstance(self):
    #     return rasterstatkernelAlgorithm()

    # def name(self):
    #     return 'kernel stat'

    # def displayName(self):
    #     return self.tr('03 Points Kernel Statistics')

    # def group(self):
    #     return self.tr('01 Data preparation')

    # def groupId(self):
    #     return '01 Data preparation'

    # def shortHelpString(self):
    #     return self.tr("It calculates kernel statistic from raster around points: real, max, min, std, sum, average, range")

    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Points'), types=[QgsProcessing.TypeVectorPoint], defaultValue=None))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT1, self.tr('Raster'), defaultValue=None))
        self.addParameter(QgsProcessingParameterVectorLayer(self.EXTENT, self.tr('Contour polygon'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER, 'Buffer radious in pixels', type=QgsProcessingParameterNumber.Integer, defaultValue = 4,  minValue=1))
        #self.addParameter(QgsProcessingParameterNumber('minSlopeAcceptable', 'Min slope acceptable', type=QgsProcessingParameterNumber.Integer, defaultValue = 3,  minValue=1))
        #self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, 'Output layer', type=QgsProcessing.TypeVectorAnyGeometry, createByDefault=True, defaultValue='/tmp/nasa_1km3857clean_r6s3SE250mcomplete.shp'))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, self.tr('Output layer'), defaultValue=None,fileFilter='ESRI Shapefile (*.shp *.SHP)'))
        #self.addParameter(QgsProcessingParameterFeatureSink(self.INPUT,self.tr('output layer'),[QgsProcessing.TypeVector]))
        #self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT1, 'Output layer', type=QgsProcessing.TypeVectorAnyGeometry, createByDefault=True, defaultValue=None))


    def process(self, parameters, context, model_feedback):
        self.f=tempfile.gettempdir()
        feedback = QgsProcessingMultiStepFeedback(1, model_feedback)
        results = {}
        outputs = {}
        #parameters['Out']='/tmp/nasa_1km3857clean_r6s3SE250m_statr4.shp'
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

        #parameters['out1']=self.parameterAsSink(parameters,self.OUTPUT,context)
        #if parameters['out1'] is None:
        #    raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))

        parameters['Out'] = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        if parameters['Out'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))

        #parameters['Out'] = self.parameterAsSink(
        #    parameters,
        #    self.OUTPUT,
        #    context)

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

        # vlayer = QgsVectorLayer(parameters['Out'], 'vector', "ogr")
        # QgsProject.instance().addMapLayer(vlayer)

        # # Join attributes by location
        # alg_params = {
        #     'DISCARD_NONMATCHING': False,
        #     'INPUT': parameters['Inventory'],
        #     'JOIN': parameters['Out'],
        #     'JOIN_FIELDS': [''],
        #     'METHOD': 1,
        #     'PREDICAT4E': [2],
        #     'PREFIX': '',
        #     'OUTPUT': parameters['Out1']
        # }
        # outputs['JoinAttributesByLocation'] = processing.run('native:joinattributesbylocation', alg_params, context=context, feedback=feedback, is_child_algorithm=True)
        # results['Out'] = outputs['JoinAttributesByLocation']['OUTPUT']

        # vlayer = QgsVectorLayer(parameters['Out1'], 'vector', "ogr")
        # QgsProject.instance().addMapLayer(vlayer)

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
        # xc = ds.RasterXSize
        # yc = ds.RasterYSize
        # geot=ds.GetGeoTransform()
        # newXNumPxl=np.round(abs(xmax-xmin)/(abs(geot[1]))).astype(int)
        # newYNumPxl=np.round(abs(ymax-ymin)/(abs(geot[5]))).astype(int)
        # try:
        #     os.system('gdal_translate -of GTiff -ot Float32 -strict -outsize ' + str(newXNumPxl) +' '+ str(newYNumPxl) +' -projwin ' +str(xmin)+' '+str(ymax)+' '+ str(xmax) + ' ' + str(ymin) +' -co COMPRESS=DEFLATE -co PREDICTOR=1 -co ZLEVEL=6 ' + Slope +' '+ '/tmp/sizedslopexxx.tif')
        # except:
        #     raise ValueError  # Failure to save sized cause, see 'WoE' Log Messages Panel
        # del ds
        #ds1=gdal.Open('/tmp/sizedslopexxx.tif')
        #print('a')
        if ds1 is None:
            print("ERROR: can't open raster input")
        nodata=ds1.GetRasterBand(1).GetNoDataValue()
        band1=ds1.GetRasterBand(1)
        #print('a00')
        raster[0] = band1.ReadAsArray()
        #print('a0')
        raster[0][raster[0]==nodata]=-9999
        x = ds1.RasterXSize
        y = ds1.RasterYSize



        layer=QgsVectorLayer(parameters['INPUT3'], '', 'ogr')
        # rs=layer.crs()
        # campi=[]
        # for field in layer.fields():
        #     campi.append(field.name())
        # campi.append('geom')
        #
        # driverd = ogr.GetDriverByName('ESRI Shapefile')
        # ds9 = driverd.Open(parameters['INPUT_VECTOR_LAYER'])
        # layer = ds9.GetLayer()
        #rint(layer.QgsPointXY())
        crs=layer.crs()
        features=layer.getFeatures()
        count=0
        for feature in features:
            count +=1
            geom = feature.geometry().asPoint()
            #print(geom)
            #print(geom.asMultiPoint())
            xy=np.array([geom[0],geom[1]])
            #print(geom[0])
            try:
                XY=np.vstack((XY,xy))
            except:
                XY=xy



        # driverd = ogr.GetDriverByName('ESRI Shapefile')
        # #print('a1')
        # ds9 = driverd.Open(parameters['INPUT3'],0)
        # #print('b')
        # layer = ds9.GetLayer()
        # print('a')
        # for feature in layer:
        #     geom = feature.GetGeometryRef()
        #     xy=np.array([geom.GetX(),geom.GetY()])
        #     try:
        #         XY=np.vstack((XY,xy))
        #     except:
        #         XY=xy
        gtdem= ds1.GetGeoTransform()
        #print('c')
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
        #del ds9
        return raster,ds1,XY,crs

    def indexing(parameters):
        ggg=np.zeros(np.shape(parameters['INPUT3'][0]),dtype=np.float32)
        ggg[:]=parameters['INPUT3'][0][:]
        ggg[(ggg==-9999)]=np.nan
        numbb=parameters['INPUT']*2+1
        row,col=np.where(parameters['INPUT3'][1]==1)
        #del parameters['INPUT4']
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
            #print(ggg,'ggg')
            if ix == 0:
                g[ix] = ggg[:]
                #print(g[ix])
            if ix == 1:
                g[ix] = scipy.ndimage.generic_filter(ggg, np.nanmax, size=(numbb,numbb))
                #print(g[ix],'1')
            if ix == 2:
                g[ix] = scipy.ndimage.generic_filter(ggg, np.nanmin, size=(numbb,numbb))
                #print(g[ix])
            if ix == 3:
                g[ix] = scipy.ndimage.generic_filter(ggg, np.std, size=(numbb,numbb))
                #print(g[ix])
            if ix == 4:
                g[ix] = scipy.ndimage.generic_filter(ggg, np.sum, size=(numbb,numbb))
                #print(g[ix])
            if ix == 5:
                g[ix] = scipy.ndimage.generic_filter(ggg, np.average, size=(numbb,numbb))
                #print(g[ix])
            #if ix == 6:
            #    g[ix] = scipy.ndimage.generic_filter(ggg, np.mean, size=(numbb,numbb))
                #print(g[ix])
            if ix == 6:
                print(g)
                g[ix] = g[1]-g[2]
                #print(g[ix])

            #g[ix][(ggg==-9999)]=-9999
            #g[ix][(parameters['INPUT3'][1]==0)]=-9999
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
            #g = {}
            fn = self.f+'/stat'+str(lll[ix])+'.shp'
            if os.path.isfile(fn):
                os.remove(fn)
            layerFields = QgsFields()
            layerFields.append(QgsField('ID', QVariant.Int))
            layerFields.append(QgsField(lll[ix], QVariant.Double))



            # transform_context = QgsProject.instance().transformContext()
            # save_options = QgsVectorFileWriter.SaveVectorOptions()
            # save_options.driverName = 'SHP'
            # save_options.fileEncoding = 'UTF-8'
            #
            # writer = QgsVectorFileWriter.create(
            #   fn,
            #   layerFields,
            #   QgsWkbTypes.Point,
            #   parameters['CRS'],
            #   transform_context,
            #   save_options
            # )
            #
            # if writer.hasError() != QgsVectorFileWriter.NoError:
            #     print("Error when creating shapefile: ",  writer.errorMessage())

            writer = QgsVectorFileWriter(fn, 'UTF-8', layerFields, QgsWkbTypes.Point, parameters['CRS'], 'ESRI Shapefile')
            XYcoords=XYcoord[1:]
            for i in range(len(XYcoords)):
                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(XYcoords[i,0]) , float(XYcoords[i,1]))))
                l=[]
                l=[i]
                #print(parameters['INPUT3'],'input3')
                #print(l+parameters['INPUT3'][i],'MIO')
                #print(l+[attributi[i][ix]],'ciao')
                feat.setAttributes(l+[attributi[i][ix]])
                writer.addFeature(feat)
            del(writer)
            #iface.addVectorLayer(fn, 'layer', 'ogr')


        print('100 %...end filtering')
        del parameters['INPUT2']
        #print(XYcoord)
        XYcoord=XYcoord[1:]
        del ggg
        del parameters['INPUT3']
        #print(attributi)
        #print(attributi[0][0])
        return XYcoord,attributi

    def saveV(parameters):
        if os.path.isfile(parameters['OUTPUT']):
            os.remove(parameters['OUTPUT'])
        # set up the shapefile driver
        # create fields
        layerFields = QgsFields()
        layerFields.append(QgsField('id', QVariant.Int))
        layerFields.append(QgsField('real', QVariant.Double))
        layerFields.append(QgsField('max', QVariant.Double))
        layerFields.append(QgsField('min', QVariant.Double))
        layerFields.append(QgsField('std', QVariant.Double))
        layerFields.append(QgsField('sum', QVariant.Double))
        layerFields.append(QgsField('average', QVariant.Double))
        #layerFields.append(QgsField('mean', QVariant.Double))
        layerFields.append(QgsField('range', QVariant.Double))

        #layerFields.append(QgsField('range', QVariant.Double))

        fn = parameters['OUTPUT']



        #crs = QgsProject.instance().crs()
        # transform_context = QgsProject.instance().transformContext()
        # save_options = QgsVectorFileWriter.SaveVectorOptions()
        # save_options.driverName = 'GPKG'
        # save_options.fileEncoding = 'UTF-8'
        #
        # writer = QgsVectorFileWriter.create(
        #   fn,
        #   layerFields,
        #   QgsWkbTypes.Point,
        #   parameters['CRS'],
        #   transform_context,
        #   save_options
        # )

        writer = QgsVectorFileWriter(fn, 'UTF-8', layerFields, QgsWkbTypes.Point, parameters['CRS'], 'ESRI Shapefile')
        #print(parameters['INPUT2'],'2')
        #print(parameters['INPUT3'],'3')
        if writer.hasError() != QgsVectorFileWriter.NoError:
            print("Error when creating file: ",  writer.errorMessage())
        for i in range(len(parameters['INPUT2'])):
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(parameters['INPUT2'][i,0]) , float(parameters['INPUT2'][i,1]))))
            l=[]
            l=[i]
            #print(parameters['INPUT3'],'input3')
            #print(l+parameters['INPUT3'][i],'MIO')
            feat.setAttributes(l+parameters['INPUT3'][i])
            writer.addFeature(feat)

        del writer
        #iface.addVectorLayer(fn, '', 'ogr')

        # for i, row in df.iterrows():
        #     fet = QgsFeature()
        #     fet.setGeometry(QgsGeometry.fromWkt(row['geom']))
        #     fet.setAttributes(list(map(float,list(df.loc[ i, df.columns != 'geom']))))
        #     writer.addFeature(fet)

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








#         driver = ogr.GetDriverByName("ESRI Shapefile")
#         # Remove output shapefile if it already exists
#         if os.path.exists(parameters['OUTPUT']):
#             driver.DeleteDataSource(parameters['OUTPUT'])
#
#         ds=driver.CreateDataSource(parameters['OUTPUT'])
#         srs=osr.SpatialReference(wkt = parameters['INPUT'].GetProjection())
#         del parameters['INPUT']
#         # create the layer
#         layer = ds.CreateLayer("inventory_cleaned", srs, ogr.wkbPoint)
#
#         # Add the fields we're interested in
#         field_name = ogr.FieldDefn("id", ogr.OFTInteger)
#         field_name.SetWidth(5)
#         layer.CreateField(field_name)
#
#         field_name1 = ogr.FieldDefn("real", ogr.OFTReal)
#         field_name1.SetWidth(100)
#         layer.CreateField(field_name1)
#
#         field_name2 = ogr.FieldDefn("max", ogr.OFTReal)
#         field_name2.SetWidth(100)
#         layer.CreateField(field_name2)
#
#         field_name3 = ogr.FieldDefn("min", ogr.OFTReal)
#         field_name3.SetWidth(100)
#         layer.CreateField(field_name3)
#
#         field_name4 = ogr.FieldDefn("std", ogr.OFTReal)
#         field_name4.SetWidth(100)
#         layer.CreateField(field_name4)
#
#         field_name5 = ogr.FieldDefn("sum", ogr.OFTReal)
#         field_name5.SetWidth(100)
#         layer.CreateField(field_name5)
#
#         field_name6 = ogr.FieldDefn("average", ogr.OFTReal)
#         field_name6.SetWidth(100)
#         layer.CreateField(field_name6)
#
#         field_name7 = ogr.FieldDefn("mean", ogr.OFTReal)
#         field_name7.SetWidth(100)
#         layer.CreateField(field_name7)
#
#         field_name8 = ogr.FieldDefn("range", ogr.OFTReal)
#         field_name8.SetWidth(100)
#         layer.CreateField(field_name8)
#         # Process the text file and add the attributes and features to the shapefile
#         for i in range(len(parameters['INPUT2'])):
#             # create the feature
#             feature = ogr.Feature(layer.GetLayerDefn())
#             # Set the attributes using the values from the delimited text file
#             feature.SetField("id", i)
#             #print(np.float64(parameters['INPUT3'][i][0]))
#             feature.SetField("real", np.float64(parameters['INPUT3'][i][0]))
#             feature.SetField("max", np.float64(parameters['INPUT3'][i][1]))
#             feature.SetField("min", np.float64(parameters['INPUT3'][i][2]))
#             feature.SetField("std", np.float64(parameters['INPUT3'][i][3]))
#             feature.SetField("sum", np.float64(parameters['INPUT3'][i][4]))
#             feature.SetField("average", np.float64(parameters['INPUT3'][i][5]))
#             feature.SetField("mean", np.float64(parameters['INPUT3'][i][6]))
#             feature.SetField("range", np.float64(parameters['INPUT3'][i][7]))
#             # create the WKT for the feature using Python string formatting
#             wkt = "POINT(%f %f)" % (float(parameters['INPUT2'][i,0]) , float(parameters['INPUT2'][i,1]))
#             # Create the point from the Well Known Txt
#             point = ogr.CreateGeometryFromWkt(wkt)
#             # Set the feature geometry using the point
#             feature.SetGeometry(point)
#             # Create the feature in the layer (shapefile)
#             layer.CreateFeature(feature)
#             # Dereference the feature
#             feature = None
#         # Save and close the data source
#         feature = None
#
#
#
#
#         # Get the input Layer
#         inShapefile = "states.shp"
#         inDriver = ogr.GetDriverByName("ESRI Shapefile")
#         inDataSource = inDriver.Open(inShapefile, 0)
#         inLayer = inDataSource.GetLayer()
#
#         # Create the output Layer
#         outShapefile = "states_centroids.shp"
#         outDriver = ogr.GetDriverByName("ESRI Shapefile")
#
#         # Remove output shapefile if it already exists
#         if os.path.exists(outShapefile):
#             outDriver.DeleteDataSource(outShapefile)
#
#         # Create the output shapefile
#         outDataSource = outDriver.CreateDataSource(outShapefile)
#         outLayer = outDataSource.CreateLayer("states_centroids", geom_type=ogr.wkbPoint)
#
#         # Add input Layer Fields to the output Layer
#         inLayerDefn = inLayer.GetLayerDefn()
#         for i in range(0, inLayerDefn.GetFieldCount()):
#             fieldDefn = inLayerDefn.GetFieldDefn(i)
#             outLayer.CreateField(fieldDefn)
#
#         # Get the output Layer's Feature Definition
#         outLayerDefn = outLayer.GetLayerDefn()
#
#         # Add features to the ouput Layer
#         for i in range(0, inLayer.GetFeatureCount()):
#             # Get the input Feature
#             inFeature = inLayer.GetFeature(i)
#             # Create output Feature
#             outFeature = ogr.Feature(outLayerDefn)
#             # Add field values from input Layer
#             for i in range(0, outLayerDefn.GetFieldCount()):
#                 outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
#             # Set geometry as centroid
#             geom = inFeature.GetGeometryRef()
#             inFeature = None
#             centroid = geom.Centroid()
#             outFeature.SetGeometry(centroid)
#             # Add new feature to output Layer
#             outLayer.CreateFeature(outFeature)
#             outFeature = None
#
#         # Save and close DataSources
#         inDataSource = None
#         outDataSource = None
#
# # Save and close DataSource
#         inDataSource = None
#         outDataSource = None
#         del ds
