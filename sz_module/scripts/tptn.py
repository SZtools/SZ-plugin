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

import os
import sys
sys.setrecursionlimit(10000)
from qgis.PyQt.QtCore import QVariant
from qgis.core import (QgsProcessing,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingMultiStepFeedback,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFileDestination,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterVectorLayer,
                       QgsVectorLayer,
                       QgsProject,
                       QgsField,
                       QgsFields,
                       QgsVectorFileWriter,
                       QgsWkbTypes,
                       QgsFeature,
                       QgsGeometry,
                       QgsProcessingParameterField,
                       QgsProcessingParameterField,
                       QgsProcessingContext
                       )
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,confusion_matrix, ConfusionMatrixDisplay
import tempfile
import matplotlib.pyplot as plt


class FPAlgorithm(QgsProcessingAlgorithm):
 
    def init(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('Input layer'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING, 'Index', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING2, 'Field of dependent variable (0 for absence, > 0 for presence)', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterNumber(self.NUMBER, self.tr('Cutoff percentile (if empty use the YOUDEN index)'), minValue=1,type=QgsProcessingParameterNumber.Integer,optional=True))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, 'Output confusion matrix', fileFilter='GeoPackage (*.gpkg *.GPKG)', defaultValue=None))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT3, 'Diagnostic plots folder', defaultValue=None, createByDefault=True))


    def process(self, parameters, context, feedback):
        self.f=tempfile.mkdtemp(prefix='SZ_confusion_matrix_')
        feedback = QgsProcessingMultiStepFeedback(1, feedback)
        results = {}
        outputs = {}

        source = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))
        parameters['covariates']=source.source()

        parameters['field1'] = self.parameterAsString(parameters, self.STRING, context)
        if parameters['field1'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING))

 
        parameters['fieldlsd'] = self.parameterAsString(parameters, self.STRING2, context)
        if parameters['fieldlsd'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING2))

        parameters['testN'] = self.parameterAsInt(parameters, self.NUMBER, context)
        if parameters['testN'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.NUMBER))
   
        parameters['out'] = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        if not parameters['out']:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))
        
        parameters['folder'] = self.parameterAsString(parameters, self.OUTPUT3, context)
        if not parameters['folder']:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT3))

        alg_params = {
            'INPUT_VECTOR_LAYER': parameters['covariates'],
            'field1': parameters['field1'],
            'lsd' : parameters['fieldlsd'],
            'testN':parameters['testN'],
            'fold':self.f,
            'OUT': parameters['folder']
        }

        outputs['df'],outputs['nomi'],outputs['crs']=Functions.load(alg_params)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        alg_params = {
            'df': outputs['df'],
            'crs': outputs['crs'],
            'OUT': parameters['out']
        }
        Functions.save(alg_params)

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        results[self.OUTPUT] = parameters['out']
        results[self.OUTPUT3] = parameters['folder']
 
        fileName = parameters['out']
        layer1 = QgsVectorLayer(fileName,"confusion_matrix","ogr")
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
            context.addLayerToLoadOnCompletion(sub_vlayer.id(), QgsProcessingContext.LayerDetails("confusion_matrix", context.project(),'LAYER1'))

        feedback.setCurrentStep(4)
        if feedback.isCanceled():
            return {}

        return results

class Functions():
    def load(parameters):
        f=parameters['fold']
        layer = QgsVectorLayer(parameters['INPUT_VECTOR_LAYER'], '', 'ogr')
        crs=layer.crs()
        campi=[]
        for field in layer.fields():
            campi.append(field.name())
        campi.append('geom')
        gdp=pd.DataFrame(columns=campi,dtype=float)
        features = layer.getFeatures()
        count=0
        feat=[]
        for feature in features:
            attr=feature.attributes()
            geom = feature.geometry()
            feat=attr+[geom.asWkt()]
            gdp.loc[len(gdp)] = feat
            count=+ 1
        gdp.to_csv(os.path.join(f, 'file.csv'))
        del gdp
        gdp=pd.read_csv(os.path.join(f, 'file.csv'))
        gdp['ID']=np.arange(1,len(gdp.iloc[:,0])+1)
        df=pd.DataFrame(data=gdp[parameters['field1']].to_numpy(), columns=[parameters['field1']])
        nomi=list(df.head())
        lsd=gdp[parameters['lsd']]
        lsd[lsd>0]=1
        df['y']=lsd#.astype(int)
        df['ID']=gdp['ID']
        df['geom']=gdp['geom']
        df=df.dropna(how='any',axis=0)
        x=df[parameters['field1']].to_numpy()
        y=df['y'].to_numpy()
        if parameters['testN']==0:
            fpr1, tpr1, tresh1 = roc_curve(y,x)
            j = tpr1 - fpr1
            best_idx = int(np.argmax(j))
            cutoff = float(tresh1[best_idx])# x YOUDEN INDEX
        else:
            xx_desc = np.sort(x)[::-1]
            cutoff = float(np.percentile(xx_desc, parameters['testN']))
        print('cutoff: ',cutoff)

        y_pred = (x > cutoff).astype(int)  # 1=positive, 0=negative

        # Confusion matrix: tn, fp, fn, tp (sklearn order)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")

        # Save as PDF
        plt.savefig(os.path.join(parameters['OUT'], "confusion_matrix.pdf"), format="pdf")
        plt.close()

        print("tp=", tp)
        print("tn=", tn)
        print("fp=", fp)
        print("fn=", fn)

        # Add columns similar to yours (but safe assignments)
        df["class_cut"] = np.where(y_pred == 1, "positive", "negative")
        df["presabs"] = np.where(y == 1, "true", "false")

        # Match your tptnfpfn coding:
        # 0=tp, 1=tn, 2=fp, 3=fn
        # Use vectorized logic (no chained .iloc)
        df["tptnfpfn"] = np.select(
            [
                (y_pred == 1) & (y == 1),  # tp
                (y_pred == 0) & (y == 0),  # tn
                (y_pred == 1) & (y == 0),  # fp
                (y_pred == 0) & (y == 1),  # fn
            ],
            [0, 1, 2, 3],
            default=-1
        ).astype(int)

        return df, nomi, crs

    def save(parameters):
        df=parameters['df']
        nomi=list(df.columns)
        fields = QgsFields()
        for field in nomi:
            if field=='geom':
                continue
            elif field in ('ID', 'y', 'tptnfpfn'):
                fields.append(QgsField(field, QVariant.Int))
            elif field in ('class_cut', 'presabs'):
                fields.append(QgsField(field, QVariant.String))
            else:
                fields.append(QgsField(field, QVariant.Double))

        transform_context = QgsProject.instance().transformContext()
        save_options = QgsVectorFileWriter.SaveVectorOptions()
        save_options.driverName = 'GPKG'
        save_options.fileEncoding = 'UTF-8'

        writer = QgsVectorFileWriter.create(
          parameters['OUT'],
          fields,
          QgsWkbTypes.Polygon,
          parameters['crs'],
          transform_context,
          save_options
        )
        
        if writer.hasError() != QgsVectorFileWriter.NoError:
            raise QgsProcessingException(
                f"Error creating GeoPackage: {writer.errorMessage()}"
            )

        attribute_columns = [field for field in df.columns if field != 'geom']
        for i, row in df.iterrows():
            fet = QgsFeature(fields)
            fet.setGeometry(QgsGeometry.fromWkt(row['geom']))
            attributes = []
            for field in attribute_columns:
                value = row[field]
                if isinstance(value, np.generic):
                    value = value.item()
                attributes.append(value)
            fet.setAttributes(attributes)
            if not writer.addFeature(fet):
                raise QgsProcessingException(
                    f"Unable to write output feature {i}"
                )
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
