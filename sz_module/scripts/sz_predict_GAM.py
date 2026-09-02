#!/usr/bin/python
# coding=utf-8
"""Predict a new layer with a previously fitted GAM model."""

import os

import numpy as np

from qgis.core import (
    QgsProcessing,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingMultiStepFeedback,
    QgsFeature,
    QgsFeatureRequest,
    QgsField,
    QgsFields,
    QgsProcessingParameterField,
    QgsProcessingParameterFile,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterVectorLayer,
    QgsVectorFileWriter,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QVariant

from sz_module.scripts.algorithms import GAM_utils
from sz_module.scripts.utils import SZ_utils


class CoreAlgorithmGAM_predict():

    BATCH_SIZE=5000

    def init(self,config=None):
        self.addParameter(QgsProcessingParameterFile(
            self.FILE,
            'Trusted pre-trained GAM model (.pkl)',
            behavior=QgsProcessingParameterFile.File,
            extension='pkl',
            defaultValue=None,
        ))
        self.addParameter(QgsProcessingParameterVectorLayer(
            self.INPUT,
            self.tr('Input layer for prediction'),
            types=[QgsProcessing.TypeVectorPolygon],
            defaultValue=None,
        ))
        self.addParameter(QgsProcessingParameterField(
            self.STRING6,
            'Year field to copy to output',
            parentLayerParameterName=self.INPUT,
            defaultValue=None,
            allowMultiple=False,
            type=QgsProcessingParameterField.Any,
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT,
            'Prediction output',
            fileFilter='GeoPackage (*.gpkg *.GPKG)',
            defaultValue=None,
        ))
        self.addParameter(QgsProcessingParameterFolderDestination(
            self.OUTPUT3,
            'Outputs folder destination',
            defaultValue=None,
            createByDefault=True,
        ))

    def process(self,parameters,context,feedback):
        feedback=QgsProcessingMultiStepFeedback(4,feedback)
        results={}

        model_file=self.parameterAsFile(parameters,self.FILE,context)
        if not model_file:
            raise QgsProcessingException(self.invalidSourceError(parameters,self.FILE))
        try:
            model_artifact=GAM_utils.GAM_load(model_file)
        except ValueError as error:
            raise QgsProcessingException(str(error)) from error

        source=self.parameterAsVectorLayer(parameters,self.INPUT,context)
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters,self.INPUT))

        year_field=self.parameterAsString(parameters,self.STRING6,context)
        output_path=self.parameterAsFileOutput(parameters,self.OUTPUT,context)
        if not output_path:
            raise QgsProcessingException(self.invalidSourceError(parameters,self.OUTPUT))
        output_folder=self.parameterAsString(parameters,self.OUTPUT3,context)
        if not output_folder:
            raise QgsProcessingException(self.invalidSourceError(parameters,self.OUTPUT3))
        SZ_utils.make_directory({'path':output_folder})

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        completed=CoreAlgorithmGAM_predict.predict_to_geopackage(
            self,
            source,model_artifact,year_field,output_path,context,feedback
        )
        if not completed:
            return {}

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        layer=QgsVectorLayer(output_path,'GAM prediction','ogr')
        for sublayer in layer.dataProvider().subLayers():
            name=sublayer.split('!!::!!')[1]
            uri=f'{output_path}|layername={name}'
            output_layer=QgsVectorLayer(uri,name,'ogr')
            if not output_layer.isValid():
                raise QgsProcessingException(f'Unable to load output layer: {uri}')
            context.temporaryLayerStore().addMapLayer(output_layer)
            context.addLayerToLoadOnCompletion(
                output_layer.id(),
                QgsProcessingContext.LayerDetails(
                    'GAM prediction',context.project(),'LAYER'
                ),
            )

        results[self.OUTPUT]=output_path
        results[self.OUTPUT3]=output_folder
        return results

    def predict_to_geopackage(self,source,model_artifact,year_field,output_path,context,feedback):
        predictor_names=model_artifact['feature_names']
        source_fields=source.fields()
        missing_fields=[name for name in predictor_names if source_fields.indexOf(name)<0]
        if missing_fields:
            raise QgsProcessingException(
                'The prediction layer is missing these GAM predictors: '
                + ', '.join(missing_fields)
            )
        if year_field and source_fields.indexOf(year_field)<0:
            raise QgsProcessingException(
                f"The year field '{year_field}' is not present in the prediction layer."
            )

        copied_year=year_field if year_field not in predictor_names else ''
        output_fields=QgsFields()
        for name in predictor_names:
            output_fields.append(QgsField(source_fields[source_fields.indexOf(name)]))
        if copied_year:
            output_fields.append(QgsField(source_fields[source_fields.indexOf(copied_year)]))
        output_fields.append(QgsField('ID',QVariant.LongLong))
        output_fields.append(QgsField('SI',QVariant.Double))

        options=QgsVectorFileWriter.SaveVectorOptions()
        options.driverName='GPKG'
        options.fileEncoding='UTF-8'
        options.layerName=os.path.splitext(os.path.basename(output_path))[0]
        writer=QgsVectorFileWriter.create(
            output_path,output_fields,source.wkbType(),source.crs(),
            context.transformContext(),options,
        )
        if writer.hasError()!=QgsVectorFileWriter.NoError:
            message=writer.errorMessage()
            del writer
            raise QgsProcessingException(
                f'Unable to create prediction GeoPackage: {message}'
            )

        requested_names=list(dict.fromkeys(
            predictor_names+([copied_year] if copied_year else [])
        ))
        request=QgsFeatureRequest().setSubsetOfAttributes(requested_names,source_fields)
        total_features=max(source.featureCount(),0)
        batch=[]
        output_id=0

        for feature in source.getFeatures(request):
            if feedback.isCanceled():
                del writer
                return False
            geometry=feature.geometry()
            if geometry is None or geometry.isEmpty():
                feedback.reportError(
                    f'Skipping feature {feature.id()}: empty geometry.',fatalError=False
                )
                continue
            predictor_values=[feature[name] for name in predictor_names]
            year_value=feature[copied_year] if copied_year else None
            batch.append((geometry,predictor_values,year_value,output_id,feature.id()))
            output_id+=1

            if len(batch)>=CoreAlgorithmGAM_predict.BATCH_SIZE:
                CoreAlgorithmGAM_predict.write_prediction_batch(
                    self,
                    writer,output_fields,batch,model_artifact,copied_year
                )
                batch=[]
                if total_features:
                    feedback.setProgress(min(100.0,100.0*output_id/total_features))

        if batch:
            CoreAlgorithmGAM_predict.write_prediction_batch(
                self,
                writer,output_fields,batch,model_artifact,copied_year
            )
        del writer
        feedback.setProgress(100.0)
        return True

    def write_prediction_batch(self,writer,output_fields,batch,model_artifact,copied_year):
        try:
            matrix=np.asarray([item[1] for item in batch],dtype=float)
        except (TypeError,ValueError) as error:
            feature_ids=', '.join(str(item[4]) for item in batch[:5])
            raise QgsProcessingException(
                'Non-numeric or invalid predictor values near feature IDs: '+feature_ids
            ) from error

        for name,statistics in model_artifact['scaler'].items():
            column_index=model_artifact['feature_names'].index(name)
            matrix[:,column_index]=(
                matrix[:,column_index]-statistics['mean']
            )/statistics['std']

        estimator=model_artifact['estimator']
        try:
            if model_artifact['family']=='binomial':
                predictions=estimator.predict_proba(matrix)
            else:
                predictions=estimator.predict(matrix)
        except (TypeError,ValueError) as error:
            feature_ids=', '.join(str(item[4]) for item in batch[:5])
            raise QgsProcessingException(
                'GAM prediction failed near feature IDs: '+feature_ids
            ) from error

        output_features=[]
        for item,prediction in zip(batch,predictions):
            geometry,predictor_values,year_value,output_id,_=item
            output_feature=QgsFeature(output_fields)
            output_feature.setGeometry(geometry)
            attributes=list(predictor_values)
            if copied_year:
                attributes.append(year_value)
            attributes.extend([output_id,float(prediction)])
            output_feature.setAttributes(attributes)
            output_features.append(output_feature)
        if not writer.addFeatures(output_features):
            raise QgsProcessingException(
                'Unable to write a prediction batch to the output GeoPackage.'
            )
