#!/usr/bin/python
# coding=utf-8
"""Predict a new layer with a previously fitted GAM model."""

import tempfile

from qgis.core import (
    QgsProcessing,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingMultiStepFeedback,
    QgsProcessingParameterField,
    QgsProcessingParameterFile,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterVectorLayer,
    QgsVectorLayer,
)

from sz_module.scripts.algorithms import Algorithms,GAM_utils
from sz_module.scripts.utils import SZ_utils


class CoreAlgorithmGAM_predict():

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

        predictor_names=model_artifact['feature_names']
        load_parameters={
            'INPUT_VECTOR_LAYER':source.source(),
            'nomi':predictor_names,
            'lsd':'__sz_no_target__',
            'family':model_artifact['family'],
            'time':year_field if year_field not in predictor_names else None,
        }
        try:
            prediction_df,prediction_crs=SZ_utils.load_cv(
                tempfile.gettempdir(),load_parameters
            )
        except KeyError as error:
            raise QgsProcessingException(
                'The prediction layer does not contain every predictor required by '
                f"the GAM model: {', '.join(predictor_names)}"
            ) from error

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        try:
            prediction_df=Algorithms.GAM_transfer({
                'predictors_weights':model_artifact,
                'nomi':predictor_names,
                'family':model_artifact['family'],
                'df':prediction_df,
            })
        except ValueError as error:
            raise QgsProcessingException(str(error)) from error

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        SZ_utils.save({
            'df':prediction_df,
            'crs':prediction_crs,
            'OUT':output_path,
        })

        feedback.setCurrentStep(3)
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
