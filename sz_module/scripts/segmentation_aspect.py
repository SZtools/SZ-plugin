#!/usr/bin/python
#coding=utf-8

"""
Model exported as python.
Name : segmentation_aspect
Group : SU metrics
With QGIS : 32811
"""

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
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingMultiStepFeedback,
                       QgsProcessingParameterFileDestination,
                       QgsVectorLayer,
                       QgsProcessingParameterMultipleLayers)

from qgis import processing
import numpy as np
from qgis import *
import pandas as pd
import tempfile
import processing

from qgis.core import (
    QgsProcessingParameterFile,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingMultiStepFeedback,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterNumber
)
import processing
from sz_module.scripts.utils import SZ_utils
#import geopandas as gpd

from shapely import wkt
from shapely.strtree import STRtree
import time




class segmentationAspectAlgorithm():
   
    def init(self, config=None):
        self.addParameter(QgsProcessingParameterMultipleLayers('su', 'SU', layerType=QgsProcessing.TypeVectorPolygon, defaultValue=['su_test','su1_test','su2_test']))
        self.addParameter(QgsProcessingParameterVectorLayer('point', 'Landslide Identification Points', types=[QgsProcessing.TypeVectorPoint], defaultValue='lip_test'))
        self.addParameter(QgsProcessingParameterVectorLayer('poly', 'Landslide Polygons', types=[QgsProcessing.TypeVectorPolygon], defaultValue='lsd_test'))
        self.addParameter(QgsProcessingParameterRasterLayer('dem', 'DEM', defaultValue='dem_test'))
        self.addParameter(QgsProcessingParameterNumber('minarea', 'Minimum area', type=QgsProcessingParameterNumber.Integer, defaultValue=None))
        #self.addParameter(QgsProcessingParameterFileDestination('output_a', 'Output Area Metric', '*.csv', defaultValue=None))
        #self.addParameter(QgsProcessingParameterFileDestination('output_d', 'Output Density Metric', '*.csv', defaultValue=None))
        #self.addParameter(QgsProcessingParameterFileDestination('output_sm', 'Output Aspect Segmentation Metric', '*.csv', defaultValue=None))
        #self.addParameter(QgsProcessingParameterFileDestination('output_s', 'Output Segmentation', '*.csv', defaultValue=None))
        self.addParameter(QgsProcessingParameterFile('folder', 'Destination folder', behavior=QgsProcessingParameterFile.Folder, fileFilter='All files (*.*)', defaultValue=None,optional=True))



    def process(self, parameters, context, feedback):
        start_time = time.time()


        self.f=tempfile.mkdtemp(prefix="SZ_")
        print('TEMP folder: ',self.f)
        feedback = QgsProcessingMultiStepFeedback(1, feedback)
        results = {}
        outputs = {}

        parameters['SU'] = self.parameterAsLayerList(parameters, 'su', context)
        
        parameters['lip'] = self.parameterAsVectorLayer(parameters, 'point', context)
        if parameters['lip'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, 'point'))
        parameters['lip']=parameters['lip'].source().split('|')[0]
        
        parameters['poly_lsd'] = self.parameterAsVectorLayer(parameters, 'poly', context)
        if parameters['poly_lsd'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, 'poly'))
        parameters['poly_lsd']=parameters['poly_lsd'].source().split('|')[0]

        parameters['dem'] = self.parameterAsRasterLayer(parameters, 'dem', context)
        if parameters['dem'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, 'dem'))
        parameters['dem']=parameters['dem'].source().split('|')[0]

        parameters['minarea'] = self.parameterAsInt(parameters, 'minarea', context)
        if parameters['minarea'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, 'minarea'))

        # parameters['outcsv_a'] = self.parameterAsFileOutput(parameters, 'output_a', context)
        # if parameters['outcsv_a'] is None:
        #     raise QgsProcessingException(self.invalidSourceError(parameters, 'output_a'))

        # parameters['outcsv_d'] = self.parameterAsFileOutput(parameters, 'output_d', context)
        # if parameters['outcsv_d'] is None:
        #     raise QgsProcessingException(self.invalidSourceError(parameters, 'output_d'))
        
        # parameters['outcsv_sm'] = self.parameterAsFileOutput(parameters, 'output_sm', context)
        # if parameters['outcsv_sm'] is None:
        #     raise QgsProcessingException(self.invalidSourceError(parameters, 'output_sm'))
        
        # parameters['outcsv_s'] = self.parameterAsFileOutput(parameters, 'output_s', context)
        # if parameters['outcsv_s'] is None:
        #     raise QgsProcessingException(self.invalidSourceError(parameters, 'output_s'))

        parameters['folder'] = self.parameterAsString(parameters, 'folder', context)
        if parameters['folder'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, 'folder'))
        
        if parameters['folder']=='':
            parameters['folder']=self.f
        

        # Aspect
        alg_params = {
            'INPUT': parameters['dem'],
            'Z_FACTOR': 1,
            'OUTPUT': self.f+'/dem.tif'
        }
        outputs['Aspect'] = processing.run('native:aspect', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        # SIN Raster calculator
        alg_params = {
            'BAND_A': 1,
            'BAND_B': None,
            'BAND_C': None,
            'BAND_D': None,
            'BAND_E': None,
            'BAND_F': None,
            'EXTRA': '',
            'FORMULA': 'sin(A)',
            'INPUT_A': outputs['Aspect']['OUTPUT'],
            'INPUT_B': parameters['dem'],
            'INPUT_C': parameters['dem'],
            'INPUT_D': parameters['dem'],
            'INPUT_E': parameters['dem'],
            'INPUT_F': parameters['dem'],
            'NO_DATA': None,
            'OPTIONS': '',
            'PROJWIN': None,
            'RTYPE': 5,  # Float32
            'OUTPUT': self.f+'/SinRasterCalculator.tif'
        }
        outputs['SinRasterCalculator'] = processing.run('gdal:rastercalculator', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        # COS Raster calculator
        alg_params = {
            'BAND_A': 1,
            'BAND_B': None,
            'BAND_C': None,
            'BAND_D': None,
            'BAND_E': None,
            'BAND_F': None,
            'EXTRA': '',
            'FORMULA': 'cos(A)',
            'INPUT_A': outputs['Aspect']['OUTPUT'],
            'INPUT_B': parameters['dem'],
            'INPUT_C': parameters['dem'],
            'INPUT_D': parameters['dem'],
            'INPUT_E': parameters['dem'],
            'INPUT_F': parameters['dem'],
            'NO_DATA': None,
            'OPTIONS': '',
            'PROJWIN': None,
            'RTYPE': 5,  # Float32
            'OUTPUT': self.f+'/CosRasterCalculator.tif'
        }
        outputs['CosRasterCalculator'] = processing.run('gdal:rastercalculator', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(3)
        if feedback.isCanceled():
            return {}
        
        table=pd.DataFrame({'name':[],'V':[],'I':[],'F':[]})
        final_table=pd.DataFrame({'name':[],'A':[],'D':[],'F':[], 'S':[]})

        for i,SU in enumerate(parameters['SU']):

            print(SU)

            alg_params = {
                'INPUT': SU.source(),
                'METHOD': 1,  # Structure
                'OUTPUT': self.f+'/FixGeometries'+str(i)+'.gpkg'
            }
            outputs['FixGeometries'] = processing.run('native:fixgeometries', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

            # Add geometry attributes
            alg_params = {
                'CALC_METHOD': 0,  # Layer CRS
                'INPUT': outputs['FixGeometries']['OUTPUT'],
                'OUTPUT': self.f+'/Area'+str(i)+'.gpkg'
            }
            outputs['Area'] = processing.run('qgis:exportaddgeometrycolumns', alg_params, context=context, feedback=feedback, is_child_algorithm=True)
            
            alg_params = {
            'FIELD': 'area',
            'INPUT': outputs['Area']['OUTPUT'],
            'OPERATOR': 2,  # <
            'VALUE': parameters['minarea'],
            'OUTPUT': self.f+'/SUclean'+str(i)+'.gpkg'
            }
            outputs['SUclean'] = processing.run('native:extractbyattribute', alg_params, context=context, feedback=feedback, is_child_algorithm=True)
            
            alg_params = {
                'COLUMN': ['area'],
                'INPUT': outputs['SUclean']['OUTPUT'],
                'OUTPUT': self.f+'/DropFields'+str(i)+'.gpkg'
            }
            outputs['DropFields'] = processing.run('native:deletecolumn', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

            alg_params = {
                'FIELD_LENGTH': 100,
                'FIELD_NAME': 'fid',
                'FIELD_PRECISION': 0,
                'FIELD_TYPE': 1,  # Integer (32 bit)
                'FORMULA': '$id',
                'INPUT': outputs['DropFields']['OUTPUT'],
                'OUTPUT': self.f+'/FieldCalculator'+str(i)+'.gpkg'
            }
            outputs['FieldCalculator'] = processing.run('native:fieldcalculator', alg_params, context=context, feedback=feedback, is_child_algorithm=True)
       
            #################Area

            # Intersection
            alg_params = {
                'GRID_SIZE': None,
                'INPUT': parameters['poly_lsd'],
                'INPUT_FIELDS': [''],
                'OVERLAY': outputs['FieldCalculator']['OUTPUT'],
                'OVERLAY_FIELDS': [''],
                'OVERLAY_FIELDS_PREFIX': '',
                'OUTPUT': self.f+'/IntersectionA'+str(i)+'.gpkg'
            }
            outputs['IntersectionA'] = processing.run('native:intersection', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

            feedback.setCurrentStep(1)
            if feedback.isCanceled():
                return {}

            # Count points in polygon
            alg_params = {
                'CLASSFIELD': '',
                'FIELD': 'lip',
                'POINTS': parameters['lip'],
                'POLYGONS': outputs['IntersectionA']['OUTPUT'],
                'WEIGHT': '',
                'OUTPUT': self.f+'/CountPointsInPolygonA'+str(i)+'.gpkg'
            }
            outputs['CountPointsInPolygonA'] = processing.run('native:countpointsinpolygon', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

            feedback.setCurrentStep(2)
            if feedback.isCanceled():
                return {}

            # Add geometry attributes
            alg_params = {
                'CALC_METHOD': 0,  # Layer CRS
                'INPUT': outputs['CountPointsInPolygonA']['OUTPUT'],
                'OUTPUT': self.f+'/AddGeometryAttributesA'+str(i)+'.gpkg'
            }
            outputs['AddGeometryAttributesA'] = processing.run('qgis:exportaddgeometrycolumns', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

            alg_params={
                'INPUT': outputs['AddGeometryAttributesA']['OUTPUT'],
                #'OUTPUT': parameters['folder']+'/area_metric.csv'
            }
            outputs['A']=Functions.area_metric(alg_params)

            #################Density

            alg_params = {
            'CLASSFIELD': '',
            'FIELD': 'lip',
            'POINTS': parameters['lip'],
            'POLYGONS': outputs['FieldCalculator']['OUTPUT'],
            'WEIGHT': '',
            'OUTPUT': self.f+'/CountPointsInPolygon'+str(i)+'.gpkg'#QgsProcessing.TEMPORARY_OUTPUT
            }
            outputs['CountPointsInPolygon'] = processing.run('native:countpointsinpolygon', alg_params, context=context, feedback=feedback, is_child_algorithm=True)
            
            alg_params={
                'INPUT': outputs['CountPointsInPolygon']['OUTPUT'],
                #'OUTPUT': parameters['folder']+'/density_metric.csv'
            }
            outputs['D']=Functions.density_metric(alg_params)

            ##################Segmentation

            # Add geometry attributes
            alg_params = {
                'CALC_METHOD': 0,  # Layer CRS
                'INPUT': outputs['FieldCalculator']['OUTPUT'],
                'OUTPUT': self.f+'/AddGeometryAttributes'+str(i)+'.gpkg'
            }
            outputs['AddGeometryAttributes'] = processing.run('qgis:exportaddgeometrycolumns', alg_params, context=context, feedback=feedback, is_child_algorithm=True)
                        
            # Sin Zonal statistics
            alg_params = {
                'COLUMN_PREFIX': 'sin_',
                'INPUT': outputs['AddGeometryAttributes']['OUTPUT'],
                'INPUT_RASTER': outputs['SinRasterCalculator']['OUTPUT'],
                'RASTER_BAND': 1,
                'STATISTICS': [1],  # Sum
                'OUTPUT': self.f+'/SinZonalStatistics'+str(i)+'.gpkg'
            }
            outputs['SinZonalStatistics'] = processing.run('native:zonalstatisticsfb', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

            feedback.setCurrentStep(4)
            if feedback.isCanceled():
                return {}

            # Cos Zonal statistics
            alg_params = {
                'COLUMN_PREFIX': 'cos_',
                'INPUT': outputs['SinZonalStatistics']['OUTPUT'],
                'INPUT_RASTER': outputs['CosRasterCalculator']['OUTPUT'],
                'RASTER_BAND': 1,
                'STATISTICS': [1,0],  # Sum,Count
                'OUTPUT': self.f+'/zonalstat'+str(i)+'.gpkg'
            }
            outputs['CosZonalStatistics'] = processing.run('native:zonalstatisticsfb', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

            results['cv'] = outputs['CosZonalStatistics']['OUTPUT']


            alg_params = {
                'INPUT': self.f+'/zonalstat'+str(i)+'.gpkg',
                'METHOD': 1,  # Structure
                'OUTPUT': self.f+'/FixGeometriesZonalStatistics'+str(i)+'.gpkg'
            }
            outputs['FixGeometriesZonalStatistics'] = processing.run('native:fixgeometries', alg_params, context=context, feedback=feedback, is_child_algorithm=True)


            feedback.setCurrentStep(5)
            if feedback.isCanceled():
                return {}

            alg_params = {
                'INPUT':self.f+'/FixGeometriesZonalStatistics'+str(i)+'.gpkg',
            }
            outputs['gdp'],outputs['crs']=SZ_utils.load_geopackage(alg_params['INPUT'])

            feedback.setCurrentStep(7)
            if feedback.isCanceled():
                return {}
            
            ######################### V calculation

            alg_params = {
                'INPUT':outputs['gdp'],
                'FIELD':'area',
            }
            outputs['V']= Functions.V_calculator(alg_params)

            ########################### I calculation

            alg_params = {
                'INPUT':outputs['gdp'],
            }
            outputs['adj']=Functions.adjacent_matrix(alg_params)

            outputs['adj'].to_csv(self.f+'/adj'+str(i)+'.csv')

            ####################################################################

            # outputs['adj']=pd.read_csv('/tmp/SZ_7d2eq19g/adj0.csv')

            # alg_params = {
            #     'INPUT':'/tmp/SZ_7d2eq19g/FixGeometriesZonalStatistics0.gpkg',
            # }
            # outputs['gdp'],outputs['crs']=SZ_utils.load_geopackage(alg_params['INPUT'])

            ####################################################################
    
            alg_params = {
                'INPUT':outputs['gdp'],
                'INPUT1': outputs['adj'],
            }
            outputs['I']= Functions.I_calculator(alg_params)

            #print(ciao)

            table=table._append({'name': SU.name(),'V':outputs['V'],'I':outputs['I']}, ignore_index = True)
            final_table=final_table._append({'name': SU.name(),'A':outputs['A'],'D':outputs['D'],'F':[],'S':[]}, ignore_index = True)
        
        alg_params = {
                'INPUT':table,
                'OUTPUT':parameters['folder']+'/aspect_segmentation.csv'
            }
        outputs['F']= Functions.F_calculator(alg_params)

        final_table['F']=outputs['F']

        alg_params = {
                'INPUT':final_table,
            }
        outputs['S']= Functions.S_calculator(alg_params)
        outputs['S'].to_csv(parameters['folder']+'/segmentation_metric.csv')
        results['OUTPUT']=outputs['S']
        #results['OUTPUT']=outputs['gdp']

        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.4f} seconds")

        
        return results
    
class Functions():
    def load(parameters):
        layer = QgsVectorLayer(parameters['INPUT'], '', 'ogr')
        crs=layer.crs()
        gdp=gpd.read_file(parameters['INPUT'])
        return gdp,crs
    
    def V_calculator(parameters):
        df=parameters['INPUT']
        area=parameters['FIELD']
        cv=1-(np.sqrt(np.power(df['sin_sum'].to_numpy(),2)+np.power(df['cos_sum'].to_numpy(),2))/df['cos_count'].to_numpy())

        #if np.isnan(cv).any():
        #    raise ValueError("DEM too small")

        mask = ~pd.isnull(cv)  # mask where cv is not null

        V = np.sum(cv[mask] * df[area].to_numpy()[mask]) / np.sum(df[area].to_numpy()[mask])

        #V = np.sum(cv*df[area].to_numpy())/np.sum(df[area].to_numpy())
        print('V: ',V)
        return V

    # def adjacent_matrix(self,parameters):
    #     gdf_neighbors = lp.weights.Queen.from_dataframe(parameters['INPUT'])
    #     gdf_adj_mtx, gdf_adj_mtx_indices = gdf_neighbors.full()
    #     gdf_adj_list = gdf_neighbors.to_adjlist()
    #     return gdf_adj_list
    
    def adjacent_matrix_old(parameters):
        gdf = parameters['INPUT'].reset_index(drop=True)
        polygon_coords = []
        #gdf['geom']=gdf['geom'].apply(wkt.loads)

        for wkt_str in gdf['geom']:
            geom = wkt.loads(wkt_str)
            poly_vertices = set()
            for polygon in geom.geoms:
                poly_vertices.update(polygon.exterior.coords)
            polygon_coords.append(poly_vertices)


        # Step 2: Build adjacency list as DataFrame rows
        rows = []

        for i in range(len(polygon_coords)):
            for j in range(len(polygon_coords)):
                if i != j and polygon_coords[i] & polygon_coords[j]:
                    rows.append({'focal': i, 'neighbor': j})

        # Step 3: Create DataFrame
        df = pd.DataFrame(rows)

        return df

    def adjacent_matrix(parameters):
        gdf = parameters['INPUT'].reset_index(drop=True)
        geometries = [wkt.loads(wkt_str) for wkt_str in gdf['geom']]

        # Create spatial index and reverse map
        tree = STRtree(geometries)
        geom_index_map = {geom: i for i, geom in enumerate(geometries)}

        rows = []
        for i, geom in enumerate(geometries):
            neighbors = tree.query(geom)
            for neighbor in neighbors:
                    if neighbor is not None:
                        adj_geom=tree.geometries.take(neighbor)
                        if geom != adj_geom:
                            j = geom_index_map.get(adj_geom)
                            if geom.touches(adj_geom):
                                rows.append({'focal': i, 'neighbor': j})

        return pd.DataFrame(rows)



    
        # # Load geometries
        # geometries = [wkt.loads(w) for w in parameters['INPUT']['geom']]

        # # Spatial index
        # tree = STRtree(geometries)
        # geom_index = {id(g): i for i, g in enumerate(geometries)}

        # seen = set()
        # adjacency_list = []

        # for i, geom in enumerate(geometries):
        #     neighbors = tree.query(geom)
        #     print()
        #     for neighbor in neighbors:
        #         j = geom_index.get(id(neighbor))
        #         if j is None or i == j:
        #             continue
        #         if geom.touches(neighbor):
        #             pair = tuple(sorted((i, j)))
        #             if pair not in seen:
        #                 adjacency_list.append({'focal': i, 'neighbor': j})
        #                 seen.add(pair)
        # print(adjacency_list)
        # # Ensure DataFrame always has correct columns
        # return pd.DataFrame(adjacency_list, columns=['focal', 'neighbor'])
    
    def I_calculator(parameters):
        df=parameters['INPUT']
        adj=parameters['INPUT1']
        
        aspect_mean_SU = np.arctan(df['sin_sum'].to_numpy()/df['cos_sum'].to_numpy()) #aspect mean per SU (formula 4)
        aspect_mean_Stu = np.arctan(np.sum(df['sin_sum'].to_numpy())/np.sum(df['cos_sum'].to_numpy())) #aspect mean per study area (formula 4)
        teta_SU = np.arctan((np.sin(aspect_mean_SU)-np.sin(aspect_mean_Stu))/(np.cos(aspect_mean_SU)-np.cos(aspect_mean_Stu))) #teta per solope unit (formula 7-8)
        first_denom = np.sum(np.power(teta_SU,2)) #first argument of denominator in formula 2
        second_denom = len(adj) #second argument of denominator in formula 2
        denom = first_denom * second_denom # denominator of formula 2
        N = df.shape[0] # numberof SU
        aspect_product = np.cos(teta_SU[adj['focal']])*np.cos(teta_SU[adj['neighbor']])+np.sin(teta_SU[adj['focal']])*np.sin(teta_SU[adj['neighbor']]) # formula 6
        numerator = N*np.sum(aspect_product) # numerator of formula 2
        I = numerator/denom # formula 2
        print('I: ',I)
        return I
    
    def F_calculator(parameters):# F calculation from formula 3
        df=parameters['INPUT']
        v=df['V'].to_numpy(dtype=float)
        I=df['I'].to_numpy(dtype=float)
        F=((np.max(v)-v)/(np.max(v)-np.min(v)))+((np.max(I)-I)/(np.max(I)-np.min(I)))
        print(F,'F')
        df['F']=F
        df.to_csv(parameters['OUTPUT'])
        return df['F']
    
    def density_metric(parameters):
        # Continue only if CRS matches
        su=QgsVectorLayer(parameters['INPUT'],'','ogr')
        lip_sum=0
        count_above_zero=0
        for feat in su.getFeatures():
            lip_value = int(feat['lip'])
            lip_sum += lip_value
            if lip_value > 0:
                count_above_zero += 1
        D=1/(lip_sum/count_above_zero)
        #D=1/(su["lip"].sum()/len(su[su["point_count"]>0]))
        print('D: ',D)
        return(D)
    
    def area_metric(parameters):
        su=QgsVectorLayer(parameters['INPUT'],'','ogr')
        lsd_area_included=0
        lsd_area=0
        for feat in su.getFeatures():
            lip_value = int(feat['lip'])
            lsd_area += feat['area']
            if lip_value > 0:
                lsd_area_included += feat['area']
        A=lsd_area_included/lsd_area
        #D=1/(su["lip"].sum()/len(su[su["point_count"]>0]))
        print('A: ',A)
        return A

    def S_calculator(parameters):# F calculation from formula 3
        df=parameters['INPUT']
        a=df['A'].to_numpy(dtype=float)
        d=df['D'].to_numpy(dtype=float)
        f=df['F'].to_numpy(dtype=float)

        A=np.divide((a-np.min(a)), (a.max() - a.min()))
        D=np.divide((d-np.min(d)), (d.max() - d.min()))
        F=np.divide((f-np.min(f)), (f.max() - f.min()))

        #A=np.divide(np.diff(np.max(df['A'].to_numpy()),df['A'].to_numpy()),np.diff(np.max(df['A'].to_numpy()),np.min(df['A'].to_numpy())))
        #D=np.divide(np.diff(np.max(df['D'].to_numpy()),df['D'].to_numpy()),np.diff(np.max(df['D'].to_numpy()),np.min(df['D'].to_numpy())))
        #F=np.divide(np.diff(np.max(df['F'].to_numpy()),df['F'].to_numpy()),np.diff(np.max(df['F'].to_numpy()),np.min(df['F'].to_numpy())))

        S=np.multiply(A,np.multiply(D,F))
        df['S']=S
        print('S: ',S)
        return df
