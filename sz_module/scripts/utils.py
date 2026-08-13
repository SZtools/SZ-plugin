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

import matplotlib.pyplot as plt
import fiona
import numpy as np
import os
import pandas as pd
import sqlite3
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, cohen_kappa_score
from scipy.stats import pearsonr
from shapely.geometry import shape
from shapely.wkt import dumps
from qgis.core import (QgsVectorLayer,
                       QgsFields,
                       QgsField,
                       QgsProject,
                       QgsVectorFileWriter,
                       QgsFeature,
                       QgsGeometry,
                       QgsProcessingContext,
                       QgsProcessingUtils,
)
from qgis.PyQt.QtCore import QVariant

class SZ_utils():
    def generate_ghost_input(input,output):
        input_shapefile_path = input
        layer = QgsVectorLayer(input_shapefile_path, 'Input Layer', 'ogr')
        transform_context = QgsProject.instance().transformContext()
        save_options = QgsVectorFileWriter.SaveVectorOptions()
        save_options.driverName = 'GPKG'
        save_options.fileEncoding = 'UTF-8'
        writer = QgsVectorFileWriter.writeAsVectorFormat(
          layer,  
          output,
          save_options
        )
        print(output)
        del writer
        del layer

    def load_geopackage(file_path, table_name='file'):
        print('loading dataframe')

        layer = QgsVectorLayer(file_path, 'Input Layer', 'ogr')
        crs = layer.crs()

        records = []

        with fiona.open(file_path) as source:
            for i, feature in enumerate(source):
                try:
                    properties = dict(feature['properties'])

                    if feature['geometry'] is None:
                        print(f"Skipping feature {i}: null geometry")
                        continue

                    geom = shape(feature['geometry'])

                    if geom is None or geom.is_empty:
                        print(f"Skipping feature {i}: empty geometry")
                        continue

                    try:
                        from shapely import force_2d
                        geom = force_2d(geom)
                    except ImportError:
                        from shapely.ops import transform

                        def _to_2d(x, y, z=None):
                            return (x, y)

                        geom = transform(_to_2d, geom)

                    # Important: force the WKT writer itself to output 2D
                    geom_wkt = dumps(geom, output_dimension=2)

                    properties['geom'] = geom_wkt
                    records.append(properties)

                except Exception as e:
                    print("Bad feature index:", i)
                    print("Feature id:", feature.get("id"))
                    print("Geometry:", feature.get("geometry"))
                    raise e

        df = pd.DataFrame(records)

        del layer
        del records

        return df, crs

    def get_id_column(file_path, table_name='file'):
        conn = sqlite3.connect(file_path)
        columns_info = pd.read_sql_query(f"PRAGMA table_info({table_name});", conn)
        conn.close()
        id_column = columns_info[columns_info['pk'] > 0]['name']
        if not id_column.empty:
            return id_column.values[0]
        else:
            return None

    def load_cv(directory,parameters):
        temporary_input = QgsProcessingUtils.generateTempFilename('sz_input.gpkg')
        SZ_utils.generate_ghost_input(
            parameters['INPUT_VECTOR_LAYER'],
            temporary_input,
        )
        gdp,crs=SZ_utils.load_geopackage(temporary_input)
        if 'time' in parameters:
            if parameters['time']==None:
                df=pd.DataFrame(gdp[parameters['nomi']].copy())
            else:
                df=pd.DataFrame(gdp[parameters['nomi']+[parameters['time']]].copy())
        else:
            df=pd.DataFrame(gdp[parameters['nomi']].copy())
        try:
            lsd=gdp[parameters['lsd']]
            if parameters['family']=='gaussian' and parameters['scale']=='log_scale':
                lsd[lsd>0]=np.log(lsd[lsd>0])
            elif parameters['family']=='gaussian' and parameters['scale']=='linear_scale':
                print('do not scale target')
            elif parameters['family']=='MLP_regressor' and parameters['scale']=='log_scale':
                lsd[lsd>0]=np.log(lsd[lsd>0])
            elif parameters['family']=='MLP_regressor' and parameters['scale']=='linear_scale':
                print('do not scale target')
            elif parameters['family']=='SVM_regressor' or parameters['family']=='DT_regressor' or parameters['family']=='RF_regressor':
                print('do not scale target')
            else:
                lsd[lsd>0]=1
            df['y']=lsd#.astype(int)
        except KeyError:
            print('no target required')
        df['ID']=gdp.index
        df['geom']=gdp['geom']
        print('input layer loaded:', list(df.columns))
        del gdp
        return(df,crs)

    def stampfit(parameters):
        print('plotting....')
        df=parameters['df']
        y_true = df["y"].to_numpy()
        y_true = (y_true > 0).astype(int)
        scores=df['SI'].to_numpy(dtype=float)
        ################################figure
        fpr1, tpr1, tresh1 = roc_curve(y_true,scores)
        j = tpr1 - fpr1
        idx = int(np.argmax(j))
        best_thr = float(tresh1[idx])# x YOUDEN INDEX
        r=roc_auc_score(y_true, scores)
        print(r,'AUC')
        y_pred = (scores > best_thr).astype(int)
        # Extra metrics
        f1_tot = f1_score(y_true, y_pred)
        ck_tot = cohen_kappa_score(y_true, y_pred)
        fig=plt.figure()
        lw = 2
        plt.plot(fpr1, tpr1,color="green",lw=lw,label=f"AUC = {r:0.2f}, F1 = {f1_tot:0.2f}, K = {ck_tot:0.2f}\nThr = {best_thr:0.4g}")
        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        os.makedirs(parameters['OUT'], exist_ok=True)
        fig.savefig(os.path.join(parameters['OUT'], 'fig_fit.png'))

    def stamp_cv(parameters):
        print('plotting....')
        df = parameters["df"].dropna(subset=["SI"]).reset_index(drop=True)
        df=df.dropna(subset=['SI'])
        test_ind=parameters['test_ind']
        y_v=df["y"].to_numpy()
        y_v = (y_v > 0).astype(int)
        scores_v=df['SI'].to_numpy(dtype=float)
        lw = 2
        ################################figure
        fig=plt.figure()
        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')

        for i in range(len(test_ind)):

            yt = y_v[test_ind[i]]
            st = scores_v[test_ind[i]]

            fprv, tprv, treshv = roc_curve(yt, st)
            aucv = roc_auc_score(yt, st)
            print("ROC", str(i), "AUC=", aucv)

            j = tprv - fprv
            best_idx = int(np.argmax(j))
            best_thr = float(treshv[best_idx])

            y_pred = (st > best_thr).astype(int)
            f1_tot = f1_score(yt, y_pred)
            ck_tot = cohen_kappa_score(yt, y_pred)
            print("F1=", f1_tot, "K=", ck_tot, "Thr=", best_thr)
            plt.plot(fprv, tprv,lw=lw, alpha=0.5, label='ROC fold '+str(i+1)+' AUC = %0.2f, F1 = %0.2f, K = %0.2f\nThr = %0.4g' %(aucv, f1_tot,ck_tot, best_thr))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        print('ROC curve figure = ',os.path.join(parameters['OUT'], 'fig_cv.pdf'))
        os.makedirs(parameters['OUT'], exist_ok=True)
        fig.savefig(os.path.join(parameters['OUT'], 'fig_cv.pdf'))

    def stamp_qq(parameters):
        print('plotting....')
        df=parameters['df']
        df=df.dropna(subset=['SI'])
        test_ind=parameters['test_ind']
        y_v=df['y']
        scores_v=df['SI']

        fig, ax = plt.subplots(figsize=(11, 6))
        M=[]
        m=[]
        for i in range(len(test_ind)):
            df_train=y_v[test_ind[i]]
            df_trans=scores_v[test_ind[i]]
            errors=SZ_utils.errors(df_train,df_trans)
            # Compute percentiles
            percentiles_train = np.percentile(df_train, np.arange(0, 101, 1))
            percentiles_trans = np.percentile(df_trans, np.arange(0, 101, 1))
            plt.scatter(percentiles_train, percentiles_trans,marker='o',s=3, alpha=0.6, label='QQ '+str(i+1)+' MAE = %0.2f, RMSE = %0.2f, R2 = %0.2f, PCC = %0.2f' %(errors[0], errors[1],errors[2],errors[3]))
            M.append(max(max(percentiles_train),max(percentiles_trans)))
            m.append(min(min(percentiles_train),min(percentiles_trans)))
        MM=max(M)
        mm=min(m)
        plt.plot([mm, MM], [mm, MM], color='black', lw=2, linestyle='--')
        ax.set_aspect('equal','box')
        plt.xlabel('Observed')
        plt.ylabel('Predicted')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize='small')
        plt.tight_layout()
        print('QQ figure = ',os.path.join(parameters['OUT'], 'fig_qq.pdf'))
        os.makedirs(parameters['OUT'], exist_ok=True)
        fig.savefig(os.path.join(parameters['OUT'], 'fig_qq.pdf'))

    def stamp_qq_fit(parameters):
        print('plotting....')
        df=parameters['df'].dropna(how='any',axis=0)
        df_true=df['y']
        df_predict=df['SI']
        errors=SZ_utils.errors(df_true,df_predict)
        percentiles_train = np.percentile(df_true, np.arange(0, 101, 1))
        percentiles_trans = np.percentile(df_predict, np.arange(0, 101, 1))
        fig, ax = plt.subplots()
        plt.scatter(percentiles_train, percentiles_trans,marker='o',s=3, alpha=0.5, label='QQ MAE = %0.2f, RMSE = %0.2f, R2 = %0.2f, PearsCorr = %0.2f' %(errors[0], errors[1],errors[2],errors[3]))
        MM=np.max([np.max(percentiles_train),np.max(percentiles_trans)])
        mm=np.min([np.min(percentiles_train),np.min(percentiles_trans)])
        plt.plot([mm, MM], [mm, MM], color='black', lw=2, linestyle='--')
        ax.set_aspect('equal','box')
        plt.xlabel('Observed')
        plt.ylabel('Predicted')
        plt.legend(bbox_to_anchor =(0.5,-0.3), loc='lower center',fontsize='small')
        plt.tight_layout()
        print('QQ figure = ',os.path.join(parameters['OUT'], 'fig_qq_fit.pdf'))
        os.makedirs(parameters['OUT'], exist_ok=True)
        fig.savefig(os.path.join(parameters['OUT'], 'fig_qq_fit.pdf'))

    def save(parameters):
        print('Writing output GeoPackage...')

        df = parameters['df']
        crs = parameters['crs']
        output_path = parameters['OUT']

        uri = f"Polygon?crs={crs.authid()}"
        layer = QgsVectorLayer(uri, "temp_layer", "memory")
        pr = layer.dataProvider()

        # Step 2: Define and add fields
        fields = QgsFields()
        for field in df.columns:
            if field == 'geom':
                continue
            elif field == 'ID':
                fields.append(QgsField(field, QVariant.Int))
            #elif field == 'iid':################Perla
                #fields.append(QgsField(field, QVariant.Int))###############Perla
            else:
                fields.append(QgsField(field, QVariant.Double))

        pr.addAttributes(fields)
        layer.updateFields()

        # Step 3: Add features
        feats = []
        for i, row in df.iterrows():
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromWkt(row['geom']))
            attr = [row[col] for col in df.columns if col != 'geom']
            feat.setAttributes(attr)
            feats.append(feat)

        pr.addFeatures(feats)
        layer.updateExtents()

        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.fileEncoding = "UTF-8"

        transform_context = QgsProject.instance().transformContext()
        error, error_string = QgsVectorFileWriter.writeAsVectorFormatV2(
            layer,
            output_path,
            transform_context,
            options
        )

        if error != QgsVectorFileWriter.NoError:
            print("Failed to write GPKG:", error_string)
        else:
            print("Saved with CRS:", layer.crs().authid())

    def addmap(parameters):
        context=parameters()
        fileName = parameters['trainout']
        layer = QgsVectorLayer(fileName,"train","ogr")
        subLayers =layer.dataProvider().subLayers()

        for subLayer in subLayers:
            name = subLayer.split('!!::!!')[1]
            uri = "%s|layername=%s" % (fileName, name,)
            # Create layer
            sub_vlayer = QgsVectorLayer(uri, name, 'ogr')
            if not sub_vlayer.isValid():
                print('layer failed to load')
            # Add layer to map
            context.temporaryLayerStore().addMapLayer(sub_vlayer)
            context.addLayerToLoadOnCompletion(sub_vlayer.id(), QgsProcessingContext.LayerDetails('layer', context.project(),'LAYER'))

    def errors(y_true,predict):
        min_absolute_error = mean_absolute_error(y_true, predict)
        rmse = np.sqrt(mean_squared_error(y_true, predict))
        r_squared = r2_score(y_true, predict)
        pearson_coefficient, _ = pearsonr(y_true, predict)
        errors=[min_absolute_error,rmse,r_squared,pearson_coefficient]
        return(errors)
    
    def check_validity(parameters):
        for i in parameters['tensor']:
            if i not in parameters['linear']+parameters['continuous']+parameters['categorical']:
                continue
            else:
                return False
        return True
    
    def make_directory(parameters):
        if not os.path.exists(parameters['path']):
            os.mkdir(parameters['path'])
