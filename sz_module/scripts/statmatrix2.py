"""
/***************************************************************************
    matrixAlgorithm
        begin                : 2021-11
        copyright            : (C) 2021 by Giacomo Titti,
                               Padova, November 2021
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
    matrixAlgorithm
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

from qgis.core import QgsProcessing
from qgis.core import QgsProcessingAlgorithm
from qgis.core import QgsProcessingMultiStepFeedback
from qgis.core import QgsProcessingParameterRasterLayer
from qgis.core import QgsProcessingParameterField
from qgis.core import QgsProcessingParameterVectorDestination
from qgis.core import QgsProcessingParameterFileDestination
from qgis.core import QgsRasterLayer
from qgis.core import QgsProcessingParameterExpression
import processing
#import plotly.express as px
import numpy as np
#import chart_studio.plotly as py
#import plotly.plotly as py
#import chart_studio.plotly as py
#import plotly.graph_objs as go
#import plotly.offline
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import QgsMessageLog
from qgis.core import Qgis
import pandas as pd
from osgeo import gdal,ogr
from PyQt5.QtCore import QFileInfo
import csv
#from processing.algs.gdal.GdalAlgorithm import GdalAlgorithm


class matrixAlgorithm(QgsProcessingAlgorithm):
    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer('lsi', 'classed lsi', defaultValue=None))
        self.addParameter(QgsProcessingParameterRasterLayer('r1', 'classed cause', defaultValue=None))
        self.addParameter(QgsProcessingParameterFileDestination('out', 'out statistics', '*.csv', defaultValue=None))
        #self.addParameter(QgsProcessingParameterRasterLayer(self.a, 'a', defaultValue=None))
        #self.addParameter(QgsProcessingParameterRasterLayer('r2', 'r2', defaultValue=None))
        #self.addParameter(QgsProcessingParameterRasterLayer('r3', 'r3', defaultValue=None))
        #self.addParameter(QgsProcessingParameterRasterLayer('r4', 'r4', defaultValue=None))

    def processAlgorithm(self, parameters, context, model_feedback):
        # Use a multi-step feedback, so that individual child algorithm progress reports are adjusted for the
        # overall progress through the model
        feedback = QgsProcessingMultiStepFeedback(1, model_feedback)
        results = {}
        outputs={}
        parameters['lsi'] = self.parameterAsRasterLayer(parameters, 'lsi', context).source()
        parameters['r1'] = self.parameterAsRasterLayer(parameters, 'r1', context).source()


        alg_params = {
            'INPUT': parameters['lsi']
        }
        outputs['Mlsi']=self.input(alg_params)
#        ##############################################
        # Input
        alg_params = {
            'INPUT': parameters['r1']
        }
        outputs['Mr1']=self.input(alg_params)
        #Stat
        alg_params = {
            'INPUT1': outputs['Mlsi'],
            'INPUT2': outputs['Mr1']
        }
        outputs['indlsi1']=self.stat(alg_params)
#
        # Frame
        alg_params = {
            'INPUT': outputs['indlsi1'],
            'OUTPUT': parameters['out']
        }
        self.frame(alg_params)
        return{}

    def input(self,parameters):
            self.ds2 = gdal.Open(parameters['INPUT'])
            a=self.ds2.GetRasterBand(1)
            NoData=a.GetNoDataValue()
            matrix=np.array([])
            matrix = np.array(a.ReadAsArray()).astype(float)
            self.xsize = self.ds2.RasterXSize
            self.ysize = self.ds2.RasterYSize
            gt=self.ds2.GetGeoTransform()
            w=gt[1]
            h=gt[5]
            xmin=gt[0]
            ymax=gt[3]
            xmax=xmin+(self.xsize*w)
            ymin=ymax+(self.ysize*h)
            return matrix

    def stat(self,parameters):
        lsim=parameters['INPUT1'].astype(int)
        m2=parameters['INPUT2'].astype(int)
        indm2={}
        indlsi={}
        row,col=lsim.shape
        no=np.where(lsim<=-9999)
        self.sizelsi=len(lsim[lsim>-9999])
        print(self.sizelsi,'size lsi')
        for i in range(1,lsim.max()+1,1):
            indm2={}
            for ii in range(1,m2.max()+1,1):
                countii=0
                ind=None
                ind=np.where((lsim==i) & (m2==ii))
                indm2[ii]=len(ind[0])
            indlsi[i]=indm2
        return indlsi

    def frame(self,parameters):
        d=parameters['INPUT']
        data=[(k, k1, v1) for k, v in list(d.items()) for k1, v1 in list(v.items())]
        with open(parameters['OUTPUT'],'w') as out:
            csv_out=csv.writer(out)
            csv_out.writerow(['total not None pixels = %.02f' %(self.sizelsi)])
            csv_out.writerow(['lsi','cause','length'])
            for row in data:
                csv_out.writerow(row)

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def name(self):
        return 'StatMatrix2'

    def displayName(self):
        return '07 Statistic Matrix'

    def group(self):
        return 'Raster analysis'

    def groupId(self):
        return 'Raster analysis'

    def shortHelpString(self):
        return self.tr("Classified matrixes comparison to estimate mutual percentage (Distance Matrix)")

    def createInstance(self):
        return matrixAlgorithm()
