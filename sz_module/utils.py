import subprocess
import os
from qgis.gui import QgsMessageBar
from qgis.core import Qgis
from qgis.utils import iface
import sys
sys.setrecursionlimit(10000)
import tempfile
from datetime import datetime
from qgis.core import QgsVectorLayer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from qgis.PyQt.QtCore import QVariant
from qgis.core import *
from qgis import *
# ##############################
import matplotlib.pyplot as plt
from processing.algs.gdal.GdalUtils import GdalUtils
import plotly.graph_objs as go
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

class first_installation():

    def requirements():
        dir=os.path.dirname(os.path.abspath(__file__))
        with open(dir+'/requirements.txt', "r") as file:
            for line in file:
                    parts=line.split("==")
                    try:
                        library=parts[0]
                        version=parts[1][:-1]
                    except:
                        library=parts[0][:-1]
                        version=None

                    print(library)

                # try:
                    installed_version=first_installation.get_package_version(library)
                    #exec(f"import {library}")
                    #installed_version=eval(f"{library}.__version__")
                    if installed_version is None:
                        #iface.messageBar().pushMessage('SZ:',f'{library} is installed!',Qgis.Success)
                        #MessageHandler.success(f'QGINLA: {library} is installed!')
                        #iface.messageBar().pushMessage("QGINLA:",f'installing {library}...',Qgis.Info, duration=5)
                        first_installation.install(library,version)
                    else:
                        print(installed_version,version)
                        if str(installed_version)==str(version):
                            iface.messageBar().pushMessage("SZ:",f'{library} is installed!',Qgis.Success)
                            #MessageHandler.success(f'QGINLA: {library} is installed!')
                        else:
                            iface.messageBar().pushMessage("SZ:",f'{library} is already installed but the actual version '+f'({installed_version}) is different than the required ({version}). It may cause errors!',Qgis.Warning)
                            #MessageHandler.warning(f'QGINLA: {library} is already installed but the actual version '+f'({installed_version}) is different than the required ({version}). It may cause errors!')

                    #except ImportError:
                    #    iface.messageBar().pushMessage("QGINLA:",f'installing {library}...',Qgis.Info, duration=5)
                    #    install(iface,library)

    def install(library, version=None):
        # Define the command you want to run
        print(library,version,'dai')
        if version is None:
            command = ['pip', 'install', library]
        else:
            command = ['pip', 'install', library+'=='+version]
        
        print(command)
        

        log_file = os.path.join(tempfile.gettempdir(), "SZ-logs.txt")

            
        # Add the current date and time to the log file
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"\nLog created on: {current_datetime}\n"

        if os.path.exists(log_file):
            # If the log file already exists, open it in append mode
            with open(log_file, 'a') as log:
                log.write(log_entry)
        else:
            # If the log file doesn't exist, create it and write the log entry
            with open(log_file, 'w') as log:
                log.write(log_entry)
        try:
            with open(log_file, 'a') as log:
                # Create a temporary log file
                subprocess.check_call(command, stderr=log, stdout=log)    
                iface.messageBar().pushMessage("SZ:", 'Dependencies installed successfully!', Qgis.Success)
        except subprocess.CalledProcessError:
            # Read the error message from the temporary log file
            with open(log_file, 'r') as log:
                error_message = log.read()

            log_link = f'{log_file}'

            # Construct the error message with the log file link
            error_message = f'Error occurred while installing dependencies. \n{error_message} \nread {log_link} for details.'
            iface.messageBar().pushMessage("SZ Log:", error_message, Qgis.Critical)
            #sys.exit('Error occurred while installing dependencies. Check the log for details.')
            sys.exit(error_message)

        # Remove the temporary log file
        os.remove(log_file)

    def get_package_version(package_name):
        try:
            # Use pip to get package information
            result = subprocess.check_output(['pip', 'show', package_name], universal_newlines=True)
            
            # Split the output into lines and find the line containing "Version"
            lines = result.strip().split('\n')
            for line in lines:
                if line.startswith("Version: "):
                    # Extract and return the version number
                    return line[len("Version: "):].strip()
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        return None
    

class SZ_utils():

    def load_simple(directory,parameters):
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
        gdp.to_csv(directory+'/file.csv')
        del gdp
        gdp=pd.read_csv(directory+'/file.csv')
        gdp['ID']=np.arange(1,len(gdp.iloc[:,0])+1)
        df=gdp[parameters['field1']]
        nomi=list(df.head())
        lsd=gdp[parameters['lsd']]
        lsd[lsd>0]=1
        df['y']=lsd#.astype(int)
        df['ID']=gdp['ID']
        df['geom']=gdp['geom']
        df=df.dropna(how='any',axis=0)
        X=[parameters['field1']]
        if parameters['testN']==0:
            train=df
            test=pd.DataFrame(columns=nomi,dtype=float)
        else:
            # split the data into train and test set
            per=int(np.ceil(df.shape[0]*parameters['testN']/100))
            train, test = train_test_split(df, test_size=per, random_state=42, shuffle=True)
        return train, test, nomi,crs
    

    def stampfit(parameters):
        df=parameters['df']
        y_true=df['y']
        scores=df['SI']
        ################################figure
        fpr1, tpr1, tresh1 = roc_curve(y_true,scores)
        norm=(scores-scores.min())/(scores.max()-scores.min())
        r=roc_auc_score(y_true, scores)

        fig=plt.figure()
        lw = 2
        plt.plot(fpr1, tpr1, color='green',lw=lw, label= 'Complete dataset (AUC = %0.2f)' %r)
        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        try:
            fig.savefig(parameters['OUT']+'/fig01.png')
        except:
            os.mkdir(parameters['OUT'])
            fig.savefig(parameters['OUT']+'/fig01.png')
    
    def stampcv(parameters):
        train=parameters['train']
        y_t=train['y']
        scores_t=train['SI']

        test=parameters['test']
        y_v=test['y']
        scores_v=test['SI']
        lw = 2
        
        fprv, tprv, treshv = roc_curve(y_v,scores_v)
        fprt, tprt, tresht = roc_curve(y_t,scores_t)

        aucv=roc_auc_score(y_v, scores_v)
        auct=roc_auc_score(y_t, scores_t)
        normt=(scores_t-scores_t.min())/(scores_t.max()-scores_t.min())
        normv=(scores_v-scores_v.min())/(scores_v.max()-scores_v.min())

        fig=plt.figure()
        plt.plot(fprv, tprv, color='green',lw=lw, label= 'Prediction performance (AUC = %0.2f)' %aucv)
        plt.plot(fprt, tprt, color='red',lw=lw, label= 'Success performance (AUC = %0.2f)' %auct)
        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        #plt.show()
        try:
            fig.savefig(parameters['OUT']+'/fig02.pdf')
        except:
            os.mkdir(parameters['OUT'])
            fig.savefig(parameters['OUT']+'/fig02.pdf')

    def save(parameters):

        df=parameters['df']
        nomi=list(df.head())
        fields = QgsFields()

        for field in nomi:
            if field=='ID':
                fields.append(QgsField(field, QVariant.Int))
            if field=='geom':
                continue
            if field=='y':
                fields.append(QgsField(field, QVariant.Int))
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
            print("Error when creating shapefile: ",  writer.errorMessage())
        for i, row in df.iterrows():
            fet = QgsFeature()
            fet.setGeometry(QgsGeometry.fromWkt(row['geom']))
            fet.setAttributes(list(map(float,list(df.loc[ i, df.columns != 'geom']))))
            writer.addFeature(fet)

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
