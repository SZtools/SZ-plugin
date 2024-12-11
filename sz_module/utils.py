from qgis.core import Qgis
from qgis.utils import iface
import sys
sys.setrecursionlimit(10000)
from qgis.core import *
from qgis import *
import os
from qgis.core import Qgis, QgsMessageLog

def log(message):
    QgsMessageLog.logMessage(message, "SZ", level=Qgis.MessageLevel.Info)

def warn(message):
    QgsMessageLog.logMessage(message, "SZ", level=Qgis.MessageLevel.Warning)

#def icon(name):
#    return QIcon(os.path.join(os.path.dirname(__file__), "icons", name))


# Function to load environment variables from a .env file
def load_env_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            # Skip comments and empty lines
            if line.strip() and not line.strip().startswith('#'):
                # Split key-value pairs
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
  
def clean_memory():        
    for name in list(globals().keys()):
        if not name.startswith('__'):
            del globals()[name]
    