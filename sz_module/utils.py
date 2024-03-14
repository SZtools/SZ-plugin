from qgis.core import Qgis
from qgis.utils import iface
import sys
sys.setrecursionlimit(10000)
from qgis.core import *
from qgis import *

from qgis.core import Qgis, QgsMessageLog

def log(message):
    QgsMessageLog.logMessage(message, "SZ", level=Qgis.MessageLevel.Info)

def warn(message):
    QgsMessageLog.logMessage(message, "SZ", level=Qgis.MessageLevel.Warning)

#def icon(name):
#    return QIcon(os.path.join(os.path.dirname(__file__), "icons", name))

    


    
    