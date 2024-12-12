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
import os
from qgis.core import Qgis
import sys
sys.setrecursionlimit(10000)
from qgis.core import *
from qgis import *
from qgis.utils import iface
from qgis.PyQt.QtWidgets import QMessageBox
from qgis.core import Qgis, QgsMessageLog,QgsApplication
import traceback
import platform
import shutil
from ..utils import log
from .utils import (
    locate_py,
    add_venv,
    install_pip,
    pip_install_reqs,
    add_QGIS_env,
)
from qgis.PyQt.QtCore import QSettings

class installer():
    def __init__(self,version,plugin_settings):
        self.plugin_settings=plugin_settings
        self.plugin_module = os.path.basename(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
        self.plugin_venv = "."+self.plugin_module+version.replace('.', '')
        self._defered_packages = []
        self.plugins_path = os.path.join(
            QgsApplication.qgisSettingsDirPath(), "python", "plugins"
        )
        self.prefix_path = os.path.join(
            QgsApplication.qgisSettingsDirPath().replace("/", os.path.sep),
            "python",
            "dependencies",
        )
        self.qgis_python_interpreter = locate_py()
        self.venv_path = os.path.join(self.prefix_path,self.plugin_venv)
        self.site_packages_path=''
        self.bin_path='' 

    def is_already_installed(self):
        if self.plugin_settings is False:
            return False
        elif self.plugin_settings==str(self.version):
            return True
        elif self.plugin_settings!=str(self.version):
            return False
        
    def preliminay_req(self):
        try:
            add_venv(self.prefix_path,self.venv_path,self.plugin_venv,self.qgis_python_interpreter)
        except Exception as e:
            log(f"An error occurred: {e}")
            return False
        try:
            self.site_packages_path, self.bin_path=add_QGIS_env(self.prefix_path,self.plugin_venv)
        except Exception as e:
            log(f"An error occurred: {e}")
            return False
        try:
            try:
                #windows
                command=install_pip(['ensurepip'],os.path.join(self.venv_path,"Scripts","pythonw.exe"))
            except Exception:
                #linux and macos
                command=install_pip(['ensurepip'],os.path.join(self.venv_path,"bin","python")) 
        except Exception as e:
            log(f"An error occurred: {e}")
            return False
           
    def requirements(self):
        dir=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
        log(f"verify requirements")
        if platform.system() == 'Windows':
            req="requirements.txt"
        else:
            req="requirements_linux.txt"
        with open(os.path.join(dir,req), "r") as file:
            list_libraries={}
            for line in file:
                    parts=line.split("==")
                    try:
                        library=parts[0]
                        version=parts[1][:-1]
                    except:
                        library=parts[0][:-1]
                        version=None
                    list_libraries[library]=version
        return self.install(list_libraries)

    def install(self,list_libraries):
            if len(list_libraries.keys())>0:
                reqs_to_install = [f"{library}=={version}" if version else library for library, version in list_libraries.items()]
                if QMessageBox.question(None, f"{os.getenv('PLUGIN_NAME')} for Processing Python dependencies not installed",
                    f"Do you automatically want install missing python modules {reqs_to_install}? \r\n"
                    "QGIS will be non-responsive for a couple of minutes.",
                    QMessageBox.Ok | QMessageBox.Cancel) == QMessageBox.Ok:
                    try:
                        log(f"Will install selected dependencies : {reqs_to_install}")
                        try:
                            #windows
                            command=pip_install_reqs(self.prefix_path,self.plugin_venv,reqs_to_install,os.path.join(self.venv_path,"Scripts","pythonw.exe"))
                        except:
                            #linux and macos
                            command=pip_install_reqs(self.prefix_path,self.plugin_venv,reqs_to_install,os.path.join(self.venv_path,"bin","python"))
                        QMessageBox.information(None, "Packages successfully installed",
                                                #"To make all parts of the plugin work it is recommended to restart your QGIS-session.")
                                                f"To make all parts of the plugin work it is recommended to restart your QGIS-session. You can find the {os.getenv('PLUGIN_NAME')}-plugin in the Processing-toolbox")
                    except Exception as e:
                        QgsMessageLog.logMessage(traceback.format_exc(), level=Qgis.Warning)
                        QMessageBox.information(None, "An error occurred",
                                                f"{os.getenv('PLUGIN_NAME')} couldn't install Python packages!\n"
                                                "See 'General' tab in 'Log Messages' panel for details.\n"
                                                "Report any errors to https://github.com/SZtools/SZ/issues")
                        log(f"An error occurred:{e}")
                        return False
                else:
                    QMessageBox.information(None,"Information", f"Packages not installed. Some {os.getenv('PLUGIN_NAME')} tools will not be fully operational.")
                    sys.path_importer_cache.clear()
                    log(f"Packages not installed. Some {os.getenv('PLUGIN_NAME')} tools will not be fully operational.")
                    return False
                
                sys.path_importer_cache.clear()

    def unload(self):
            # Remove path alterations
            if self.site_packages_path in sys.path:
                sys.path.remove(self.site_packages_path)
                os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"].replace(
                    self.bin_path + ";", ""
                )
                os.environ["PATH"] = os.environ["PATH"].replace(self.bin_path + ";", "")
            try:
                # Attempt to delete the folder and its contents using shutil
                shutil.rmtree(self.venv_path)
                print(f"Folder '{self.venv_path}' and its contents deleted successfully.")
                log(f"Folder '{self.venv_path}' and its contents deleted successfully.")
            except PermissionError:
                # If permission error occurs, try using os module with elevated privileges
                try:
                    if platform.system() == 'Windows':
                        os.system(f'rmdir /s /q "{self.venv_path}"')
                        log(f"Folder '{self.venv_path}' and its contents deleted successfully.")
                    else:
                        os.system(f'sudo rm -rf "{self.venv_path}"')
                        log(f"Folder '{self.venv_path}' and its contents deleted successfully.")
                except Exception as e:
                    print(f"Error deleting folder '{self.venv_path}': {e}")
                    log(f"Error deleting folder '{self.venv_path}': {e}")
            QSettings().remove("SZ")
            print('unloaded')
