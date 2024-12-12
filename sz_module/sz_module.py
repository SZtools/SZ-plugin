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
import inspect
from qgis.core import QgsApplication
from qgis.PyQt.QtCore import QSettings
from .installer.installer import installer
from .utils import log,warn,load_env_file

cmd_folder = os.path.split(inspect.getfile(inspect.currentframe()))[0]

if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

class classePlugin(object):

    def __init__(self):
        #self.settings = QgsSettings()
        #self.settings.beginGroup("SZ")
        self.provider = None
        dir=(os.path.dirname(os.path.abspath(__file__)))
        load_env_file(os.path.join(dir, ".env"))
        with open(dir+'/metadata.txt','r') as file:
            for line in file:
                if line.startswith('version='):
                    self.version = line.strip().split('version=')[1].strip()
        self.plugin_settings = QSettings().value("SZ",False)

    def initProcessing(self):
        from .sz_module_provider import classeProvider
        """Init Processing provider for QGIS >= 3.8."""
        self.provider = classeProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.installer=installer(self.version,self.plugin_settings)
        print('Plugin already installed? ',self.plugin_settings)
        if os.environ.get('DEBUG')=='False':
            if self.installer.preliminay_req() is False:
                self.installer.unload()
                log(f"An error occured during the installation")
                raise RuntimeError("An error occured during the installation")
            else:
                if self.installer.is_already_installed(self.version) is False:
                    if self.installer.requirements() is False:
                        self.installer.unload()
                        log(f"An error occured during the installation")
                        raise RuntimeError("An error occured during the installation")
                    else:
                        QSettings().setValue("SZ", str(self.version))
                        self.initProcessing() 
                else:  
                    self.initProcessing()  
        else:
            self.initProcessing()  

    def unload(self):
        QgsApplication.processingRegistry().removeProvider(self.provider)     