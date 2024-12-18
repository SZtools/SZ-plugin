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

from qgis.core import Qgis
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