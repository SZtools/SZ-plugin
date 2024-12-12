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
import subprocess
from subprocess import (
    PIPE,
    STDOUT,
    Popen,
)
from typing import List, Union
from pkg_resources import ResolutionError
from qgis.core import Qgis, QgsApplication
from qgis.utils import iface
from ..utils import log,warn
import os
import platform
import subprocess
import sys
import platform
from pathlib import Path
from packaging import version
import glob

if platform.system() == "Windows":
    from subprocess import (
        STARTF_USESHOWWINDOW,
        STARTF_USESTDHANDLES,
        STARTUPINFO,
        SW_HIDE,
    )

def run_cmd(args, description="sz-plugin load...."):
    log(f'command:{args}')
    startupinfo = None
    if os.name == "nt":
        startupinfo = STARTUPINFO()
        startupinfo.dwFlags |= STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = SW_HIDE
    process = Popen(args, stdout=PIPE, stderr=STDOUT, startupinfo=startupinfo)
    full_output = ""
    while True:
        QgsApplication.processEvents()
        try:
            # FIXME : this doesn't seem to timeout
            out, _ = process.communicate(timeout=0.1)
            output = out.decode(errors="replace").strip()
            full_output += output
            if output:
                log(output)
        except subprocess.TimeoutExpired:
            pass
        if process.poll() is not None:
            break

    if process.returncode != 0:
        warn(f"Command failed.")
    else:
        log("Command succeeded.")
        iface.messageBar().pushMessage(
            "Success",
            f"{description.capitalize()} succeeded",
            level=Qgis.Success,
        )

def locate_py():
        # get Python version
        str_ver_qgis = sys.version.split(" ")[0]
        try:
            # non-Linux
            path_py = os.environ["PYTHONHOME"]
        except Exception:
            # Linux
            path_py = sys.executable
        # convert to Path for eaiser processing
        path_py = Path(path_py)     
        # pre-defined paths for python executable
        if platform.system() == "Windows":
            candidates = [
                path_py
                / (
                    "../../bin/pythonw.exe" if version.parse(str_ver_qgis) >= version.parse("3.9.1")
                    else "pythonw.exe"
                ),
                path_py.with_name("pythonw.exe"),
            ]
        else:
            candidates = [
                path_py / "bin" / "python3",
                path_py / "bin" / "python",
                path_py.with_name("python3"),
                path_py.with_name("python"),
            ]
        for candidate_path in candidates:
            if candidate_path.exists():
                log(f"Python interpreter is located in {candidate_path}")
                return candidate_path

def add_venv(prefix_path,venv_path,plugin_venv,interpreter="python"):
    """
    Installs given reqs with pip
    """
    os.makedirs(prefix_path, exist_ok=True)
    if os.path.exists(venv_path) and os.path.isdir(venv_path):
        log(f"{plugin_venv} already exist in {os.path.join(prefix_path, plugin_venv)} with {interpreter}")
        pass
    else:
        log(f"creating {plugin_venv} in {os.path.join(prefix_path, plugin_venv)} with {interpreter}")
        run_cmd(
            [
                interpreter,
                "-m",
                "venv",                
                os.path.join(prefix_path, plugin_venv)
            ],
            f"creating a dedicated virtual-environment",
        )

def add_QGIS_env(prefix_path,plugin_venv):
    if platform.system() == 'Windows':
        site_packages_path = os.path.join(prefix_path,plugin_venv,"Lib", "site-packages")
        bin_path = os.path.join(prefix_path,plugin_venv,"Scripts")
    else:
        search_pattern = os.path.join(prefix_path,plugin_venv, "lib", "python*", "site-packages")
        site_packages_path = glob.glob(search_pattern)[0]
        bin_path = os.path.join(prefix_path,plugin_venv,"bin")

        search_pattern = os.path.join(prefix_path,plugin_venv, "lib64", "python*", "site-packages")
        site_packages_path1 = glob.glob(search_pattern)[0]
        bin_path = os.path.join(prefix_path,plugin_venv,"bin")

    if site_packages_path not in sys.path:
        if platform.system() == 'Windows': 
            log(f"Adding {site_packages_path} to PYTHONPATH")
            sys.path.insert(0, site_packages_path)
            os.environ["PYTHONPATH"] = (
                site_packages_path + ";" + os.environ.get("PYTHONPATH", "")
            )
        else:
            log(f"Adding {site_packages_path} to PYTHONPATH")
            sys.path.insert(0, site_packages_path)
            os.environ["PYTHONPATH"] = (
                site_packages_path + ";" + os.environ.get("PYTHONPATH", "")
            )
            
            log(f"Adding {site_packages_path1} to PYTHONPATH")
            sys.path.insert(0, site_packages_path1)
            os.environ["PYTHONPATH"] = (
                site_packages_path1 + ";" + os.environ.get("PYTHONPATH", "")
            )
    
    if bin_path not in os.environ["PATH"]:
        log(f"Adding {bin_path} to PATH")
        os.environ["PATH"] = bin_path + ";" + os.environ["PATH"]  
    return (site_packages_path, bin_path)

def install_pip(reqs_to_install,interpreter="python"):
    #Installs given reqs with pip
    log(f"Will install {reqs_to_install} with {interpreter}")
    cmd=[
            interpreter,
            "-m",
            *reqs_to_install
        ]
    run_cmd(cmd)
    return(cmd)

def pip_uninstall_reqs(reqs_to_uninstall, extra_args=[],interpreter="python pip -m"):
    #Unnstalls given deps with pip
    log(f"Will pip uninstall {reqs_to_uninstall}")
    cmd=[
            interpreter,
            #"-um",
            #"pip",
            "uninstall",
            "-y",
            *reqs_to_uninstall,
        ],   
    run_cmd(cmd,f"installing {len(reqs_to_uninstall)} requirements: {reqs_to_uninstall}")
    return(cmd)

def pip_install_reqs(prefix_path,plugin_venv,reqs_to_install,interpreter="python pip -m"):
    #Installs given reqs with pip
    log(f"Will pip install {reqs_to_install} in {os.path.join(prefix_path, plugin_venv)} with {interpreter}")
    cmd=[
            interpreter,
            "-m",
            "pip",
            "install",
            *reqs_to_install,
        ]
    run_cmd(cmd,f"installing {len(reqs_to_install)} requirements: {reqs_to_install}")
    return(cmd)

def get_package_version(qgis_python_interpreter,package_name):
        try:
            # Use pip to get package information
            result = subprocess.check_output([qgis_python_interpreter,'-m','pip', 'show', package_name], universal_newlines=True)
            # Split the output into lines and find the line containing "Version"
            lines = result.strip().split('\n')
            for line in lines:
                if line.startswith("Version: "):
                    # Extract and return the version number
                    return line[len("Version: "):].strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        return None