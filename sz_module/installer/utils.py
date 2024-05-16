import os
import subprocess
from subprocess import (
    PIPE,
    STARTF_USESHOWWINDOW,
    STARTF_USESTDHANDLES,
    STARTUPINFO,
    STDOUT,
    SW_HIDE,
    Popen,
)
from typing import List, Union

from pkg_resources import ResolutionError
from qgis.core import Qgis, QgsApplication
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QProgressDialog
from qgis.utils import iface
from ..utils import log,warn

import os
import platform
import subprocess
import sys
import platform
from pathlib import Path
from packaging import version


def run_cmd(args, description="Installing...."):
    log(f'command:{args}')

    # progress_dlg = QProgressDialog(
    #     description, "Abort", 0, 0, parent=iface.mainWindow()
    # )
    # progress_dlg.setWindowModality(Qt.WindowModal)

    
    #progress_dlg.show()

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
                #progress_dlg.setLabelText(output)
                log(output)
        except subprocess.TimeoutExpired:
            pass

        #if progress_dlg.wasCanceled():
        #    process.kill()
        if process.poll() is not None:
            break

    #progress_dlg.close()

    if process.returncode != 0:
        warn(f"Command failed.")
        # message = QMessageBox(
        #     QMessageBox.Warning,
        #     "Command failed",
        #     f"Encountered an error while {description} !",
        #     parent=iface.mainWindow(),
        # )
        # message.setDetailedText(full_output)
        # message.exec_()
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

def install_pip(reqs_to_install,interpreter="python"):
    """
    Installs given reqs with pip
    """
    log(f"Will install {reqs_to_install} with {interpreter}")
    cmd=[
            interpreter,
            "-m",
            *reqs_to_install
        ]
    run_cmd(cmd)#,f"installing {len(reqs_to_install)} requirements: {reqs_to_install}")
    return(cmd)

def pip_uninstall_reqs(reqs_to_uninstall, extra_args=[],interpreter="python pip -m"):
    """
    Unnstalls given deps with pip
    """
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
    """
    Installs given reqs with pip
    """
    log(f"Will pip install {reqs_to_install} in {os.path.join(prefix_path, plugin_venv)} with {interpreter}")
    cmd=[
            interpreter,
            "-m",
            "pip",
            "install",
            *reqs_to_install,
            #"-U",
            #"--prefer-binary",
            #"--user"
            #"--prefix",
            #prefix_path,
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
    
