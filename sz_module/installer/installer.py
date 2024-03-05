import os
import platform
import subprocess
import sys
from collections import defaultdict, namedtuple
from importlib import metadata

import pkg_resources
import qgis
from pkg_resources import DistributionNotFound, VersionConflict
from qgis.core import QgsApplication, QgsSettings
from qgis.PyQt.QtWidgets import QAction

from pathlib import Path
from packaging import version

from .utils import Lib, Req, icon, log, run_cmd

import subprocess
from subprocess import Popen,PIPE
import os
from qgis.core import Qgis
import sys
sys.setrecursionlimit(10000)
import tempfile
from datetime import datetime
from qgis.core import *
from qgis import *
from qgis.utils import iface

from qgis.PyQt.QtWidgets import QMessageBox
from qgis.core import Qgis, QgsMessageLog
import traceback






MissingDep = namedtuple("MissingDep", ["package", "requirement", "state"])

class installer():
    def __init__(self):
        self._defered_packages = []
        self.plugins_path = os.path.join(
            QgsApplication.qgisSettingsDirPath(), "python", "plugins"
        )
        self.prefix_path = os.path.join(
            QgsApplication.qgisSettingsDirPath().replace("/", os.path.sep),
            "python",
            "dependencies",
        )

        self.qgis_python_interpreter = self.locate_py()

        self.venv_path = os.path.join(self.prefix_path,".venv")
        self.site_packages_path = os.path.join(self.prefix_path,".venv","Lib", "site-packages")
        self.bin_path = os.path.join(self.prefix_path,".venv","Scripts")

        self.add_venv(self.qgis_python_interpreter)

        try:
            #windows
            #self.uninstall_pip(['pip'],os.path.join(self.venv_path,"Scripts","python"))
            command=self.install_pip(['ensurepip'],os.path.join(self.venv_path,"Scripts","pythonw.exe"))
        except Exception:
            #linux and macos
            #self.uninstall_pip(['pip'],os.path.join(self.venv_path,"bin","python"))
            command=self.install_pip(['ensurepip'],os.path.join(self.venv_path,"bin","python"))
        
        if self.site_packages_path not in sys.path:
            log(f"Adding {self.site_packages_path} to PYTHONPATH")
            sys.path.insert(0, self.site_packages_path)
            os.environ["PYTHONPATH"] = (
                self.site_packages_path + ";" + os.environ.get("PYTHONPATH", "")
            )

        if self.bin_path not in os.environ["PATH"]:
            log(f"Adding {self.bin_path} to PATH")
            os.environ["PATH"] = self.bin_path + ";" + os.environ["PATH"]

        self.requirements()
        

    def locate_py(self):

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
            print(candidates,'c')
        else:
            candidates = [
                path_py / "bin" / "python3",
                path_py / "bin" / "python",
                path_py.with_name("python3"),
                path_py.with_name("python"),
            ]

        for candidate_path in candidates:
            if candidate_path.exists():
                return candidate_path


    def add_venv(self,interpreter="python"):
        """
        Installs given reqs with pip
        """
        os.makedirs(self.prefix_path, exist_ok=True)
        if os.path.exists(self.venv_path) and os.path.isdir(self.venv_path):
            pass
        else:
            log(f"creating .venv")
            run_cmd(
                [
                    interpreter,
                    "-m",
                    "venv",                
                    os.path.join(self.prefix_path, ".venv")
                ],
                f"creating venv",
            )

    def install_pip(self, reqs_to_install,interpreter="python"):
        """
        Installs given reqs with pip
        """
        log(f"Will install {reqs_to_install}")

        cmd=[
                interpreter,
                "-m",
                *reqs_to_install
            ]
            
        run_cmd(cmd,f"installing {len(reqs_to_install)} requirements")
        return(cmd)

    def pip_uninstall_reqs(self, reqs_to_uninstall, extra_args=[],interpreter="python pip -m"):
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
        run_cmd(cmd,f"uninstalling {len(reqs_to_uninstall)} requirements")
        return(cmd)

    def pip_install_reqs(self, reqs_to_install,interpreter="python pip -m"):
        """
        Installs given reqs with pip
        """
        log(f"Will pip install {reqs_to_install}")

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
                #self.prefix_path,
            ]
        run_cmd(cmd,f"installing {len(reqs_to_install)} requirements")
        return(cmd)
        
    
    def requirements(self):
        dir=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
        with open(os.path.join(dir,"requirements.txt"), "r") as file:
            list_libraries={}
            for line in file:
                    parts=line.split("==")
                    try:
                        library=parts[0]
                        version=parts[1][:-1]
                    except:
                        library=parts[0][:-1]
                        version=None

                    print(library)

                    installed_version=self.get_package_version(library)
                    if installed_version is None:
                        #self.install(library,version)
                        list_libraries[library]=version
                    else:
                        if str(installed_version)==str(version) or version==None:
                            iface.messageBar().pushMessage("SZ:",f'{library} is installed!',Qgis.Success)
                        else:
                            iface.messageBar().pushMessage("SZ:",f'{library} is already installed but the actual version '+f'({installed_version}) is different than the required ({version}). It may cause errors!',Qgis.Warning)
        print(list_libraries)
        self.install(list_libraries)

    def install(self,list_libraries):
        #dialog = MainDialog(
        #    list_libraries.keys()
        #)
        #if dialog.exec_():

        #    reqs_to_install = dialog.reqs_to_install
            if len(list_libraries.keys())>0:
                reqs_to_install = [f"{library}=={version}" if version else library for library, version in list_libraries.items()]

                if QMessageBox.question(None, "UMEP for Processing Python dependencies not installed",
                    f"Do you automatically want install missing python modules {reqs_to_install}? \r\n"
                    "QGIS will be non-responsive for a couple of minutes.",
                    QMessageBox.Ok | QMessageBox.Cancel) == QMessageBox.Ok:

                    try:
                        log(f"Will install selected dependencies : {reqs_to_install}")
                        try:
                            #windows
                            print('install')
                            command=self.pip_install_reqs(reqs_to_install,os.path.join(self.venv_path,"Scripts","pythonw.exe"))
                        except Exception:
                            #linux and macos
                            command=self.pip_install_reqs(reqs_to_install,os.path.join(self.venv_path,"bin","pip"))
                        QMessageBox.information(None, "Packages successfully installed",
                                                "To make all parts of the plugin work it is recommended to restart your QGIS-session.")
                    except Exception as e:
                        QgsMessageLog.logMessage(traceback.format_exc(), level=Qgis.Warning)
                        QMessageBox.information(None, "An error occurred",
                                                "SZ couldn't install Python packages!\n"
                                                "See 'General' tab in 'Log Messages' panel for details.\n"
                                                "Report any errors to https://github.com/SZtools/SZ/issues")
                else:
                    QMessageBox.information(None,"Information", "Packages not installed. Some UMEP tools will not be fully operational.")

                


                #sys.path_importer_cache.clear()


                # log_file = os.path.join(tempfile.gettempdir(), "SZ-logs.txt")
                # # Add the current date and time to the log file
                # current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # log_entry = f"\nLog created on: {current_datetime}\n"

                # if os.path.exists(log_file):
                #     # If the log file already exists, open it in append mode
                #     with open(log_file, 'a') as log:
                #         log.write(log_entry)
                # else:
                #     # If the log file doesn't exist, create it and write the log entry
                #     with open(log_file, 'w') as log:
                #         log.write(log_entry)
                # try:
                #     with open(log_file, 'a') as log:
                #         # Create a temporary log file
                #         subprocess.check_call(command, stderr=log, stdout=log)    
                #         iface.messageBar().pushMessage("SZ:", 'Dependencies installed successfully!', Qgis.Success)
                # except subprocess.CalledProcessError:
                #     # Read the error message from the temporary log file
                #     with open(log_file, 'r') as log:
                #         error_message = log.read()

                #     log_link = f'{log_file}'

                #     # Construct the error message with the log file link
                #     error_message = f'Error occurred while installing dependencies. \n{error_message} \nread {log_link} for details.'
                #     iface.messageBar().pushMessage("SZ Log:", error_message, Qgis.Critical)
                #     #sys.exit('Error occurred while installing dependencies. Check the log for details.')
                #     sys.exit(error_message)

                # # Remove the temporary log file
                # os.remove(log_file)

    def get_package_version(self,package_name):
        try:
            # Use pip to get package information
            result = subprocess.check_output([self.qgis_python_interpreter,'-m','pip', 'show', package_name], universal_newlines=True)
            
            # Split the output into lines and find the line containing "Version"
            lines = result.strip().split('\n')
            for line in lines:
                if line.startswith("Version: "):
                    # Extract and return the version number
                    return line[len("Version: "):].strip()

            
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        return None
    
    
    
    

