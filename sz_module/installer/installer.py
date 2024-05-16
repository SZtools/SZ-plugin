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
from ..utils import log,warn
from .utils import (
    locate_py,
    add_venv,
    install_pip,
    pip_install_reqs,
    get_package_version,
)


class installer():
    def __init__(self,version):
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
        self.site_packages_path = os.path.join(self.prefix_path,self.plugin_venv,"Lib", "site-packages")
        self.bin_path = os.path.join(self.prefix_path,self.plugin_venv,"Scripts")
        if self.site_packages_path not in sys.path:
            log(f"Adding {self.site_packages_path} to PYTHONPATH")
            sys.path.insert(0, self.site_packages_path)
            os.environ["PYTHONPATH"] = (
                self.site_packages_path + ";" + os.environ.get("PYTHONPATH", "")
            )

        if self.bin_path not in os.environ["PATH"]:
            log(f"Adding {self.bin_path} to PATH")
            os.environ["PATH"] = self.bin_path + ";" + os.environ["PATH"]    

    def preliminay_req(self):
        try:
            add_venv(self.prefix_path,self.venv_path,self.plugin_venv,self.qgis_python_interpreter)
        except Exception as e:
            log(f"An error occurred: {e}")
            return False
        try:
            try:
                #windows
                #self.uninstall_pip(['pip'],os.path.join(self.venv_path,"Scripts","python"))
                command=install_pip(['ensurepip'],os.path.join(self.venv_path,"Scripts","pythonw.exe"))
            except Exception:
                #linux and macos
                #self.uninstall_pip(['pip'],os.path.join(self.venv_path,"bin","python"))
                command=install_pip(['ensurepip'],os.path.join(self.venv_path,"bin","python")) 
        except Exception as e:
            log(f"An error occurred: {e}")
            return False
           

    def requirements(self):
        dir=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
        log(f"verify requirements")
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
                    installed_version=get_package_version(self.qgis_python_interpreter,library)
                    if installed_version is None:
                        list_libraries[library]=version
                    else:
                        if str(installed_version)==str(version) or version==None:
                            iface.messageBar().pushMessage("SZ:",f'{library} is already installed!',Qgis.Success)
                            log(f'{library} is already installed!')
                        else:
                            log(f'{library} is already installed but the actual version '+f'({installed_version}) is different than the required ({version}). It may cause errors!')
                            iface.messageBar().pushMessage("SZ:",f'{library} is already installed but the actual version '+f'({installed_version}) is different than the required ({version}). It may cause errors!',Qgis.Warning)
        return self.install(list_libraries)

    def install(self,list_libraries):
            if len(list_libraries.keys())>0:
                reqs_to_install = [f"{library}=={version}" if version else library for library, version in list_libraries.items()]
                if QMessageBox.question(None, "SZ for Processing Python dependencies not installed",
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
                            command=pip_install_reqs(self.prefix_path,self.plugin_venv,reqs_to_install,os.path.join(self.venv_path,"bin","pip"))
                        QMessageBox.information(None, "Packages successfully installed",
                                                #"To make all parts of the plugin work it is recommended to restart your QGIS-session.")
                                                "You can find the SZ-plugin in the Processing-toolbox")
                    except Exception as e:
                        QgsMessageLog.logMessage(traceback.format_exc(), level=Qgis.Warning)
                        QMessageBox.information(None, "An error occurred",
                                                "SZ couldn't install Python packages!\n"
                                                "See 'General' tab in 'Log Messages' panel for details.\n"
                                                "Report any errors to https://github.com/SZtools/SZ/issues")
                        log("An error occurred:", e)
                        return False
                else:
                    QMessageBox.information(None,"Information", "Packages not installed. Some SZ tools will not be fully operational.")
                    sys.path_importer_cache.clear()
                    log("Packages not installed. Some SZ tools will not be fully operational.")
                    return False
                
                sys.path_importer_cache.clear()
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

        
        
    

