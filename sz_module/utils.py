import subprocess
import os
from qgis.core import Qgis
from qgis.utils import iface
import sys
sys.setrecursionlimit(10000)
import tempfile
from datetime import datetime
from qgis.core import *
from qgis import *

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
                        if str(installed_version)==str(version) or version==None:
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
    


    
    