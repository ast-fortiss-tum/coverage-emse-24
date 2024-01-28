
import subprocess
from utils import file_utils
import os

# mseed_folder = os.getcwd() + os.sep + "results" + os.sep + "analysis" + os.sep + "multiseed"

# folder_runs_mseed = file_utils.get_subfolder_paths(mseed_folder)

# from utils import file_utils
#     seed_paths = [file + os.sep for file in file_utils.get_subfolder_paths_level(main_folder=analysis_parent_folder,
#                                     target_level=2)]


analysis_parent_folder = "results" + os.sep + "analysis" + os.sep + "multiseed"

seed_paths = [file + os.sep for file in file_utils.get_subfolder_paths_level(main_folder=analysis_parent_folder,
                                    target_level=2)]

for path in seed_paths:
    #print(path)
    output = subprocess.call(["python",
                            "analysis.py", 
                            "-p", path + os.sep])