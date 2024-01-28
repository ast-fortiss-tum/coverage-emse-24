import os
from visualization.combined import make_comparison_plot, retrieve_metric_data_from_csv

subplot_names_1 = ["C_Large",
                  "C_Medium",
                  "C_Small"]

# subplot_names_3 = ["CID [20]",
#                  "FPS[8000]"]

algo_names = ["NSGA-II (Diverse)","NSGA-II","RS"]

paths = [
         os.getcwd() + r"/scripts/data/final/mnist/coverage/ALL_ORACLE_SIMPLE/large_05/cid/combined_cid.csv",
         os.getcwd() +  r"/scripts/data/final/mnist/coverage/ALL_ORACLE_SIMPLE/medium_07/cid/combined_cid.csv",
         os.getcwd() +  r"/scripts/data/final/mnist/coverage/ALL_ORACLE_SIMPLE/small_095/cid/combined_cid.csv",
         ]
print(paths)

analysis_folder = os.getcwd() + r"/scripts/output/cid_comparison//"

n_func_evals_lim = 1000
distance_tick = 100

metric_data_loaded = retrieve_metric_data_from_csv(paths, n_algos=3)

make_comparison_plot(n_func_evals_lim, 
            analysis_folder, 
            metric_data_loaded, 
            subplot_names_1, 
            algo_names, 
            distance_tick=distance_tick, 
            suffix="_gs_fps")
