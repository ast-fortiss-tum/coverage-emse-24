import os
from visualization.combined import make_comparison_plot, retrieve_metric_data_from_csv

subplot_names_1 = ["O_Large",
                  "O_Medium",
                  "O_Small"]

# subplot_names_3 = ["CID [20]",
#                  "FPS[8000]"]

algo_names = ["RS","NSGA-II-D","NSGA-II"]

paths = [
         os.getcwd() + r"/scripts/data/final/mnist/coverage/mseed/20/C05/avg_combined.csv",
         os.getcwd() + r"/scripts/data/final/mnist/coverage/mseed/20/C07/avg_combined.csv",
         os.getcwd() + r"/scripts/data/final/mnist/coverage/mseed/20/C095/avg_combined.csv"
         ]
print(paths)

analysis_folder = os.getcwd() + r"/scripts/output/mseed/cid_comparison//"

n_func_evals_lim = 1000
distance_tick = 100

metric_data_loaded = retrieve_metric_data_from_csv(paths, n_algos=3)

make_comparison_plot(n_func_evals_lim, 
            analysis_folder, 
            metric_data_loaded, 
            subplot_names_1, 
            algo_names, 
            distance_tick=distance_tick, 
            suffix="",
            colors=['#ffbb00','#9a226a','#1347ac'])