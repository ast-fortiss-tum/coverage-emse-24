from pathlib import Path
import pandas as pd
import os
from visualization import combined
from utils.file_utils import *

#multi_folder = "/home/lev/Projects/testing/SBT-research-MNIST/results/analysis/round1_multiseed_CriticalMNISTConf_07/"

algos = [
        "RS",
         "NSGA-II",
         "NSGA-II-D"
        ]

NAME_COMBINED_CID = "combined_cid.csv"
NAME_AVG_COMBINED_CID = "avg_combined.csv"
##############################

def generate_cid_avg(paths, output_folder):
    
    print(paths[0])
    sum = pd.read_csv(paths[0])

    columns = sum.columns[0:]

    for path in paths[1:]:
        sum = sum + pd.read_csv(path)
    res = sum/len(paths)
    out_path = output_folder + os.sep + NAME_AVG_COMBINED_CID
    res.to_csv(out_path, 
                sep=",", 
                columns=columns,
                index=False)
    
    return res, out_path

def create_overview(paths, algos, save_folder, suffix = None):

    print(f"++++++ overview paths: {paths}")
    import csv
    
    values_algo = {}
    
    for algo in algos:
        values_algo[algo] = []

    for path in paths:
        cid_combined = pd.read_csv(path,index_col=0)
        print(cid_combined)
        n_rows = cid_combined.shape[0]
        print(f"n_rows: {n_rows}")
        for algo in algos:
            val = cid_combined.iloc[-1][algo + "_cid"]
            values_algo[algo].append(val)

    Path(save_folder).mkdir(parents=True, exist_ok=True)

    with open(save_folder + os.sep + f'overview_cid{suffix if suffix is not None else ""}.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)
        header = ['run']
        for algo in algos:
            header.append(algo)
                          
        write_to.writerow(header)

        for i in range(0, len(paths)):
            line =  [f'{i+1}']
            for algo in algos:
                line.append(values_algo[algo][i])
            write_to.writerow(line)
        f.close()
#############################

titles_plot = [
            "Multiple Seeds Oracle 0.5",
            "Multiple Seeds Oracle 0.7",
            "Multiple Seeds Oracle 0.95"
            ]
multi_folders = [
    "/home/lev/Projects/testing/SBT-research-MNIST/results/analysis/multiseed/",
"/home/lev/Projects/testing/SBT-research-MNIST/results/analysis/multiseed_CriticalMNISTConf_07/",
"/home/lev/Projects/testing/SBT-research-MNIST/results/analysis/multiseed_CriticalMNISTConf_095/"
]

for i, multi_folder in enumerate(multi_folders):
    paths = get_subfolder_paths(multi_folder)

    # print(f"found subfolders: {paths}")

    paths_cid = []
    for path in paths:
        cid_file = find_file_in_subdirectory(path,NAME_COMBINED_CID)
        paths_cid.append(cid_file)

    print(f"combined cids found: {paths_cid}")

    ####### cid avg

    _, path_cid_avg = generate_cid_avg(paths_cid, multi_folder)

    ####### stat analysis

    create_overview(paths_cid,
                    algos,
                    save_folder=multi_folder)

    combined.statistical_analysis_from_overview(input_folder=multi_folder,
                    metric_name="cid",
                    save_folder=multi_folder)

    ######## plotting

    n_func_evals_lim = 1000
    distance_tick = 100

    metric_data_loaded = combined.retrieve_metric_data_from_csv([path_cid_avg],
                                                                n_algos=len(algos))

    combined.make_comparison_plot(max_evaluations = n_func_evals_lim, 
                                    save_folder = multi_folder, 
                                    subplot_metrics = metric_data_loaded, 
                                    subplot_names = ["CID"], 
                                    algo_names = algos, 
                                    distance_tick=distance_tick,
                                    title_plot=titles_plot[i])
