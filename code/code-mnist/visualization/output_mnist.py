from pathlib import Path
from utils.duplicates import duplicate_free
import os
from problem.mnist.exploration import Exploration
import matplotlib.pyplot as plt
import numpy as np
from problem.mnist import evaluator, features

def output_optimal_digits(res, save_folder):
    save_folder_opt= save_folder + "digits" + os.sep + "optimal"
    Path(save_folder_opt).mkdir(parents=True, exist_ok=True)
    clean_pop = duplicate_free(res.opt)
    print(f"Length of optimal {len(clean_pop)}")
    for ind in clean_pop:
        ind.get("DIG").export_relative(parent_folder=save_folder_opt,all=True)

    print("Exported optimal digits.")

def output_critical_digits(res, save_folder):
    save_folder_opt= save_folder + "digits" + os.sep + "critical"
    Path(save_folder_opt).mkdir(parents=True, exist_ok=True)
    critical, _ = res.obtain_all_population().divide_critical_non_critical()
    clean_pop = duplicate_free(critical)
    print(f"Length of critical {len(clean_pop)}")
    for ind in clean_pop:
        ind.get("DIG").export_relative(parent_folder=save_folder_opt,all=True)

    print("Exported optimal digits.")

def output_explored_digits(res, save_folder):
    for digit in Exploration.all_inputs:
        digit.export_relative(parent_folder=save_folder,all=True)

    print("Exported digits stored in exploration.")

def output_seed_digits(res, save_folder):
    save_folder_seed= save_folder + "digits" + os.sep + "seed"
    Path(save_folder_seed).mkdir(parents=True, exist_ok=True)
    for digit in res.problem.seed_digits:
        digit.export_relative(parent_folder=save_folder_seed,all=True)

    print("Exported seed digits.")

def output_optimal_digits_all(res, save_folder):
    save_folder_opt= save_folder + "digits" + os.sep
    Path(save_folder_opt).mkdir(parents=True, exist_ok=True)
    optimal = duplicate_free(res.opt)
    
    output_subplots_digits(optimal.get("DIG"), 
                    title="Optimal Digits",
                    filename="optimal_all.png", 
                    save_folder=save_folder_opt,
                    problem=res.problem)

def trunc_float(number, n_digits):
    res = int(number * (10**n_digits))/(10**n_digits)
    return float(res)

def output_subplots_digits(digits, 
                            title, 
                            filename, 
                            save_folder, 
                            is_seed=False, 
                            problem=None,
                            ids = None):
    n = len(digits)

    if n == 0 or None in digits:
        print("No plottable digits available, could not generate subplots of digits.")
        return 
    n_col = min(10,n)
    n_row = int(np.ceil(n / n_col))
    offset = 1
    scale_title_x = 1
    scale_title_y = 2
    scale_figure = 2
    fig = plt.figure(figsize=(round(n_col*scale_figure*scale_title_x) + offset, 
                              round(n_row*scale_figure*scale_title_y) + offset))

    # print(f"[output_subplots_digits] ({title}) figure size is: {fig.get_size_inches()}")
    fig.subplots_adjust(wspace=0.25, 
                        hspace=2,
                        top=0.85,
                        bottom=0.07)
    # plt.subplots_adjust(left=0.1,
    #                     bottom=0.01,
    #                     right=0.9,
    #                     top=0.9,
    #                     wspace=0.8,
    #                     hspace=0.30)

    for i, digit in enumerate(digits):
        image = digit.purified.reshape(28, 28)
        # display reconstructed image
        ax = plt.subplot(n_row, n_col, i  + 1 )
        plt.imshow(image)
        if ids is not None:
            id = ids[i]
        else:
            id = digit.id
            
        if is_seed:
            plt.title(f"Digit {id}")
        else:
            plt.title(f"Digit {id}\
                \np: {digit.predicted_label}\
                \nc: {trunc_float(digit.confidence,3)}\
                \nb: {trunc_float(digit.brightness(problem.min_saturation),3)}\
                \nv: {trunc_float(digit.coverage(problem.min_saturation),3)}",
                loc="left",pad=10)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)   
    #plt.suptitle(title)
    #plt.tight_layout(rect=[0, 0.03, 1, 2])
    plt.savefig(save_folder + filename, format="png")

    plt.close()

def output_critical_digits_all(res, save_folder):
    save_folder_crit= save_folder + "digits" + os.sep
    Path(save_folder_crit).mkdir(parents=True, exist_ok=True)
    critical, _ = res.obtain_all_population().divide_critical_non_critical()
    critical = duplicate_free(critical)
    
    output_subplots_digits(critical.get("DIG"), 
                    title="Critical Digits",
                    filename="critical_all.png", 
                    save_folder=save_folder_crit,
                    problem=res.problem)


def output_seed_digits_all(res, save_folder):
    save_folder_crit= save_folder + "digits" + os.sep
    Path(save_folder_crit).mkdir(parents=True, exist_ok=True)

    output_subplots_digits(res.problem.seed_digits, 
                    title="Seed Digits",
                    filename="seeds_all.png", 
                    save_folder=save_folder_crit,
                    problem=res.problem)

''' Write down the population for each generation'''
def write_generations_digit(res, save_folder):

    save_folder_history = save_folder + "generations_digit" + os.sep
    Path(save_folder_history).mkdir(parents=True, exist_ok=True) 

    problem = res.problem
    hist = res.history
    design_names = problem.design_names
    objective_names = problem.objective_names

    for i, algo in enumerate(hist):
        output_subplots_digits(algo.pop.get("DIG"), 
                    title=f"Generation {i+1}",
                    filename=f'gen_{i+1}.png', 
                    save_folder=save_folder_history,
                    is_seed=False,
                    problem=res.problem)


def output_summary(res, save_folder):
    save_folder_crit= save_folder + "digits" + os.sep
    Path(save_folder_crit).mkdir(parents=True, exist_ok=True)
    critical, _ = res.obtain_all_population().divide_critical_non_critical()

    # output average distance of solutions
    dists = []
    for ind in critical:
        d = evaluator.dist_from_nearest_archived(ind.get("DIG"), critical.get("DIG"), 1)
        dists.append(d)
    print(f"Average distance of critical digits is: {np.mean(dists)}")

    optimal = duplicate_free(res.opt)
    # output average distance of solutions
    dists = []
    for ind in optimal:
        d = evaluator.dist_from_nearest_archived(ind.get("DIG"), optimal.get("DIG"), 1)
        dists.append(d)
    print(f"Average distance of optimal digits is: {np.mean(dists)}")
