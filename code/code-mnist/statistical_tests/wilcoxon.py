from math import sqrt
import numpy as np
from numpy import mean
from numpy import var
from scipy.stats import wilcoxon
from statistical_tests import vargha_delaney
from statistical_tests.cohend import cohend
import logging as log

def run_wilcoxon_and_cohend(data1, data2):
    w_statistic, pvalue = wilcoxon(data1, data2)#, mode='exact')
    cohensd = cohend(data1, data2)
    log.info(f"P-Value is: {pvalue}")
    log.info(f"Cohen's D is: {cohensd}")
    return pvalue, cohensd[0]

def run_wilcoxon_and_delaney(data1, data2):
    w_statistic, pvalue = wilcoxon(data1, data2)#, mode='exact')
    delaney = vargha_delaney.VD_A(data1, data2)
    log.info(f"P-Value is: {pvalue}")
    log.info(f"Delaney's effect size is: {delaney}")
    return pvalue, delaney[0]

def main():
    # data1 =  # first distribution
    # data2 =  # second distribution
    data1 = [0.18184417518118298, 0.2245256586915804, 0.17647426490463292, 0.16230448654974916, 0.18738405941240882, 0.3101233019449206, 0.3054707753488759, 0.27873281142439615, 0.25083069533060276, 0.1612208601808664]
    data2 = [0.0753816102665673, 0.06437859231871611, 0.0666155505693268, 0.07059016975382326, 0.07065245326751085, 0.06454400375044805, 0.07402294266750688, 0.07003813819778802, 0.0687958141591721, 0.07122835191659796]

    run_wilcoxon_and_delaney(data1, data2)
    run_wilcoxon_and_cohend(data1, data2)

if __name__ == '__main__':
    main()