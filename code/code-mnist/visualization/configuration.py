import matplotlib.pyplot as plt

'''Font in plots'''
SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

font = {'family': 'sans serif',
        'weight': 'normal',
        'size': MEDIUM_SIZE}

plt.rc('font', **font)  # controls default text sizes
plt.rc('axes', titlesize=18)  # fontsize of the axes title
plt.rc('axes', labelsize=17)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


'''Colour mapping in plots'''
color_optimal = 'red' #'coral'
color_not_optimal = 'gray'
color_critical = 'darkviolet' # 'lightseagreen'
color_not_critical = 'khaki'

'''Metric Name'''
COVERAGE_METRIC_NAME = "CID"

'''Colors for graphs'''
c_map= {
   "NLSGA-II" : "#9a226a",
   "NSGA-II" :  "#1347ac",
   "LNSGA-II" : "#00CC99",
   "NSGA-II-F" :"#ffbb00",
   "NSGA-II-DT": "#808080",
   "NSGA-II-D" : '#9999FF',
   "RS": "#9a226a"
}
