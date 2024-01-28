# Search-based Test Case Generation
## Intro


The tool implements search-based critical test case generation using [NSGA2](https://orbilu.uni.lu/bitstream/10993/28071/1/ASE16BenAbdessalem.pdf) and [NSGA2-DT](https://orbilu.uni.lu/bitstream/10993/33706/1/ICSE-Main-24.pdf).

NSGA2-DT is an algorithm, where additionally to the standard search approach using NSGA2 simulated scenario instances are classified using a criticality metric and clustered using decision tree classification. NSGA2 is applied in the following tree iteration to search in regions that are considered as more critical than a specified threshold.

## Preliminaries

The tool can be used together with the Prescan Simulator []() and the Carla Simulator[](). 
Python (>= 3.7) needs to be installed to use the Prescan Simulator.

Create first a virtual environment (install virtualenv with 'pip install virtualenv' if not present):

```
virtualenv --python C:\Path\To\Python\python.exe venv
```

Activate the virtual environment using:

```
source venv/Scripts/activate
```

Install dependencies in the virtual environment:

```
python -m pip install -r requirements.txt
```

### Preliminaries using Carla

Follow the steps desribed  [s. here](/carla_simulation) to integrate Carla.

### Preliminaries using Prescan

To use Prescan matlab needs to be installed. 
Compatibility has been tested with Prescan 2021.3.0 and MATLAB R2019.b.

#### Matlab

Matlab R2019.b can be downloaded from <file://///fs01/Install/Mathworks> (VPN to fortiss network/local LAN connection required). Further installation instruction is available here:  <https://one.fortiss.org/sites/workinghere/Wikipages/install%20some%20software.aspx>

The matlab engine needs to be exported to python.
Execute 

```
cd MATLAB_DIR/extern/engines/python
py -3.7 setup.py install --prefix="C:\Path\To\Project\venv"

```
Further options [s. here](https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

**For automatic engine sharing**

The matlab engine can be automatically shared, when Matlab is started via the Prescan Process Manager.
MATLAB executes a script named  [`startup`](https://de.mathworks.com/help/matlab/ref/startup.html) that is added to the userpath (check location vie `userpath`). Add a script named `startup` to the experiment location and change the userpath to point to the startup script to enable automatic sharing.

### Usage

The tool can be used to generate test cases for scenarios in Prescan and Carla. We have also implement for testing the algorithm the search on test problems.
The results are written in the *results* folder.

### Search for multiobjective test examples (Experiment 4, Experiment 6)

Run the following to execute search with a mathematical multobjective problem.

```
python run.py -e 4
```

### Search for a carla scenario (Experiment 2, Experiment 5)

To run search with a scenario in carla we have implemented two examples  where a pedestrian crosses the lane of the ego. In one the environment is modified, and the other only the pedestrians speed and host speed is modified.
To 
```
python run.py -e 2
```


### Search for a prescan scenario

Start MATLAB using the Prescan Process Manager and share the engine by executing in the terminal:

```
matlab.engine.shareEngine
```

#### Real Example

To run search with an example Prescan experiment
make sure **PrescanHeedsExperiment** is downloaded in a folder **experiments** that is placed next to this.

Run the following to execute search:

```
python run.py -e 3
```

#### New Experiment

Make sure to have a file named **UpdateModel.m** in the experiments folder that reads from a json file **input.json** parameter values and sets the values in the experiment model.
Consider as an example experiment **experiments/PrescanHeedsExperiment**

Run the tool by providing the path to the experiment file, the upper and lower bounds, as well the names of the parameters to vary (should match with the ones set by **UpdateModel.m**):

```
python run.py -f <experiment.pb> -min 1 1 1 -max 10 20 10 -m "par1 "par2" "par3"
```

### Optional Parameters

All flags that can be set are (get options by -h flag):

```
  -e EXP_NUMBER         Hardcoded example scenario to use [2 to 6].
  -i N_ITERATIONS       Number iterations to perform.
  -n SIZE_POPULATION    The size of the initial population of scenario candidates.
  -a ALGORITHM          The algorithm to use for search, 1 for NSGA2, 2 for NSGA2-DT.
  -t MAXIMAL_EXECUTION_TIME
                        The time to use for search with nsga2-DT (actual search time can be above the threshold, since algorithm might        
                        perform nsga2 iterations, when time limit is already reached.
  -f XOSC               The path to the scenario description file/experiment.
  -min VAR_MIN [VAR_MIN ...]
                        The lower bound of each parameter.
  -max VAR_MAX [VAR_MAX ...]
                        The upper bound of each parameter.
  -m DESIGN_NAMES [DESIGN_NAMES ...]
                        The names of the variables to modify.
  -dt MAX_TREE_ITERATIONS
                        The maximum number of total decision tree generations (when using NSGA2-DT algoritm).
```

## Limitations

Since OpenSCENARIO support of Prescan is not mature, Prescan experiment files have to be used.

## Authors

Lev Sorokin (sorokin@fortiss.org), Tiziano Munaro (tiziano@fortiss.org)
