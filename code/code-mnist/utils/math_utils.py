import numpy as np
# avoid rounding up/down to even numbers when number is x.5 
def round(number):
    return np.round(number - 0.00001)