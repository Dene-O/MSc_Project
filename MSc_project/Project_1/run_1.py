
import numpy as np
import pandas as pd
from sklearn import preprocessing

import parkinson as pk



pk = pk.Parkinson()

x_values, y_values = pk.x_y_values()

print(x_values.shape)
print(y_values.shape)
