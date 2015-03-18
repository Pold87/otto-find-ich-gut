import pandas as pd
import numpy as np
from ggplot import *


data = pd.read_csv("data/train.csv")


y = data.iloc[:, -1]


print(y) 

