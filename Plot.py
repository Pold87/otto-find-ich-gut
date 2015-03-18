import pandas as pd
import numpy as np
<<<<<<< HEAD
import matplotlib.pyplot as plt

data = pd.read_csv("data/train.csv")
y = data.iloc[:, -1]

plt.hist(y)
=======
from ggplot import *


data = pd.read_csv("data/train.csv")


y = data.iloc[:, -1]


print(y) 
>>>>>>> a78238ffd0c280eae61ed763ac054109140a8aa8

