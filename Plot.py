import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data/train.csv")
y = data.iloc[:, -1]

plt.hist(y)

