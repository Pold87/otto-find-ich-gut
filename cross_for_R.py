import pandas as pd
import numpy as np

sample_sub = "submissions/sampleSubmission.csv"
sample_sub_df = pd.read_csv(sample_sub)

def normalize(row, epsilon=1e-15):
    
    row = row / np.sum(row)
    row = np.maximum(epsilon, row)
    row = np.minimum(1 - epsilon, row)
    
    return row
    
def logloss_mc(y_true, y_probs):
    
    # Normalize probability data frame
    y_probs = y_probs.apply(normalize, axis=1)
        
    log_vals = []
        
    for i, y in enumerate(y_true):
        c = int(y.split("_")[1])
        log_vals.append(- np.log(y_probs.iloc[i,c - 1]))
        
    return np.mean(log_vals)

df_holdout = pd.read_csv("data/holdout20.csv")
y_valid = df_holdout.target

sub = pd.read_csv("submission.csv").iloc[:, 1:]
ll = logloss_mc(y_valid, sub)
print ll

with open("scores.txt", 'a') as f:
	f.write(str(ll) + '\n')