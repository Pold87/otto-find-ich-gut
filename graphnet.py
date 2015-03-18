import graphlab as gl
import pandas as pd

# Load the data
train = gl.SFrame.read_csv('data/train.csv')
test = gl.SFrame.read_csv('data/test.csv')
sample = gl.SFrame.read_csv('submissions/sampleSubmission.csv')

del train['id']

# Train a model
m = gl.boosted_trees_classifier.create(dataset = train,
                                       target='target',
                                       max_iterations=300,
                                       max_depth = 9)
 
# Make submission
preds = m.predict_topk(test, output_type='probability', k=9)
preds['id'] = preds['id'].astype(int) + 1
preds = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '')
preds = preds.sort('id')
 
assert sample.num_rows() == preds.num_rows()

preds.save("submission_graph.csv", format = 'csv')
