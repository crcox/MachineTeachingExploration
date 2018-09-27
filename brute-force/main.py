import json
import random
import sys
from learner import MyLearner, setupLearner, setupRunner
import examples

with open('params.json', 'r') as f:
    cfg = json.load(f)

random.seed(cfg['random_seed'])

cfg['samples_to_search']
cfg['attempts_per_sample']
cfg['training_size']
cfg['hidden_size']
cfg['max_iter']

# Load data
orth = examples.load('data/3k/orth.csv', simplify=True)
phon = examples.load('data/3k/phon.csv', simplify=True)

for k in range(cfg['samples_to_search']):
    test_set = random.sample(range(len(orth)), len(orth) - cfg['training_size'])

    for j in range(cfg['attempts_per_sample']):
        learner = setupLearner(
                hidden_layer_sizes=(cfg['hidden_size'],),
                max_iter=cfg['max_iter'],
                input_patterns=orth,
                target_patterns=phon,
                test_set=test_set)

with open("poolmate_candidate_pool_{iter:d}.txt".format(iter=cfg['max_iter']), 'w') as f:
    for e in test_set:
        f.write("{example:d}\n".format(example=e))

with open("poolmate_test_set_{iter:d}.txt".format(iter=cfg['max_iter']), 'w') as f:
    for e in test_set:
        f.write("{example:d}\n".format(example=e))

with open("poolmate_bestloss_{iter:d}.txt".format(iter=cfg['max_iter']), 'w') as f:
    f.write("{loss:.4f}\n".format(loss=best_loss))

with open("poolmate_bestset_{iter:d}.txt".format(iter=cfg['max_iter']), 'w') as f:
    for e in best_set:
        f.write("{example:d}\n".format(example=e))

