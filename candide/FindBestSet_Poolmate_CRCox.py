import sklearn.neural_network
import poolmate.teach
import numpy

# Utility functions
def dropUnusedUnits(x):
    z = numpy.any(x,axis=0)
    return x[:,z]

def loadExamples(filename,dtype=numpy.dtype('float'),simplify=False):
    with open(filename) as f:
        ex = numpy.array([x.strip('\n').split(',') for x in f.readlines()],dtype=dtype)

    if simplify:
        ex = dropUnusedUnits(ex)

    return ex

# Define MyLearner
class MyLearner(object):
    examples = []
    model = []
    def setmodel(self, model):
        self.model = model

    def setexamples(self, in_patterns, out_patterns):
        # Stores examples as a list of 2-element tuples (input,target)
        self.examples = [(i,o) for i,o in zip(list(in_patterns),list(out_patterns))]

    def loss(self, model):
        # return some float loss
        return self.model.loss_

    def fit(self, xy):
        # return model fit on xy
        trainingset = [self.examples[i] for i in xy]
        i,o = list(zip(*trainingset))
        self.model.fit(i,o)

# Load data
orth = loadExamples('3k/orth.csv', simplify=True)
phon = loadExamples('3k/phon.csv', simplify=True)

# Define model (Multi-Layer Perceptron Classifier)
model = sklearn.neural_network.MLPClassifier(
    hidden_layer_sizes=(10,),
    activation="logistic", # Target outputs are binary
    solver='sgd', # Optimize with Stochastic GD
    alpha=0.0001,
    batch_size=1, # Small batch size to emphasize sequential dependence
    learning_rate="constant",
    learning_rate_init=0.001,
    power_t=0.5,
    max_iter=1, # Small max iter to exert control over training sequence
    shuffle=False, # Do not shuffle examples, to keep control over sequence
    random_state=None,
    tol=1e-4,
    verbose=False,
    warm_start=True, # Remember model state from .fit() to .fit()
    momentum=0.9,
    nesterovs_momentum=True,
    early_stopping=False,
    validation_fraction=0.1,
    beta_1=0.9, # Not used with SGD (Adam only)
    beta_2=0.999, # Not used with SGD (Adam only)
    epsilon=1e-8 # Not used with SGD (Adam only)
)

learner = MyLearner()
learner.setmodel(model)
learner.setexamples(orth, phon)

runner = poolmate.teach.Runner()
options = poolmate.teach.build_options(
        search_budget=1000,
        teaching_set_size=100)
candidate_pool = range(len(orth))

best_loss, best_set = runner.run_experiment(
    candidate_pool,
    learner,
    options
)

print best_loss
print best_set
with open('poolmate_bestloss.txt', 'w') as f:
    f.write("{loss:.4f}\n".format(loss=best_loss))

with open('poolmate_bestset.txt', 'w') as f:
    for e in best_set:
        f.write("{example:d}\n".format(example=e))
