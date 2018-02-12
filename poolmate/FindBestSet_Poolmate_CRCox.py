import sklearn.neural_network
import poolmate.teach
import numpy

# Utility functions
def dropUnusedUnits(x):
    z = numpy.any(x,axis=0)
    return x[:,z]

def loadExamples(filename,dtype=numpy.dtype('float'),simplify=False):
    print(filename)
    with open(filename) as f:
        ex = numpy.array([x.strip('\n').split(',') for x in f.readlines()],dtype=dtype)

    if simplify:
        ex = dropUnusedUnits(ex)

    return ex

# Load data
orth = loadExamples('C:/Users/mbmhscc4/GitHub/aae/raw/3k/orth.csv', simplify=True)
phon = loadExamples('C:/Users/mbmhscc4/GitHub/aae/raw/3k/phon.csv', simplify=True)

# Define MyLearner
class MyLearner(object):
    examples = []
    testingset = []
    model = []
    def setmodel(self, model):
        self.model = model

    def setexamples(self, in_patterns, out_patterns):
        # Stores examples as a list of 2-element tuples (input,target)
        self.examples = [(i,o) for i,o in zip(list(in_patterns),list(out_patterns))]

    def loss(self, model):
        # return some float loss
        testset = self.testingset
        i,o = list(zip(*testset))
        return self.model.score(numpy.array(i),numpy.array(o))

    def fit(self, xy):
        # return model fit on xy
        trainingset = [self.examples[i] for i in xy]
        self.testingset = [e for i,e in enumerate(self.examples) if not i in xy]
        i,o = list(zip(*trainingset))
        self.model.fit(numpy.array(i),numpy.array(o))

def setup_learner(hidden_layer_sizes, max_iter, input_patterns, target_patterns):
    # Define model (Multi-Layer Perceptron Classifier)
    model = sklearn.neural_network.MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, # (10,),
        activation="logistic", # Target outputs are binary
        solver='lbfgs', # Optimize with Stochastic GD
        alpha=0.0001,
        batch_size=0, # Small batch size to emphasize sequential dependence
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=max_iter, # 10000, # Small max iter to exert control over training sequence
        shuffle=True, # Do not shuffle examples, to keep control over sequence
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False, # Remember model state from .fit() to .fit()
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
    learner.setexamples(input_patterns, target_patterns)

    return learner

def setup_runner(search_budget=1000, teaching_set_size=200):
    runner = poolmate.teach.Runner()
    options = poolmate.teach.build_options(
            search_budget=search_budget,
            teaching_set_size=teaching_set_size)
    return (runner, options)

if __name__ == '__main__':
    import random
    import sys
    budget = sys.argv[1]
    teaching_set_size = sys.argv[1]
    teaching_set_size = sys.argv[1]
    candidate_pool = random.sample(range(len(orth)), 1000)
    learner = setup_learner(hidden_layer_sizes=(100,), max_iter=1000, input_patterns=orth, target_patterns=phon)
    runner, options = setup_runner(search_budget=1000, teaching_set_size=200)

    best_loss, best_set = runner.run_experiment(
        candidate_pool,
        learner,
        options
    )

    with open('poolmate_candidate_pool.txt', 'w') as f:
        for e in candidate_pool:
            f.write("{example:d}\n".format(example=e))

    with open('poolmate_bestloss.txt', 'w') as f:
        f.write("{loss:.4f}\n".format(loss=best_loss))

    with open('poolmate_bestset.txt', 'w') as f:
        for e in best_set:
            f.write("{example:d}\n".format(example=e))

