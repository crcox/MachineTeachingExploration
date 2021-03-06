import sklearn.neural_network
import sklearn.metrics
from poolmate.teach import Runner, build_options
import numpy

# Utility functions
def loadExamples(filename,dtype=numpy.dtype('float'),simplify=False):
    def dropUnusedUnits(x):
        z = numpy.any(x,axis=0)
        return x[:,z]

    print(filename)
    with open(filename) as f:
        ex = numpy.array([x.strip('\n').split(',') for x in f.readlines()],dtype=dtype)

    if simplify:
        ex = dropUnusedUnits(ex)

    return ex

# Define MyLearner
class MyLearner(object):
    examples = []
    trainingset = []
    testingset = []
    model = []
    def setmodel(self, model):
        self.model = model

    def setexamples(self, in_patterns, out_patterns):
        # Stores examples as a list of 2-element tuples (input,target)
        self.examples = [(i,o) for i,o in zip(list(in_patterns),list(out_patterns))]

    def settestingset(self, test_set):
        # Stores examples as a list of 2-element tuples (input,target)
        self.testingset = [e for i,e in enumerate(self.examples) if i in test_set]

    def loss(self, model):
        # return some float loss
        testset = self.testingset
        i,o = list(zip(*testset))
        return sklearn.metrics.log_loss(numpy.array(o), self.model.predict_proba(numpy.array(i)))

    def loss_training(self, model):
        # return some float loss
        trainingset = self.trainingset
        i,o = list(zip(*trainingset))
        return sklearn.metrics.log_loss(numpy.array(o), self.model.predict_proba(numpy.array(i)))

    def fit(self, xy):
        # return model fit on xy
        self.trainingset = [self.examples[i] for i in xy]
        if not self.testingset:
            self.testingset = [e for i,e in enumerate(self.examples) if not i in xy]
        i,o = list(zip(*self.trainingset))
        self.model.fit(numpy.array(i),numpy.array(o))

def setupLearner(hidden_layer_sizes, max_iter, input_patterns, target_patterns, test_set):
    # Define model (Multi-Layer Perceptron Classifier)
    model = sklearn.neural_network.MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, # (10,),
        activation="logistic", # Target outputs are binary
        solver='lbfgs', # Optimize with Stochastic GD
        alpha=0.0001,
        batch_size=0, # Small batch size to emphasize sequential dependence
        learning_rate="constant",
        learning_rate_init=0.1,
        power_t=0.5,
        max_iter=max_iter, # 10000, # Small max iter to exert control over training sequence
        shuffle=True, # Do not shuffle examples, to keep control over sequence
        random_state=None,
        tol=1e-8,
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
    learner.settestingset(test_set)

    return learner

def setupRunner(search_budget=1000, teaching_set_size=200):
    runner = Runner()
    options = build_options(
            search_budget=search_budget,
            teaching_set_size=teaching_set_size)
    return (runner, options)

if __name__ == '__main__':
    import random
    import sys
    budget = int(sys.argv[1])
    teaching_set_size = int(sys.argv[2])
    #pool_size = int(sys.argv[3])
    #eval_size = int(sys.argv[4])
    pool_file = str(sys.argv[3])
    eval_file = str(sys.argv[4])
    hidden_size = int(sys.argv[5])

    # Load data
    orth = loadExamples('C:/Users/mbmhscc4/GitHub/aae/raw/3k/orth.csv', simplify=True)
    phon = loadExamples('C:/Users/mbmhscc4/GitHub/aae/raw/3k/phon.csv', simplify=True)

    # candidate_pool = random.sample(range(len(orth)), pool_size)
    with open(pool_file,'r') as f:
        candidate_pool = [int(i.strip()) for i in f.readlines()]
    #tmp = [i for i in range(len(orth)) if not i in candidate_pool]
    #test_set = random.sample(tmp, eval_size)
    with open(eval_file,'r') as f:
        test_set = [int(i.strip()) for i in f.readlines()]
    runner, options = setupRunner(search_budget=budget, teaching_set_size=teaching_set_size)

    for max_iter in [250,500,1000,2000]:
        learner = setupLearner(
                hidden_layer_sizes=(hidden_size,),
                max_iter=max_iter,
                input_patterns=orth,
                target_patterns=phon,
                test_set=test_set)
        best_loss, best_set = runner.run_experiment(
            candidate_pool,
            learner,
            options
    )

    with open("poolmate_candidate_pool_{iter:d}.txt".format(iter=max_iter), 'w') as f:
        for e in candidate_pool:
            f.write("{example:d}\n".format(example=e))

    with open("poolmate_test_set_{iter:d}.txt".format(iter=max_iter), 'w') as f:
        for e in candidate_pool:
            f.write("{example:d}\n".format(example=e))

    with open("poolmate_bestloss_{iter:d}.txt".format(iter=max_iter), 'w') as f:
        f.write("{loss:.4f}\n".format(loss=best_loss))

    with open("poolmate_bestset_{iter:d}.txt".format(iter=max_iter), 'w') as f:
        for e in best_set:
            f.write("{example:d}\n".format(example=e))

