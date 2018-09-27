import sklearn.neural_network
import sklearn.metrics
import numpy

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