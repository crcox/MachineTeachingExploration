import sklearn.neural_network
import sklearn.metrics
#from sklearn.exceptions import ConvergenceWarning
from timeit import default_timer as timer
import itertools
import numpy
import copy
import random
import warnings
#warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

def candide_check_example(model, I, O, example):
    i = numpy.reshape(example[0], (1, len(example[0])) )
    o = numpy.reshape(example[1], (1, len(example[1])) )
    model.fit(i,o)
    O_pred = model.predict_proba(I)
    return( sklearn.metrics.log_loss(O, O_pred) )

# Define MyLearner
class MyLearner(object):
    examples = []
    model = []
    training_sequence = []
    error_on_corpus = []
    iteration = []

    def setmodel(self, model):
        self.model = model

    def setexamples(self, in_patterns, out_patterns):
        # Stores examples as a list of 2-element tuples (input,target)
        self.examples = [(i,o) for i,o in zip(list(in_patterns),list(out_patterns))]
        # self.model.classes_=range(out_patterns.shape[1]) # I do not think this is right ...

    def loss(self):
        # return some float loss
        return self.model.loss_

    def fit(self, xy):
        # return model fit on xy
        if isinstance(xy,list):
            trainingset = [self.examples[i] for i in xy]
            i,o = list(zip(*trainingset))
        else:
            example = self.examples[xy]
            i = numpy.reshape(example[0], (1, len(example[0])) )
            o = numpy.reshape(example[1], (1, len(example[1])) )

        self.model.fit(i,o)

    def partial_fit(self, xy):
        # return model fit on xy
        if isinstance(xy,list):
            trainingset = [self.examples[i] for i in xy]
            i,o = list(zip(*trainingset))
        else:
            example = self.examples[xy]
            i = numpy.reshape(example[0], (1, len(example[0])) )
            o = numpy.reshape(example[1], (1, len(example[1])) )

        self.model.partial_fit(i,o)

    def candide_search(self):
        # return model fit on xy
        I,O = list(zip(*self.examples))
        I = numpy.array(I)
        O = numpy.array(O)

        error_on_corpus = []
        lowest_error_on_corpus = float('inf')
        with open('candide_selection_error.csv', 'a') as f:
            for exampleID, example in enumerate(self.examples):
                error_on_corpus = candide_check_example(copy.deepcopy(self.model), I, O, example)
                f.write("{iter:d},{ex:d},{loss:.8f}\n".format(iter=self.model.n_iter_,ex=exampleID,loss=error_on_corpus))
                if error_on_corpus < lowest_error_on_corpus:
                    x = exampleID
                    lowest_error_on_corpus = error_on_corpus

        self.training_sequence.append(x)
        self.error_on_corpus.append(lowest_error_on_corpus)
        return(x)

# Load data
orth  = loadExamples('/mnt/sw01-home01/mbmhscc4/scratch/src/aae/raw/3k/orth.csv',  simplify=True)
phon  = loadExamples('/mnt/sw01-home01/mbmhscc4/scratch/src/aae/raw/3k/phon.csv',  simplify=True)
with open('/mnt/sw01-home01/mbmhscc4/scratch/src/aae/raw/3k/words.csv', 'r') as f:
    words = [x.strip() for x in f.readlines()]

# Define model (Multi-Layer Perceptron Classifier)
model = sklearn.neural_network.MLPClassifier(
    hidden_layer_sizes=(10,),
    activation="logistic", # Target outputs are binary
    solver='sgd', # Optimize with Stochastic GD
    alpha=0.0001,
    batch_size=1, # Small batch size to emphasize sequential dependence
    learning_rate="constant",
    learning_rate_init=0.01,
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

def main():
    samplesize = 1000
    ix = random.sample(range(len(orth)), samplesize)
    ixword = [words[i] for i in ix]
    with open('candide_sample{n:d}.csv'.format(n=samplesize), 'w') as f:
        for i,w in zip(ix,ixword):
            f.write("{wordID:d},{word:s}\n".format(wordID=i, word=w))

    print('Initialize learner')
    learner = MyLearner()
    print('Set model in learner')
    learner.setmodel(model)
    print('Set examples in learner')
    learner.setexamples(
        [orth[i] for i in ix],
        [phon[i] for i in ix]
    )

    print('start timer...')
    loopstart = timer()
    print('iter','seq','corpus_error','itertime','totaltime')
    with open('candide_training_sequence.csv', 'w') as f:
        for iteration in range(10000):
            iterstart = timer()
            exampleID = learner.candide_search()
            learner.fit(exampleID)
            iterend = timer()
            print(iteration,learner.training_sequence[-1], learner.error_on_corpus[-1], iterend-iterstart, iterend-loopstart)
            f.write("{example:d},{loss:.8f},{itertime:.4f},{totaltime:.4f}\n".format(
                    example=learner.training_sequence[-1],
                    loss=learner.error_on_corpus[-1],
                    itertime=iterend-iterstart,
                    totaltime=iterend-loopstart
                )
            )
            f.flush()

    print(learner.training_sequence)
    print(learner.error_on_corpus)

if __name__ == '__main__':
    main()

