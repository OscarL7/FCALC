from . import patterns
import numpy as np

class FcaClassifier:

    def __init__(self, context, labels, intersections = None, support = None):
        
        self.context = context
        self.labels = labels
        
        if intersections is None:
            self.intersections = [[],[]]
        else:
            self.intersections = intersections
        
        if support is None:
            self.support = []
        else:
            self.support = support

class BinarizedClassifier(FcaClassifier):
    
    def __init__(self, context, labels, intersections=None, support=None, method = "standard"):
        super().__init__(context, labels, intersections, support)
        self.method = method

    def compute_support(self, test):
        train_pos = self.context[self.labels == True]
        train_neg = self.context[self.labels == False]

        positive_support = np.empty(shape=(len(test), len(train_pos)))
        positive_counter = np.empty(shape=(len(test), len(train_pos)))
        negative_support = np.empty(shape=(len(test), len(train_neg)))
        negative_counter = np.empty(shape=(len(test), len(train_neg)))

        for i in range(len(test)):
            intsec_pos = test[i].reshape(1, -1) & train_pos
            self.intersections[0].append(intsec_pos)
             #intsec_pos = intsec_pos[intsec_pos.sum(axis=1) >= min_card]
            n_support_pos = ((intsec_pos @ (~train_pos.T)) == 0).sum(axis=1)
            n_counter_pos = ((intsec_pos @ (~train_neg.T)) == 0).sum(axis=1)

            intsec_neg = test[i].reshape(1, -1) & train_neg
            self.intersections[1].append(intsec_neg)
            # intsec_neg = intsec_neg[intsec_pos.sum(axis=1) >= min_card]
            n_support_neg = ((intsec_neg @ (~train_neg.T)) == 0).sum(axis=1)
            n_counter_neg = ((intsec_neg @ (~train_pos.T)) == 0).sum(axis=1)

            positive_support[i] = n_support_pos
            positive_counter[i] = n_counter_pos
            negative_support[i] = n_support_neg
            negative_counter[i] = n_counter_neg
        
        self.support = [(positive_support, positive_counter), (negative_support, negative_counter)]

    def predict(self, test):
        pass

class PatternClassifier(FcaClassifier):
    def __init__(self, context, labels, intersections=None, support=None, method="standard", categorical=None):
        super().__init__(context, labels, intersections, support)
        self.method = method
        if categorical is None:
            self.categorical = []
        else: 
            self.categorical = categorical
    
    def compute_support(self, test):
        train_pos = self.context[self.labels == True]
        train_neg = self.context[self.labels == False]

        positive_support = np.empty(shape=(len(test), len(train_pos)))
        positive_counter = np.empty(shape=(len(test), len(train_pos)))
        negative_support = np.empty(shape=(len(test), len(train_neg)))
        negative_counter = np.empty(shape=(len(test), len(train_neg)))

        if not self.categorical:
            for i in range(len(test)):
                for j in range(len(train_pos)):
                    intsec = patterns.IntervalPattern(test[i],train_pos[j])
                    n_support_pos = sum((~(intsec.low <= train_pos * train_pos <= intsec.high)).sum(axis=1) == 0)
                    n_counter_pos = sum((~(intsec.low <= train_neg * train_neg <= intsec.high)).sum(axis=1) == 0)

        elif len(self.categorical) == len(test[0]):
            for i in range(len(test)):
                for j in range(len(train_pos)):
                    intsec = patterns.CategoricalPattern(test[i],self.context[j])

        else:
            for i in range(len(test)):
                for j in range(len(train_pos)):
                    intsec_cat = patterns.CategoricalPattern(test[i][self.categorical],
                                                             self.context[j][self.categorical])
                    intsec_num = patterns.IntervalPattern(np.delete(test[i], self.categorical),
                                                          np.delete(self.context[j], self.categorical))

        self.support = [(positive_support, positive_counter), (negative_support, negative_counter)]
