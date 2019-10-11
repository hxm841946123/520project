# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math
import pdb

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """

    print len(trainingData)  
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    "*** YOUR CODE HERE ***"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #pdb.set_trace()
    AccuracyCount = -1
    testPrior = util.Counter()
    testLikelihood_phi = util.Counter()
    testCounts = util.Counter()

    # count from the training data
    for i in range(len(trainingData)):
      datum = trainingData[i]
      label = trainingLabels[i]
      testPrior[label] = testPrior[label] + 1
      for feature,value in datum.items():
        testCounts[(feature,label)] = testCounts[(feature,label)] + 1 
        if value >0:
          # print Likelihood_phi
          testLikelihood_phi[(feature,label)] = testLikelihood_phi[(feature,label)] + 1

 #   pdb.set_trace()
    # smoothing
    for k in kgrid:
      Prior = util.Counter()
      for feature,value in testPrior.items():
        Prior[feature] = Prior[feature] + value
      Likelihood_phi = util.Counter()
      for feature,value in testLikelihood_phi.items():
        Likelihood_phi[feature] = Likelihood_phi[feature] + value
      Counts = util.Counter()
      for feature,value in testCounts.items():
        Counts[feature] = Counts[feature] + value

      for label in self.legalLabels:
        for feature in self.features:
          Likelihood_phi[(feature, label)] = Likelihood_phi[(feature, label)] + k
          Counts[(feature, label)] = Counts[(feature, label)] + k + k
          # do not smooth the prior

#      pdb.set_trace()
      # normalize
      Prior.normalize()
      for key, value in Likelihood_phi.items():
        if value / Counts[key]*1.0 > 1:
          pdb.set_trace()
        Likelihood_phi[key] = value / Counts[key] * 1.0

 #     pdb.set_trace()
      self.prior = Prior
      self.likelihood = Likelihood_phi

      # validation
      predictions = self.classify(validationData)
      accuracyCount =  [predictions[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
      print "Performance on validation set for k=%f: (%.1f%%)" % (k, 100.0*accuracyCount/len(validationLabels))
      if accuracyCount > AccuracyCount:
        Parameter = (Prior, Likelihood_phi, k)
        AccuracyCount = accuracyCount

    self.prior, self.likelihood, self.k = Parameter      
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 #   util.raiseNotDefined()
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    
    "*** YOUR CODE HERE ***"""""""""""""""""""""""""""""""""""""""""""""
 #   pdb.set_trace()
    for label in self.legalLabels:
 #           pdb.set_trace()
            logJoint[label] = math.log(self.prior[label])
 #           pdb.set_trace()
            for feature, value in datum.items():
                if value > 0:
                  # probability of appearing
                  logJoint[label] = logJoint[label] + math.log(self.likelihood[feature,label])
                else:
                  # probability of not appear
                  logJoint[label] = logJoint[label] + math.log(1-self.likelihood[feature,label])
    # pdb.set_trace()
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   # util.raiseNotDefined()
    
    return logJoint

 #  pdb.set_trace()
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"""""""""""""""""""""""""""""""
    for feat in self.features:
      featuresOdds.append((self.conditionalProb[feat, label1]/self.conditionalProb[feat, label2], feat))
    featuresOdds.sort()
    featuresOdds = [feat for val, feat in featuresOdds[-100:]]

    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    util.raiseNotDefined()

    return featuresOdds
    

    
      
