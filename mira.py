# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Mira implementation
import util
PRINT = True

class MiraClassifier:
  """
  Mira classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "mira"
    self.automaticTuning = False 
    self.C = 0.001
    self.legalLabels = legalLabels
    self.max_iterations = max_iterations
    self.initializeWeightsToZero()

  def initializeWeightsToZero(self):
    "Resets the weights of each label to zero vectors" 
    self.weights = {}
    for label in self.legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use
  
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    "Outside shell to call your method. Do not modify this method."  
      
    self.features = trainingData[0].keys() # this could be useful for your code later...
    
    if (self.automaticTuning):
        Cgrid = [0.002, 0.004, 0.008]
    else:
        Cgrid = [self.C]
        
    return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
    """
    This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid, 
    then store the weights that give the best accuracy on the validationData.
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    representing a vector of values.
    """
    #############################################"*** YOUR CODE HERE ***"
    weights = {}

    def train_c(c):
      print "The constant constriant is ", c

      self.weights = dict((label, util.Counter()) for label in self.legalLabels)

      for iteration in range(self.max_iterations):
        print "This iteration is", iteration
        for i in range(len(trainingData)):
          features = trainingData[i]
          y_observe = trainingLabels[i]
##        for features, y_observe in zip(trainingData, trainingLabels): ##check
          y_predict = self.classify([features])[0] # check
          if y_predict != y_observe:
            tau = ((self.weights[y_predict] - self.weights[y_observe]) * features + 1.0) / ((features * features) * 2.0)
            tau_min = min([c, tau])
            updated = features.copy()
            for key, value in updated.items():
              updated[key] = value * tau_min
            self.weights[y_predict] = self.weights[y_predict] - updated
            self.weights[y_observe] = self.weights[y_observe] + updated

      weights[c] = self.weights
      validation_predict = self.classify(validationData) # check zip
      numbers = sum(int(y_observe == y_predict) for y_observe, y_predict in zip(validationLabels,validation_predict))
      return numbers
    
    c_scores = [train_c(c) for c in Cgrid]
#####################################################################
    # util.raiseNotDefined()

  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns a list of the 100 features with the greatest difference in feature values
                     w_label1 - w_label2

    """
    featuresOdds = []

    ###################################################"*** YOUR CODE HERE ***"
    diff = util.Counter() 
        
    for feature in self.features:
      diff[feature] = self.weights[label1][feature] - self.weights[label2][feature]

    features = self.features
    features.sort(lambda x, y: cmp(diff[x], diff[y])) #check

    featuresOdds = features[:100]
    #################################################################################
    return featuresOdds
