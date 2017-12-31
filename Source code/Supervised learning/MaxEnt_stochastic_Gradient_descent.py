import sys
import getopt
import os
import math
import operator
import re
import numpy as np
from collections import Counter


class Maxent:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """Maxent initialization"""
    self.stopList = set(self.readFile('../data/stop_words.txt'))
    self.numFolds = 10
    self.bag_list = []
    self.weights = []
    

  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Maxent classifier 

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    weights=self.weights
    bag_list=self.bag_list
    words=self.filterStopWords(words)
    fv=np.zeros(len(bag_list)+1)
    fv[0]=1
    i=1
    for word in bag_list:
        if word in words:
            fv[i]=1
        i=i+1
    temp=fv.dot(weights)
    temp=math.exp(temp)
    t_prob=1.0/(1.0+temp)
    #print(t_prob)
    #print(fv)
    
    if t_prob >= 0.5:
        return 'pos'
    else:
        return 'neg'
    
    # Write code here

    

  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the Maxent class.
     * Returns nothing
    """

    # Write code here

    pass
 
  def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered
    
  def train(self, split, epsilon, eta, lambdaa):
      """
      * TODO 
      * iterates through data examples
      """
      bag=[]
      pos_bag=[]
      neg_bag=[]      
      
      
      
      train_size=len(split.train)
      train_half=int(train_size/2)
      temp_train=split.train[:]
      temp_train[::2]=split.train[:train_half]
      temp_train[1::2]=split.train[-train_half:]
      split.train=temp_train
      split.train=split.train[:200]
      
      
      for example in split.train:
          #print(len(split.train))
          example.words = self.filterStopWords(example.words)
          for word in example.words:
              #word=word.lower()
              bag.append(word)
              if example.klass == 'pos':
                  pos_bag.append(word)
              elif example.klass == 'neg':
                  neg_bag.append(word)
              else:
                  print("class not found")
      
      cnt = Counter(pos_bag)
      pos_set=set(k for k, v in cnt.items() if v > 10)
      pos_list=list(pos_set)
      cnt = Counter(neg_bag)
      neg_set=set(k for k, v in cnt.items() if v > 10)
      neg_list=list(neg_set)
      bag_list= pos_list + neg_list
      bag_set=set(bag_list)
      
      #print ("neg_set : %d , pos set : %d total set :  %d", (len(neg_set), len(pos_set),len(bag_set)))
      '''
      for word in bag_list:
          if word in pos_set:
              if word in neg_set:
                  pos_set.remove(word)
                  neg_set.remove(word)
                  bag_set.remove(word)'''
               
      '''thefile = open('bag_words.txt', 'w')
      for item in bag_list:
          thefile.write("%s\n" % item)
      
      #bag_list=list(bag_set)
      pos_list=list(pos_set)
      neg_list=list(neg_set)
      bag_list=pos_list+neg_list'''
      
      '''print("Total No of words : " , len(bag_list),len(bag_set))
      print("Positive words : " , len(pos_list) , len(pos_set))
      print("Negative words : " , len(neg_list) , len(neg_set))'''

      
      bag_size=len(bag_list)
      pos_size=len(pos_list) 
      neg_size=len(neg_list)
      
      weights=[0]*(bag_size+1)
      updated_weights=[0]*(bag_size+1)      
      prob=[0]*train_size
      diff_prob=[0]*train_size
      
      fv=np.zeros(shape=(train_size,bag_size+1))
      
      j=0
      for example in split.train:
          temp_fv=[0]*(bag_size+1)
          temp_fv[0]=1
          i=1
          ex=set(example.words)
          for word in bag_list:
              if word in ex:
                  temp_fv[i]=1
              i=i+1
          fv[j]=temp_fv
          j+=1
          
      def calc_prob(cur_doc,k):
          temp=fv[cur_doc].dot(weights)
          #print(temp)
          temp=math.exp(temp)
          temp=1/(1+temp)
          prob[cur_doc]=temp
          if k == 'pos':
              #print("here")
              diff_prob[cur_doc]=1-temp
          else:
              #print("Not here")
              diff_prob[cur_doc]=0-temp 
           
                  
       
      def sum_error(cur_doc,cur_weight,k):
          if cur_weight==0:
              s=sum(diff_prob[:cur_doc+1])/(cur_doc+1)
          else:
              calc_prob(cur_doc,k)
              #for pros_doc in range(0,cur_doc+1):
                  #print("calculating probablity for doc , weight", cur_doc, cur_weight) 
              temp=fv[:cur_doc+1,[cur_weight]].reshape(1,cur_doc+1)
              s=(temp[0].dot(diff_prob[:cur_doc+1]))/(cur_doc+1)
          return s
        
      cur_doc=0
      for example in split.train:
          change=99
          while change > epsilon:
              change=0
              for cur_weight in range(0,len(weights)):
                  diff = eta*sum_error(cur_doc,cur_weight,example.klass)
                  regularization=sum(i for i in weights)
                  weights[cur_weight] = weights[cur_weight] - diff - lambdaa*regularization
                  change+=diff
              change=change/bag_size
              #weights=updated_weights
          #print("current change :", change, "current doc :", cur_doc )
          cur_doc+=1
    
      self.weights=weights
      self.bag_list=bag_list
      '''thefile = open('weights.txt', 'w')
          for item in weights:
              thefile.write("%s\n" % item)
      
      
          print("iteration : %d,  diff : %f " %(i,diff))'''
      
      
      #  probablity(words)
          #print(words)
          #self.addExample(example.klass, words)
      

  # END TODO (Modify code beyond here with caution)
  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName,encoding="latin-1")
    for line in f:
      line=line.lower()
      line=re.sub("[^\sa-z]",'',line)
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits
  

def test10Fold(args):
  pt = Maxent()
  
  splits = pt.crossValidationSplits(args[0])
  
  epsilon = float(args[1])
  eta = float(args[2])
  lambdaa = float(args[3])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = Maxent()
    accuracy = 0.0
    print(split)
    classifier.train(split, epsilon, eta, lambdaa)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print ('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)) 
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print ('[INFO]\tAccuracy: %f' % avgAccuracy)
    
    
def classifyDir(trainDir, testDir, eps, et, lamb):
  classifier = Maxent()
  trainSplit = classifier.trainSplit(trainDir)
  #print(trainSplit)
  epsilon = float(eps)
  eta = float(et)
  lambdaa = float(lamb)
  classifier.train(trainSplit, epsilon, eta, lambdaa)
  testSplit = classifier.trainSplit(testDir)
  #testFile = classifier.readFile(testFilePath)
  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print ('[INFO]\tAccuracy: %f' % accuracy)
    
def main():
  (options, args) = getopt.getopt(sys.argv[1:], '')
  
  if len(args) == 5:
    classifyDir(args[0], args[1], args[2], args[3], args[4])
  elif len(args) == 4:
    test10Fold(args)

if __name__ == "__main__":
    main()