#################################################################
##  Script Info: SemanticLexiconInduction Classifier
##  Author: Mohammed Habibllah Baig 
##  Date : 12/03/2017
#################################################################

import sys
import os
import math
import re

class SemanticLexiconInduction:
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
        """SemanticLexiconInduction initialization"""
        self.numFolds = 10
        '''index\d* in the regular expression is becuase I have attached the corresponding word index for each tag to optimize the search(suggestion on piazza post)'''
        self.pattern = re.compile("(JJindex\d* NN[S]?index\d*)|(RB[S]?[R]?index\d* JJindex\d* (?![NN][S]?))|(JJindex\d* JJindex\d* (?![NN][S]?))|(NN[S]?index\d* JJindex\d* (?![NN][S]?))|(RB[R]?[S]?index\d* VB[D]?[N]?[G]?index\d* )")
        self.index_pattern= re.compile("(\d+)")
        self.great="great"
        self.poor="poor"
        self.positive_phrase_hit_count={}
        self.negative_phrase_hit_count={}
        self.great_count=0.0
        self.poor_count=0.0
        self.Polarity_of_phrase={}


    def CalculateSemanticOrientation(self):
        '''Calculate the semantic orientation of each phrase in the dictionary'''
        for original_phrase in self.positive_phrase_hit_count.keys():
            if self.positive_phrase_hit_count[original_phrase]>=4 or self.negative_phrase_hit_count[original_phrase]>=4:
                first_log = math.log(self.positive_phrase_hit_count[original_phrase]*self.poor_count,2)
                second_log = math.log(self.negative_phrase_hit_count[original_phrase]*self.great_count,2) 
                self.Polarity_of_phrase[original_phrase]= first_log - second_log             

    def addExample(self, klass, words):
        original_word_list=[]
        #Parts of speech Tags
        POS_tags=[]
        index=0
        for word in words:
            splitted_word=word.split('_')
            original_word=splitted_word[0]
            #Attaching the index of each pattern so that we dont need to track them seperately.
            POS_tags.append(splitted_word[1]+"index"+str(index)) 
            if original_word==self.great:
                self.great_count+=1
            elif original_word==self.poor:
                self.poor_count+=1
            original_word_list.append(original_word)
            index+=1
        POS_tags= ' '.join(POS_tags)
        #print POS_tags
        #print(self.great_count)
        #print(self.poor_count)
        RE_matching_patterns=[]
        #regular expression matching for extracted relevant tags  
        RE_matching_patterns.extend(self.pattern.findall(POS_tags))
        
        '''Function to check the no of occurence of the word great or bad in the window_size of current index'''
        def IRSearchNearness(window_size, cur_index, phrase_orientation):
            count=0.01
            doc_length=len(original_word_list)
            if cur_index-window_size <0:
                window_start=0
            else:
                window_start=cur_index-window_size
            if cur_index+window_size+2>doc_length:
                window_end=doc_length
            else:
                window_end=cur_index+window_size+2
            if cur_index+2>doc_length-1:    
                window_right=doc_length-1
            else:
                window_right=cur_index+2
            #To search great or bad on the left side of current index
            for j in range(window_start, cur_index):
                if original_word_list[j]==phrase_orientation:
                    count+=1.0
            # To search great or bad on the right side of Current index
            for j in range(window_right,window_end):
                if original_word_list[j]==phrase_orientation:
                    count += 1.0
            return count
            
        for pattern in RE_matching_patterns:
            pattern=''.join(pattern) #to join the tuple
            splitted_pattern=pattern.split(' ')
            #This will give the inde of orginal word for the matched pattern
            index=self.index_pattern.findall(splitted_pattern[0]) 
            index=int(index[0])
            #print(index)
            original_phrase=original_word_list[index]+" "+original_word_list[index+1]
            #print(original_phrase)
            #update the phrase hit counts in the global dictionary
            self.positive_phrase_hit_count[original_phrase]= self.positive_phrase_hit_count.get(original_phrase, 0.0)+ IRSearchNearness(12,index,self.great)
            self.negative_phrase_hit_count[original_phrase] = self.negative_phrase_hit_count.get(original_phrase, 0.0) + IRSearchNearness(12,index, self.poor)
    
    def classify(self, words):
        original_word_list = []
        POS_tags = []
        index = 0
        for word in words:
            splitted_word = word.split('_')
            #print(splitted_word)
            POS_tags.append(splitted_word[1] +"index"+str(index))
            original_word = splitted_word[0]
            #print(original_word)
            original_word_list.append(original_word)
            index += 1
        #POS=parts of speech tags
        POS_tags = ' '.join(POS_tags)
        RE_matching_patterns = []
        '''returns the list of all matching patterns'''
        RE_matching_patterns.extend(self.pattern.findall(POS_tags))

        doc_polarity=0
        for pattern in RE_matching_patterns:
            pattern=''.join(pattern) #to join the tuples from findall function as a string
            splitted_pattern=pattern.split(' ')
            #print(splitted_pattern)
            '''extract the index from the matched tags to get the original phrase'''
            index=self.index_pattern.findall(splitted_pattern[0]) 
            index=int(index[0])
            original_phrase=original_word_list[index]+" "+original_word_list[index+1]
            #print(original_phrase)
            doc_polarity+=self.Polarity_of_phrase.get(original_phrase,0)

        if doc_polarity > 0:
            SemanticOrientation = 'pos'
        else:
            SemanticOrientation = 'neg'
        return SemanticOrientation

    def readFile(self, fileName):
        """
         * Code for reading a file.  you probably don't want to modify anything here,
         * unless you don't like the way we segment files.
        """
        contents = []
        f = open(fileName)
        for line in f:
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

    def train(self, split):
        for example in split.train:
            words = example.words
            self.addExample(example.klass, words)

    def crossValidationSplits(self, trainDir):
        """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
        splits = []
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        # for fileName in trainFileNames:
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
    nb = SemanticLexiconInduction()
    splits = nb.crossValidationSplits(args[0])
    avgAccuracy = 0.0
    fold = 0
    for split in splits:
        classifier = SemanticLexiconInduction()
        accuracy = 0.0
        for example in split.train:
            words = example.words
            classifier.addExample(example.klass, words)

        classifier.CalculateSemanticOrientation()

        for example in split.test:
            words = example.words
            guess = classifier.classify(words)
            if example.klass == guess:
                accuracy += 1.0

        #print(accuracy, len(split.test))

        accuracy = accuracy / len(split.test)
        avgAccuracy += accuracy
        print('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy))
        fold += 1
    avgAccuracy = avgAccuracy / fold
    print('[INFO]\tAccuracy: %f' % avgAccuracy)


def classifyDir( trainDir, testDir):
    classifier = SemanticLexiconInduction()
    trainSplit = classifier.trainSplit(trainDir)
    classifier.train(trainSplit)
    testSplit = classifier.trainSplit(testDir)
    accuracy = 0.0

    classifier.CalculateSemanticOrientation()

    for example in testSplit.train:
        words = example.words
        guess = classifier.classify(words)
        if example.klass == guess:
            accuracy += 1.0
    accuracy = accuracy / len(testSplit.train)
    print('[INFO]\tAccuracy: %f' % accuracy)


def main():
    args=sys.argv[1:]

    if len(args) == 2:
        classifyDir( args[0], args[1])
    elif len(args) == 1:
        test10Fold(args)


if __name__ == "__main__":
    main()
