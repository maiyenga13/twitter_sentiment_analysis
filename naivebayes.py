from collections import defaultdict
import math

#TODO: make more general? takes in number of labels
# use smoothing 
class NaiveBayesClassifier(object):
    def __init__(self, n, training_doc, training_labels, smoothing):
        self.n = n #n-gram number
        self.training_doc = list(open(training_doc, 'r')) #training documents
        self.training_labels = list(open(training_labels, 'r')) #training labels
        
        #Dictionaries containing frequency of each word in docs of each label/class
        self.pos_dict = defaultdict(int)
        self.neutral_dict = defaultdict(int)
        self.neg_dict = defaultdict(int)
        self.dicts = {'positive': self.pos_dict, 'negative': self.neg_dict, 'neutral': self.neutral_dict}
        
        #Priors = #docs of class c / #total docs
        self.priors = defaultdict(int)

        #Total number words in each class (not unique)
        self.totals = defaultdict(int)
    

        self.smoothing = smoothing # 0 is no smoothing, 1 is LaPlace, 2 is backoff and LaPlace
        self.all_words = set() #set of all words across all types
        
        self.build()
               
    #Iterates through each document in training set and calculates frequency of each word in each class
    #Calculates prior probabilities 
    def build(self):
       
        if(len(self.training_doc) != len(self.training_labels)):
            print("Error: number of documents and labels not consistent")
            return
        
        for i in range(len(self.training_doc)//2): #use half of the labeled data for training
            #Pre-process
            doc = self.preprocess(self.training_doc[i]) 
            label = self.preprocess(self.training_labels[i])[0]
            
            self.priors[label] += 1
            self.totals[label] += len(doc)
            self.parseLine(self.dicts[label], doc)

        #Calculate priors
        num_docs = len(self.training_doc) // 2
        
        self.priors['positive'] = self.priors['positive'] / num_docs
        self.priors['negative'] = self.priors['negative'] / num_docs
        self.priors['neutral'] = self.priors['neutral'] / num_docs

    #Remove trailing return characters and URLs
    def preprocess(self, doc):
        doc = doc.replace('\n', '')
        idx = doc.find('http')
        if(idx >= 0):
            doc = doc[0:idx]
        doc = doc.split(" ")
        return doc

    #TODO: add support for n grams
    #Parses a single document by increasing the frequency of each word 
    # and added each word to the set collecting all known words from training         
    def parseLine(self, dictionary, line):
        #words = line.split(" ")
        for word in line: 
            dictionary[word] += 1
            self.all_words.add(word)

    #Calculates likelihood of a line having the provided label
    def calculate(self, label, line):
        prior = self.priors[label]
        dictionary = self.dicts[label]
        total = self.totals[label]
        
        likelihood = 0
        if(prior != 0):
            likelihood = math.log(prior)
        
        for word in line: 
            if(word in self.all_words): #ignores unknown words
                likelihood += math.log((dictionary[word] + 1)/(total + len(self.all_words)))
        return likelihood

    #Determines whether a classification is a true positive, false positive or false negative 
    def makeConfusionMatrix(self, pred_label, actual_label, label, results):
        if(pred_label == actual_label and pred_label == label):
            results['tp'] += 1
        if(pred_label == label and pred_label != actual_label):
            results['fp'] += 1
        if(actual_label == label and pred_label != actual_label):
            results['fn'] += 1

    #Calculates precision and recall
    def calcPrecisionRecall(self, results):
        recall = results['tp'] / (results['tp'] + results['fn'])
        precision = results['tp'] / (results['tp'] + results['fp'])
        return [recall, precision]


    #Compares most likely label with actual and calculates precision and recall
    def test(self):  
        labels = ['positive', 'neutral', 'negative']
        pos_results = defaultdict(int)
        neg_results = defaultdict(int)
        neutral_results = defaultdict(int)
        results = {'positive': pos_results, 'negative': neg_results, 'neutral': neutral_results}
        
        for i in range(len(self.training_doc)//2, len(self.training_doc)):
            line = self.preprocess(self.training_doc[i])
            actual_label = self.preprocess(self.training_labels[i])[0]
            
            likelihoods = list(map(lambda label : self.calculate(label, line), labels))
            pred_label = labels[likelihoods.index(max(likelihoods))]

            for label in labels: 
                self.makeConfusionMatrix(pred_label, actual_label, label, results[label])
            
        avg_recall = 0
        avg_precision = 0
        for label in labels:
            dictionary = results[label]
            [recall, precision] = self.calcPrecisionRecall(dictionary)
            print(label, '- recall: ', recall, " , precision: " , precision)
            avg_recall += recall
            avg_precision += precision

        
        avg_recall = avg_recall / 3
        avg_precision = avg_precision / 3
        print("Average recall: " , avg_recall , " Average Precision: " , avg_precision)


if __name__ == "__main__":
    nb = NaiveBayesClassifier(1, "dev_text.txt", "dev_label.txt", 0)
    nb.test()


