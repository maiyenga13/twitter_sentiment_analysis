from collections import defaultdict
import math
#from typing import DefaultDict

#TODO: make more general? takes in number of labels 
class NaiveBayesClassifier(object):
    def __init__(self, n, training_doc, training_labels, smoothing):
        self.n = n
        self.training_doc = list(open(training_doc, 'r')) #training documents
        self.training_labels = list(open(training_labels, 'r')) #training labels
        #Dictionaries
        self.pos_dict = defaultdict(int)
        self.neutral_dict = defaultdict(int)
        self.neg_dict = defaultdict(int)
        #Priors = #docs of class c / #total docs
        self.pos = 0
        self.neutral = 0
        self.neg = 0
        #Total number words in each class (not unique)
        self.total_pos = 0
        self.total_neutral = 0
        self.total_neg = 0
        
        self.smoothing = smoothing # 0 is no smoothing, 1 is LaPlace, 2 is backoff and LaPlace

        self.all_words = set() #set of all words across all types
        self.build()
               
    def build(self):
       
        if(len(self.training_doc) != len(self.training_labels)):
            print("Error: number of documents and labels not consistent")
            return
        for i in range(len(self.training_doc)//2):
            doc = self.training_doc[i].replace('\n', '')
            
            idx = doc.find('http')
            if(idx >= 0):
                doc = doc[0:idx]
            doc = doc.split(" ")
            
            label = self.training_labels[i].replace('\n', '')
            
            if(label == 'positive'):
                self.parseLine(self.pos_dict, doc)
                self.pos += 1
                self.total_pos += len(doc)
            elif(label == 'neutral'):
                self.parseLine(self.neutral_dict, doc)
                self.neutral += 1
                self.total_neutral += len(doc)
            else: 
                self.parseLine(self.neg_dict, doc)
                self.neg += 1
                self.total_neg += len(doc)

        self.pos = self.pos / (len(self.training_doc)//2)
        self.neutral = self.neutral / (len(self.training_doc)//2)
        self.neg = self.neg / (len(self.training_doc)//2)


    #TODO: add support for n grams         
    def parseLine(self, dictionary, line):
        #words = line.split(" ")
        for word in line: 
            dictionary[word] += 1
            self.all_words.add(word)
    def calculate(self, label, line):
        dictionary = None
        prior = 0
        total = 0 
        if(label == 'positive'):
            dictionary = self.pos_dict
            prior = self.pos
            total = self.total_pos
        elif(label == 'neutral'):
            dictionary = self.neutral_dict
            prior = self.neutral
            total = self.total_neutral
        else: 
            dictionary = self.neg_dict
            prior = self.neg
            total = self.total_neg
        
        
        likelihood = 0
        if(prior != 0):
            likelihood = math.log(prior)
        for word in line.split(" "): 
           
            if(word in self.all_words):
                
                likelihood += math.log((dictionary[word] + 1 )/( total + len(self.all_words)))
                
       
        return likelihood

    def test(self):
        labels = ['positive', 'neutral', 'negative']
        pos_tp = 0 #pred = actual = pos
        pos_fp = 0 #pred = pos, actual = not pos
        pos_fn = 0 #pred = not post, actual = pos
        neutral_tp = 0
        neutral_fp = 0
        neutral_fn = 0
        neg_tp = 0
        neg_fp = 0
        neg_fn = 0
        for i in range(len(self.training_doc)//2, len(self.training_doc)):
            line = self.training_doc[i].replace('\n', '')
            actual_label = self.training_labels[i].replace('\n', '')
            

            pos_likelihood = self.calculate('positive', line)
            neutral_likelihood = self.calculate('neutral', line)
            neg_likelihood = self.calculate('negative', line)
            likelihoods = [pos_likelihood, neutral_likelihood, neg_likelihood]
            pred_label = labels[likelihoods.index(max(likelihoods))]
            if(actual_label == 'positive'):
                if(pred_label == 'positive'):
                    pos_tp += 1
                elif(pred_label == 'neutral'):
                    pos_fn +=1
                    neutral_fp +=1
                else:
                    pos_fn += 1
                    neg_fp += 1
            elif(actual_label == 'neutral'):
                if(pred_label == 'positive'):
                    pos_fp += 1
                    neutral_fn += 1
                elif(pred_label == 'neutral'):
                    neutral_tp += 1
                else:
                    neutral_fn += 1
                    neg_fp += 1
            else: 
                if(pred_label == 'positive'):
                    pos_fp += 1
                    neg_fn += 1
                elif(pred_label == 'neutral'):
                    neutral_fp += 1
                    neg_fn +=1
                else:
                    neg_tp += 1
        pos_recall = pos_tp / (pos_tp + pos_fn)
        pos_precision = pos_tp / (pos_tp + pos_fp)
        neutral_recall = neutral_tp / (neutral_tp + neutral_fn)
        neutral_precision = neutral_tp / (neutral_tp + neutral_fp)
        neg_recall = neg_tp / (neg_tp + neg_fn)
        neg_precision = neg_tp / (neg_tp + neg_fp)
        avg_recall = (pos_recall + neutral_recall + neg_recall) / 3
        avg_precision = (pos_precision + neutral_precision + neg_precision) / 3

        print("Positive- recall: ", pos_recall , " , precision: " , pos_precision)
        print("Neutral- recall: " , neutral_recall , " , precision: " , neutral_precision)
        print("Negative- recall: " , neg_recall , " , precision: " , neg_precision)
        print("Average recall: " , avg_recall , " Average Precision: " , avg_precision)


    

if __name__ == "__main__":
    nb = NaiveBayesClassifier(1, "dev_text.txt", "dev_label.txt", 0)
    #print(nb.pos_dict)
    #print(nb.neg_dict)
    #print(nb.total_pos)
    #print(nb.total_neg)
    nb.test()
    #print(nb.calculate('negative', "@fuckzehk @shegavemethesuc lets play monopoly when gucci gets home since he doesn't have school tomorrow"))
    #print(nb.calculate('positive', "@fuckzehk @shegavemethesuc lets play monopoly when gucci gets home since he doesn't have school tomorrow"))
    #print(nb.calculate('neutral', "@fuckzehk @shegavemethesuc lets play monopoly when gucci gets home since he doesn't have school tomorrow"))


