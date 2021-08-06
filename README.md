# twitter_sentiment_analysis

-Gameplan: 
    - Build Naive Bayes classifier with features as unigram, bigram, and trigram
        - one naive bayes class, different methods to train and test for each type? 
            - takes in train and test data + method?
            - method to parse data files
            - calculate and print out recall + precision
            - also print out confusion matrix
    - Improve with word2vec using https://www.kaggle.com/c/word2vec-nlp-tutorial
    - Try Support Vector Machine
    - Use Neural Nets?? (using NLP hw stuff)


Results: unigram with LaPlace
    Positive- recall:  0.4415954415954416  , precision:  0.5636363636363636
    Neutral- recall:  0.7967479674796748  , precision:  0.5497896213183731
    Negative- recall:  0.03184713375796178  , precision:  0.4166666666666667
    Average recall:  0.4233968476110261  Average Precision:  0.5100308838738011
