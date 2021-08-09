# twitter_sentiment_analysis
Naive Bayes classifier currently using unigram model with LaPlace smoothing to predict a tweet as 'positive', 'neutral', or 'negative'
TODO: 
- add bigram and trigrams with backoff
- further smoothing
- currently ignores unknown, may need to change this

Results: unigram with LaPlace
    Positive- recall:  0.4415954415954416  , precision:  0.5636363636363636
    Neutral- recall:  0.7967479674796748  , precision:  0.5497896213183731
    Negative- recall:  0.03184713375796178  , precision:  0.4166666666666667
    Average recall:  0.4233968476110261  Average Precision:  0.5100308838738011
