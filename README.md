# Installation

## From Github
```bash
pip3 install git+https://github.com/er1kb/distinction
```
or clone and install locally:
```bash
git clone https://github.com/er1kb/distinction.git && cd distinction && pip3 install .
```

## From PyPI
```bash
python3 -m pip install distinction
```

## Dependencies
* [Numpy](https://numpy.org/) >= 1.25.0
* [SentenceTransformers](https://sbert.net/) >= 3.0.1
* [Plotext](https://github.com/piccolomo/plotext) >= 5.3.2 (optional)



# English

## What is it
A common use case is to be able to predict whether something is this or that, when that something is a piece of text. You may be working with custom logs (customer service requests, reviews, etc.) or open-ended survey responses that need to be coded at scale. Although neural networks can be used to classify latent variables in natural language, their complexity and overhead are a disadvantage. 

Embeddings are the features that neural networks use. They quantify information in the original data. Sentence transformer models encode meaning by producing sentence embeddings, ie representing text as high-dimensional vectors. These models are comparatively fast and lightweight and can even run on the cpu. Their output is easily stored in a vector database, so that you really only have to run the model once. Since vectors are points in an abstract space, we are able to measure if these points are close to each other (similar) or further apart (unrelated or opposite). 

Classification can be done by comparing the embedding of an individual text to a "typical" embedding for a given category. To "train" the classifier, you need a manually classified dataset. The minimum size of this dataset will depend on the number of dependent variables, how well-defined these variables are, and the ability of the sentence transformer model to encode relevant signals in your dataset. 

The classifier uses a relevant subset of the vector dimensions to separate signal from noise. A similarity threshold is chosen so that similarities at least equal to the threshold equal 1, and those below are assigned a 0. Comparisons are made at the level of individual sentences, which tend to be the main unit of coherent meaning. The classifier can be optimized/tuned by repeatedly running it on a validation dataset and selecting the threshold value with the best outcome. In the absence of validation data, this process can also be done manually. 

This tool has grown out of my work with customer service requests and other textual data. The idea of semantically grouping texts by calculating centroids is not new, but I needed a framework that would fit into our data pipelines and allow for some optimization. A catalyst of this work was when I failed at compiling Tensorflow for the GPU, after spending an entire workday on it. I had succeeded before but was still annoyed with the execution speed and the need for a gpu, as well as the size of saved models. Neural networks are cool and all, but backpropagation may not be the appropriate solution to every analytical problem. 


## Tutorial

Some things to consider before diving into the examples: 
* The quality of your predictions depends on the quality of training and prediction data. Specific categories yield a lot better prediction error than general ones. 
* If possible, train and predict at the sentence level, not on entire paragraphs. In the real world, text is rarely homogenous. For example, a review may have sentences that can be considered positive, negative or somewhere in between. Sentences are the best semantic unit, as hinted by the term "sentence transformer". Tools provided [here](#split-and-combine-records) and [here](#set-up-a-prediction-pipeline) allow you to predict sentences and then aggregate the general tendency. 
* Input data is an iterable (list/generator/tuple) of dicts. Export from your favourite dataset library using polars.DataFrame.[to\_dicts()](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.to_dicts.html) or pandas.DataFrame.[to\_dict('records')](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_dict.html).
* Results are returned as generators where possible to allow for lazy computation. To get a regular list back, you need to put the generator in a list and unpack it with an asterisk, eg "predictions = [\*predictions]". The lazy nature of generators also means that no computation is done until you unpack the generator or call next() on it to get the first element. 


### Set up a binary Classifier

<details>
<summary>Expand</summary>

This is our training data, a small sample of 21 manually coded short sentences. Notice the "suggestion" variable is formatted as strings, eg with quotation marks, so we need to do some data cleaning to get the counts. This is really the only time you should have to use this helper function directly. As the need for int conversion is a common scenario, especially in a dynamically typed language, the conversion will always be done automatically when you add data to the classifier. 

```python
from distinction import Classifier, count_used_keys, ones_to_int

data = [
    { "id": 0,  "message": "Greatest burgers I ever tasted", "positive": 1, "suggestion": "0", "taste": 1, "service": 0 },
    { "id": 1,  "message": "Your fries could be a bit more salty", "positive": 0, "suggestion": "1", "taste": 1 },
    { "id": 2,  "message": "Good service at the drive-in, those people should be given a raise", "positive": 1, "suggestion": 0, "taste": 0, "service": 1 },
    { "id": 3,  "message": "This is spam", "spam": 1 },
    { "id": 4,  "message": "I've never tasted such awful fries", "positive": 0, "suggestion": "0", "taste": 1, "service": 0 },
    { "id": 5,  "message": "Thanks for helping me to get around with my wheelchair!", "positive": 1, "suggestion": "0", "taste": 0, "service": 1 },
    { "id": 6,  "message": "Maybe upgrade your burger buns, the current ones are dry and boring", "positive": 0, "suggestion": "1", "taste": 1, "service": 0 },
    { "id": 7,  "message": "Is this how you handle your customers?", "positive": 0, "suggestion": "0", "taste": 0, "service": 1 },
    { "id": 8,  "message": "The salad tasted a bit different, I think you should put some oil on it.", "positive": 0, "suggestion": "1", "taste": 1, "service": 0 },
    { "id": 9,  "message": "This is the best place in town, and the staff are customer-oriented", "positive": 1, "suggestion": "0", "taste": 0, "service": 1 },
    { "id": 10, "message": "Too much pepper, please tell your chef to be more conservative with the seasoning", "positive": 0, "suggestion": "1", "taste": 1, "service": 0 },
    { "id": 11, "message": "I appreciate your help", "positive": 1, "suggestion": "0", "taste": 0, "service": 1 },
    { "id": 12, "message": "I think you should install security cameras on the parking lot", "positive": 0, "suggestion": "1", "taste": 0, "service": 0 },
    { "id": 13, "message": "I don't like their french fries, but the burgers are ok and the staff are usually helpful", "positive": 0, "suggestion": "0", "taste": 1, "service": 1 },
    { "id": 14, "message": "There should be more suitable options for vegans who are also keto and paleo crossfitters!", "positive": 0, "suggestion": "1", "taste": 0, "service": 1 },
    { "id": 15, "message": "Big Burger is trying to POISON us all with vegetable oils", "positive": 0, "suggestion": "0", "taste": 0, "service": 1 },
    { "id": 16, "message": "I complained about my burger and got a new one with just the right amount of seasoning - great customer service!", "positive": 1, "suggestion": "0", "taste": 1, "service": 1 },
    { "id": 17, "message": "Poor excuse for an establishment, why don't you just shut down?", "positive": 0, "suggestion": "1", "taste": 0, "service": 1 },
    { "id": 18, "message": "Great food, great location", "positive": 1, "suggestion": "0", "taste": 1, "service": 0 },
    { "id": 19, "message": "More spam", "positive": 0, "suggestion": "0", "taste": 0, "service": 0, "spam": 1 },
    { "id": 20, "message": "Great service, but not so great food", "positive": 1, "suggestion": "1", "taste": 1, "service": 1 }
]

binary_variables = 'positive suggestion taste service spam'.split()
data = [*ones_to_int(data, keys = binary_variables)] # Convert strings to int - this is done automatically by the classifier
print(count_used_keys(data, ignore = 'id message'))
```

Counts of targets in the training data:
```python
{'positive': 8, 'suggestion': 8, 'spam': 2, 'taste': 10, 'service': 11}
```

### Classifier from training\_data - raw text 

#### Initiate and train the classifier

<details>
<summary>Expand</summary>

First step is to define the classifier. Using a dict for keyword arguments means the arguments are reusable. We tell the classifier which columns are binary variables (targets). A confounder is a special kind of target, as it cannot be anything else. In this example, we don't want spam messages to taint our customer service statistics. 
The train() method calls the sentence transformer to encode the texts, then calculates the centroids of each target and finally ranks the features (embedding dimensions) by relevance. 

```python

kwargs = {
    'targets': 'positive suggestion taste service'.split(),
    'confounders': ['spam'],
    'id_columns': ['id'],
    'text_column': 'message',
    'default_selection': 0.05,
    'model': 'sentence-transformers/all-MiniLM-L6-v2'
}

C = Classifier(**kwargs)
C.train(data)

```

</details>


#### Predict
<details>
<summary>Expand</summary>


Let's try to classify a couple of new texts. This is just using default parameters: looking at 5% (=39) of the 768 embedding dimensions and classifying something as 1 if the similarity with its centroid is at least 0.5. The sample size of this example is too small to reliably [optimize](#optimize-the-classifier) the classifier.
```python
prediction = [*C.predict([{"message": "I really like the taste of these burgers."},
                          {"message": "The staff was really helpful"},
                          {"message": "This is definitely spam"}
                         ])]

```

These results are ok given the small sample size and lack of optimization, although the first one should have been classified as positive. 
```python
[{'message': 'I really like the taste of these burgers.', 'positive': 0, 'service': 0, 'suggestion': 0, 'taste': 1, 'spam': 0}, {'message': 'The staff was really helpful', 'positive': 1, 'service': 1, 'suggestion': 0, 'taste': 0, 'spam': 0}, {'message': 'This is definitely spam', 'positive': 0, 'service': 0, 'suggestion': 0, 'taste': 0, 'spam': 1}]

```


For this example, we can also run predict on the original data. Using training data to validate a model is considered bad practice because of the obvious risk of overfitting, but the results below still tell us that the classifier has picked up some relevant signals. 

```python
predictions = [*C.predict(data)]
print(f"{'PREDICTIONS':<40}TEXT")
for p in predictions:
    print(f"{', '.join([k for k,v in p.items() if k in (kwargs['targets'] + kwargs['confounders']) and v == 1]):40}{p['message']}")
```


```bash
PREDICTIONS                             TEXT
positive, taste                         Greatest burgers I ever tasted
suggestion, taste                       Your fries could be a bit more salty
positive, service                       Good service at the drive-in, those people should be given a raise
spam                                    This is spam
taste                                   I\'ve never tasted such awful fries
positive, service                       Thanks for helping me to get around with my wheelchair!
suggestion, taste                       Maybe upgrade your burger buns, the current ones are dry and boring
positive, service                       Is this how you handle your customers?
suggestion, taste                       The salad tasted a bit different, I think you should put some oil on it.
positive, service                       This is the best place in town, and the staff are customer-oriented
suggestion, taste                       Too much pepper, please tell your chef to be more conservative with the seasoning
positive, service                       I appreciate your help
suggestion                              I think you should install security cameras on the parking lot
service, taste                          I don\'t like their french fries, but the burgers are ok and the staff are usually helpful
service, suggestion, taste              There should be more suitable options for vegans who are also keto and paleo crossfitters!
service, taste                          Big Burger is trying to POISON us all with vegetable oils
positive, service, taste                I complained about my burger and got a new one with just the right amount of seasoning - great customer service!
service                                 Poor excuse for an establishment, why don't you just shut down?
positive, taste                         Great food, great location
spam                                    More spam
positive, service, suggestion, taste    Great service, but not so great food

```

</details>


#### Validate
<details>
<summary>Expand</summary>



</details>






<details>
<summary>Classifier from training\_data - pre-encoded</summary>
<br>
TODO: example
```
pass
```
</details>

</details>



### Optimize the Classifier

<details>
<summary>Expand</summary>


<details>
<summary>Tune similarity</summary>
<br>
TODO: example
```
pass
```
</details>

<details>
<summary>Tune selection</summary>
<br>
TODO: example
```
pass
```
</details>

<details>
<summary>Tune with plots</summary>
<br>
TODO: example
```
pass
```
</details>

<details>
<summary>Use optimized criteria from tune()</summary>
<br>
TODO: example
```
pass
```
</details>

</details>


### Portability: save and load models

<details>
<summary>Expand</summary>
Saved models are typically a few kilobytes on disk. 

Saving:
```python
C = Classifier(**kwargs)
[*C.train(data)]
C.to_npz('my_saved_model_file')
```

Loading:
```python
C = Classifier(**kwargs) # Initiate a new classifier
C.from_npz('my_saved_model_file') # Skip the training step by loading the previously trained parameters from disk
predictions = [*C.predict(some_new_data)]
```
</details>



### Split and combine records

<details>
<summary>Expand</summary>

```python
from distinction import Classifier, split_records, combine_records

example_text = [{'text': 'This is the first sentence. Is this the second? Sentence number 3', 'binary_variable': 1},
                {'text': 'This text is a single sentence.', 'binary_variable': 0}] 
```
Note: in this example we are hard-coding a binary variable, where you would use the classifier for one or more targets. 

<details open>
<summary>Default settings</summary>
<br>

#### Split
By default, texts are split into sentences and then split into chunks when the sentence exceeds 384 tokens (the maximum for current sentence transformer models). The chunks are numbered by chunk\_id, with the last one being -1. These default settings should be used for semantic classifier pipelines, although a couple of other parameters are available to tamper with. 

```python
sentences = [*split_records(example_text)]
print(sentences)
```

```python
[{'text': 'This is the first sentence. ', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 0}, {'text': 'Is this the second? ', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 1}, {'text': 'Sentence number 3', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 2}, {'text': 'This text is a single sentence.', 'binary_variable': 0, 'doc_id': 1, 'sentence_id': 0}]
```

#### Combine

```python
results = [*combine_records(sentences, binary_targets = ['binary_variable'])]
print(results)
```

Back to the original shape, with a document\_id added. 
```python
[{'doc_id': 0, 'text': 'This is the first sentence. Is this the second? Sentence number 3', 'binary_variable': 1}, {'doc_id': 1, 'text': 'This text is a single sentence.', 'binary_variable': 0}]
```



</details>


<details>
<summary>Max sequence length</summary>
<br>

#### Split
If needed, you can set a different max number of tokens. Since whitespace and punctuation counts as tokens, set max\_sequence\_length to double the number of words you want. 
```python
sentences = [*split_records(example_text, max_sequence_length = 8)]
print(sentences)
```
```python
[{'text': 'This is the first', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 0, 'chunk_id': 0}, {'text': ' sentence. ', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 0, 'chunk_id': -1}, {'text': 'Is this the second', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 1, 'chunk_id': 0}, {'text': '? ', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 1, 'chunk_id': -1}, {'text': 'Sentence number 3', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 2}, {'text': 'This text is a', 'binary_variable': 0, 'doc_id': 1, 'sentence_id': 0, 'chunk_id': 0}, {'text': ' single sentence.', 'binary_variable': 0, 'doc_id': 1, 'sentence_id': 0, 'chunk_id': -1}]
```

#### Combine


```python
results = [*combine_records(sentences, binary_targets = ['binary_variable'])]
print(results)
```

```python
[{'doc_id': 0, 'text': 'This is the first sentence. Is this the second? Sentence number 3', 'binary_variable': 1}, {'doc_id': 1, 'text': 'This text is a single sentence.', 'binary_variable': 0}]
```

</details>

<details>
<summary>Overlap</summary>
<br>

#### Split
An alternative strategy, possibly worse, is to split text with overlap. The example code below splits the text into chunks of no more than 10 tokens (5 words), with an overlap of 3 tokens (2 words and one whitespace/punctuation). If using the overlap parameter, you must remember to use it when [combining the texts](#combine-records) back together again. 
```python
sentences = [*split_records(example_text, per_sentence = False, max_sequence_length = 8, overlap = 3)]
print(sentences)
```
```python
[{'text': 'This is the first', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 0}, {'text': 'the first sentence. Is ', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 1}, {'text': '. Is this the second', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 2}, {'text': 'the second? Sentence number ', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 3}, {'text': ' number 3', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 4}, {'text': 'This text is a', 'binary_variable': 0, 'doc_id': 1, 'sentence_id': 0}, {'text': 'is a single sentence.', 'binary_variable': 0, 'doc_id': 1, 'sentence_id': 1}, {'text': ' sentence.', 'binary_variable': 0, 'doc_id': 1, 'sentence_id': 2}]
```

#### Combine

```python
results = [*combine_records(example_text, overlap = 3, binary_targets = ['binary_variable'])]
print(results)
```

```python
[{'doc_id': 0, 'text': 'This is the first sentence. Is this the second? Sentence number 3', 'binary_variable': 1}, {'doc_id': 1, 'text': 'This text is a single sentence.', 'binary_variable': 0}]
```

</details>
</details>



### Set up a prediction pipeline

<details>
<summary>Expand</summary>


<details>
<summary>Set up a prediction pipeline for continuous data streams</summary>
<br>
TODO: example
```
pass
```
</details>

<details>
<summary>...</summary>
<br>
TODO: example
```
pass
```
</details>

</details>



### Set up a Classifier for mutually exclusive binary variables

<details>
<summary>Expand</summary>

#### Reviews data

This example uses a subset of the [Reviews](https://www.kaggle.com/datasets/ahmedabdulhamid/reviews-dataset) dataset. Although the accuracy is not that good on individual sentences, after aggregating the result we get a typical accuracy of just below 90 %. This is despite the fact that the model is not trained on individual sentences.  

<details>
<summary>Code</summary>

```python
import sys
import csv
import random

from distinction import Classifier, split_records, combine_records

# https://www.kaggle.com/datasets/ahmedabdulhamid/reviews-dataset

labels = dict([(0, 'negative'), (1, 'positive')])
reviews_data = list()

with open('TrainingDataNegative.txt', 'r') as f:
    next(f) # skip header row
    for row in f:
        record = { 'text': row.strip(),
                   'negative': 1 }
        reviews_data.append(record)

with open('TrainingDataPositive.txt', 'r') as f:
    next(f) # skip header row
    for row in f:
        record = { 'text': row.strip(),
                   'positive': 1 }
        reviews_data.append(record)

random.shuffle(reviews_data)

training_data = reviews_data[:1000] # Training on the first 1000 rows, multiple sentences
prediction_data = reviews_data[1000:1500] # Predicting/validating on the next 500 rows
split_prediction_data = [*split_records(prediction_data, text_column = 'text')] # Split by sentence

kwargs = {
            'model': 'sentence-transformers/all-MiniLM-L6-v2',
            'text_column': 'text',
            'mutually_exclusive': True,
            'default_selection': 0.5
         }

C = Classifier(**kwargs)
C.train(training_data)

split_predictions = [*C.predict(split_prediction_data, validation = True)]

# n = 5 
# print('\n\n'.join([str(r) for r in split_predictions[:n * 2]]))
# print()
# print('_' * 100)
# print()

predictions = [*combine_records(split_predictions, 
                                text_column = 'text', 
                                original_data = prediction_data,
                                aggregation = 'mutually_exclusive')]

C.error()

for label in "negative positive".split():
    for i,_ in enumerate(predictions):
        if label in predictions[i]: 
            predictions[i]['actual'] = label
            predictions[i].pop(label)

# n = 3
# print(f'First {n} predictions (entire text):')
# print('\n\n'.join(str(record) for record in predictions[:n]))

print()
print('Overall accuracy:')
correct = [p['actual'] == p['predicted'] for p in predictions ]
print(round(sum(correct) / len(predictions), 2))

sys.exit(0)

```

</details>

<details>
<summary>Output</summary>
<br>

The confusion matrix below shows you the rate of misclassification (outside of the diagonal) on individual sentences. Then the overall accuracy is calculated on the original texts after putting them back together. The overall performance is obviously better. 

```bash
❯ time python3 reviews_classification.py
Batches: 100%|██████████████████████████████████████████████████████████| 32/32 [00:01<00:00, 24.50it/s]
Done encoding training data

Batches: 100%|███████████████████████████████████████████████████████| 136/136 [00:00<00:00, 187.75it/s]
Done encoding prediction data


TARGETS             OVERALL             FALSE POSITIVE      FALSE NEGATIVE
----------------------------------------------------------------------------------------------------
negative            0.28                0.2                 0.08
positive            0.27                0.09                0.18


CONFUSION MATRIX
rows: validation/actual, columns: predicted, values sum to 1 (=100%)
Actual/Predicted % (row and column sum resp.) are calculated before rounding
--------------------------------------------------
          negat...  posit...       Actual %
negat...  0.24      0.09           0.33
posit...  0.2       0.49           0.67

Pred. %   0.44      0.58           1


Overall accuracy:
0.88

real	0m7,069s
user	0m8,649s
sys	    0m3,801s
```

</details>


#### News data

This next example uses a subset of the [AG's News Corpus](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) dataset. The typical accuracy is about 84 %. 
Each news article can belong to only one category. There is however some semantic overlap. For example, consider this title of a _world_ news article: "Rights group slams Iraqi trials". It was predicted to be a _sports_ article with a high degree of certainty. Slamming people is generally associated with the sports domain.  

<details>
<summary>Input</summary>
<br>

```python
import sys
import csv
import random

from distinction import Classifier, split_records, combine_records

labels = dict([(1, 'world'), (2, 'sports'), (3, 'business'), (4, 'science')])
ag_data = list()

with open('train.csv', 'r') as f:
    csvreader = csv.reader(f)
    _ = next(csvreader) # skip header row
    for row in csvreader:
        label = labels.get(int(row[0]))
        record = { 'text': row[2],
                   label: 1 }
        ag_data.append(record)

random.shuffle(ag_data)

training_data = ag_data[:1000] # Training on the first 1000 rows

prediction_data = ag_data[1000:1500] # Predicting/validating on the next 500 rows

split_prediction_data = [*split_records(prediction_data, text_column = 'text')]
print('Number of split sentences #1: ', len(split_prediction_data))
print()
split_prediction_data = [record for record in split_prediction_data if len(record.get('text')) > 10]
print('Number of split sentences #2: ', len(split_prediction_data)) # After filtering texts with > 10 character
print()

kwargs = {
            'model': 'sentence-transformers/all-MiniLM-L6-v2',
            'text_column': 'text',
            'mutually_exclusive': True,
            'default_selection': 0.65
         }

C = Classifier(**kwargs)
C.train(training_data)

split_predictions = [*C.predict(split_prediction_data, validation = True)]

n = 5 
print('\n\n'.join([str(r) for r in split_predictions[:n * 2]]))
print()
print('_' * 100)
print()

predictions = [*combine_records(split_predictions, 
                                text_column = 'text', 
                                original_data = prediction_data,
                                aggregation = 'mutually_exclusive')]

for topic in "world sports business science".split():
    for i,_ in enumerate(predictions):
        if topic in predictions[i]: 
            predictions[i]['actual'] = topic
            predictions[i].pop(topic)

print('First {n} predictions (entire text):')
print('\n\n'.join(str(record) for record in predictions[:n]))

print()
print('Overall accuracy:')
correct = [p['actual'] == p['predicted'] for p in predictions ]
print(sum(correct) / len(predictions))

sys.exit(0)
```
</details>

<details>
<summary>Output</summary>
<br>

A small number of predictions are shown, first at the sentence level and then concatenated back to their original form. 

The first example, where Wayne Rooney's arrival at Manchester United was originally and somewhat surprisingly in the _world_ category, although predicted as _sports_ with a score of 0.33. 
The second example with doc\_id = 1 is about arthritis and consists of two sentences. The first sentence is predicted as _science_ with a score of 0.13, which is not a high level of certainty. The second sentence is labelled _business_ with a slightly better score of 0.35. The winner in this case is _business_, which also turns out to be the original label.  


```bash
❯ time python3 ag_split_description.py
Number of split sentences #1:  946

Number of split sentences #2:  857

Batches: 100%|██████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 37.67it/s]
Done encoding training data

Batches: 100%|█████████████████████████████████████████████████████████| 27/27 [00:00<00:00, 147.39it/s]
Done encoding prediction data

{'doc_id': 0, 'sentence_id': 0, 'text': "Wayne Rooney arrives at Man Utd's training ground for a medical ahead of a 25m move from Everton.", 'world': 1, 'predicted': 'sports', 'score': 0.33}

{'business': 1, 'doc_id': 1, 'sentence_id': 0, 'text': 'There are many other treatment options for people with arthritis, and physicians are considering them patient by patient. ', 'predicted': 'science', 'score': 0.13}

{'business': 1, 'doc_id': 1, 'sentence_id': 1, 'text': 'Washington -- Physicians are pulling out their risk-versus-benefit calculators once ', 'predicted': 'business', 'score': 0.35}

{'doc_id': 2, 'science': 1, 'sentence_id': 1, 'text': 'com - A new research station at the bottom of the world may give future Antarctica researchers some special treats, like the ability to live above ground and look out a window.', 'predicted': 'science', 'score': 0.33}

{'doc_id': 3, 'sentence_id': 0, 'text': 'Namibian President Sam Nujoma #39;s chosen successor, Hifikepunye Pohamba, has won a landslide victory with 75 of the vote in the country #39;s third elections since independence, according to official results.', 'world': 1, 'predicted': 'world', 'score': 0.37}

{'business': 1, 'doc_id': 4, 'sentence_id': 0, 'text': 'The planned acquisition of Sears, Roebuck and Co. ', 'predicted': 'business', 'score': 0.48}

{'business': 1, 'doc_id': 4, 'sentence_id': 1, 'text': 'by Kmart Holding Corp. ', 'predicted': 'business', 'score': 0.51}

{'business': 1, 'doc_id': 4, 'sentence_id': 2, 'text': 'highlights a changing retail environment that could soon eliminate the department store as we know it, analysts and consultants said on Friday.', 'predicted': 'business', 'score': 0.38}

{'doc_id': 5, 'science': 1, 'sentence_id': 0, 'text': 'AP - The California Academy of Sciences in Golden Gate Park held a one-of-a-kind yard sale Sunday, offering rock-bottom prices on everything from antique wooden incubators to six-foot-tall prehistoric bird replicas.', 'predicted': 'science', 'score': 0.24}

{'doc_id': 6, 'sentence_id': 0, 'text': 'Federal Labor leader Mark Latham says the Prime Minister needs to face up to the reality that there were no stockpiles of weapons of mass destruction in Iraq.', 'world': 1, 'predicted': 'world', 'score': 0.4}

____________________________________________________________________________________________________

First {n} predictions (entire text):
{'doc_id': 0, 'text': "Wayne Rooney arrives at Man Utd's training ground for a medical ahead of a 25m move from Everton.", 'predicted': 'sports', 'score': [0.33], 'actual': 'world'}

{'doc_id': 1, 'text': 'There are many other treatment options for people with arthritis, and physicians are considering them patient by patient. Washington -- Physicians are pulling out their risk-versus-benefit calculators once ', 'predicted': 'business', 'score': [0.13, 0.35], 'actual': 'business'}

{'doc_id': 2, 'text': 'com - A new research station at the bottom of the world may give future Antarctica researchers some special treats, like the ability to live above ground and look out a window.', 'predicted': 'science', 'score': [0.33], 'actual': 'science'}

{'doc_id': 3, 'text': 'Namibian President Sam Nujoma #39;s chosen successor, Hifikepunye Pohamba, has won a landslide victory with 75 of the vote in the country #39;s third elections since independence, according to official results.', 'predicted': 'world', 'score': [0.37], 'actual': 'world'}

{'doc_id': 4, 'text': 'The planned acquisition of Sears, Roebuck and Co. by Kmart Holding Corp. highlights a changing retail environment that could soon eliminate the department store as we know it, analysts and consultants said on Friday.', 'predicted': 'business', 'score': [0.48, 0.51, 0.38], 'actual': 'business'}

Overall accuracy:
0.836

real	0m6,752s
user	0m6,727s
sys	    0m3,597s

```


</details>

</details>


