
## Installation

#### From Github
```bash
pip3 install git+https://github.com/er1kb/distinction
```
or clone and install locally:
```bash
git clone https://github.com/er1kb/distinction.git && cd distinction && pip3 install .
```

#### From PyPI
```bash
python3 -m pip install distinction
# or if you want to look at the error curves:
python3 -m pip install distinction[plot]
```

#### Dependencies
* [Numpy](https://numpy.org/) >= 1.25.0
* [SentenceTransformers](https://sbert.net/) >= 3.0.1
* [Plotext](https://github.com/piccolomo/plotext) >= 5.3.2 (optional)


## Overview
A common use case is to be able to predict whether something is this or that, when that something is a piece of text. You may be working with custom logs (customer service requests, reviews, etc.) or open-ended survey responses that need to be coded at scale. Although neural networks can be used to classify latent variables in natural language, their complexity and overhead are a disadvantage. 

Embeddings are high-dimensional features that neural networks use. They quantify information in the original data. Sentence transformer models encode meaning by producing sentence embeddings, ie representing text as high-dimensional vectors. These models are comparatively fast and lightweight and can even run on the cpu. Their output is easily stored in a vector database, so that you really only have to run the model once. Since vectors are points in an abstract space, we are able to measure if these points are close to each other (similar) or further apart (unrelated or opposite). 

Classification can be done by comparing the embedding of an individual text to a "typical" embedding for a given category. To "train" the classifier, you need a manually classified dataset. The minimum size of this dataset will depend on the number of dependent variables, how well-defined these variables are, and the ability of the sentence transformer model to encode relevant signals in your dataset. 

The classifier uses a relevant subset ("selection") of the vector dimensions to separate signal from noise. A similarity threshold is chosen as the decision boundary between 0 and 1. Ideally, comparisons are made at the level of individual sentences. The classifier can be optimized/tuned by repeatedly running it on a validation dataset and selecting the threshold values with the best outcome. In the absence of validation data, this process can also be done manually.

This tool has grown out of my work with customer service requests and other textual data. The idea of semantically grouping texts by calculating centroids is not new, but we needed a framework that would fit into our data pipelines and allow for some optimization. A catalyst of this work was when I failed at compiling Tensorflow for the GPU, after spending an entire workday on it. I had succeeded before but was still annoyed with the execution speed and the need for a gpu, as well as the size of saved models. Neural networks are cool and all, but backpropagation may not be the appropriate solution to every analytical problem. 


## How-to / Examples

Some things to consider before (or after) diving into the examples: 
* The quality of your predictions depends on the quality of training and prediction data. Specific categories yield a lot better prediction error than general ones. Use categories that are somewhat coherently expressed and not too broad. For example, _recycling_ and _energy\_consumption_ are more useful in this respect than _sustainability_. The latter is an emergent phenomenon that should be inferred from its subcategories. Our use case may be to identify all texts that are related to _sustainability_, but doing so directly would yield a much higher error rate due to greater variability in the data. 
* If possible, train and predict at the sentence level, not on entire paragraphs. In the real world, text is rarely homogenous. For example, a review may have sentences that can be considered positive, negative or somewhere in between. Sentences are the best semantic unit, as hinted by the term "sentence transformer". Tools provided [here](#split-and-combine-records) and [here](#set-up-a-prediction-pipeline-for-continuous-data-streams) allow you to predict sentences and then aggregate the general tendency. 
* Input data is an iterable (list/generator/tuple) of dicts. Export from your favourite dataframe library using polars.DataFrame.[to\_dicts()](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.to_dicts.html) or pandas.DataFrame.[to\_dict('records')](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_dict.html).
* Results are returned as generators where possible to allow for lazy computation (as needed). To get a regular list back, you need to put the generator in a list and [unpack](https://www.geeksforgeeks.org/convert-generator-object-to-list-in-python/) it with an asterisk, eg __predictions = [\*predictions]__. The lazy nature of generators also means that no computation is done until you unpack the generator or call next() on it to yield the first element. 



### Set up and use a binary Classifier for independent variables

<details>
<summary>Expand</summary>

This is our training data, a small sample of 21 manually coded short sentences. Notice the "suggestion" variable is formatted as strings, eg with quotation marks, so we need to do some data cleaning to get the counts. This is really the only time we should have to use the helper function _ones_to_int()_ directly. As the need for int conversion is a common scenario, especially in a dynamically typed language, the conversion will always be done automatically when you add data to the classifier. 

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
data = [*ones_to_int(data, keys = binary_variables)] # Convert strings to int - this is done automatically by the classifier later on
print(count_used_keys(data, ignore = 'id message'))
```

Counts of targets in the training data:
```python
{'positive': 8, 'suggestion': 8, 'spam': 2, 'taste': 10, 'service': 11}
```

#### Classifier from training\_data - raw text 

<details>
<summary>Initiate and train the classifier</summary>

First step is to define the classifier. Using a dict for keyword arguments means the arguments are reusable. We tell the classifier which columns are binary variables (targets). A confounder is a special kind of target, as it cannot be anything else. In this example, we don't want spam messages to taint our customer service statistics. 
The train() method below calls the sentence transformer to encode the texts, then calculates the centroids of each target and finally ranks the features (embedding dimensions) by relevance. 
<br><br>
If your text is not in English, remember to use another __model__ from the [Huggingface model hub](https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=downloads) that speaks your language. There are language-specific and multilingual ones to choose from. A model that doesn't understand your language will still produce results, although they won't make much sense. 

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



<details>
<summary>Predict</summary>


Let's try to classify a couple of new texts. This is just using default parameters: looking at 5% (=39) of the 768 embedding dimensions and classifying something as 1 if the similarity with its centroid is at least 0.5. The sample size of this example is too small to reliably [optimize](#optimize-the-classifier) the classifier.
```python
predictions = [*C.predict([{"message": "I really like the taste of these burgers."},
                          {"message": "The staff was really helpful"},
                          {"message": "This is definitely spam"}
                          ])]
for p in predictions:
    print(p)
```

These results are ok given the small sample size and lack of optimization, although the first one should have been classified as positive. Notice the third sample is spam and has all other targets set to 0, since _spam_ was declared to be a confounding variable. 
```bash
{'message': 'I really like the taste of these burgers.', 'positive': 0, 'service': 0, 'suggestion': 0, 'taste': 1, 'spam': 0}
{'message': 'The staff was really helpful', 'positive': 1, 'service': 1, 'suggestion': 0, 'taste': 0, 'spam': 0}
{'message': 'This is definitely spam', 'positive': 0, 'service': 0, 'suggestion': 0, 'taste': 0, 'spam': 1}
```


For this example, we can also run predict on the original data. Using training data to validate a model is considered bad practice because of the obvious risk of overfitting, but the results below still tell us that the classifier has picked up some relevant signals. 

```python
predictions = [*C.predict(data)]
print(f"{'PREDICTIONS':<40}TEXT")
for p in predictions:
    print(f"{', '.join([k for k,v in p.items() if k in (kwargs['targets'] + kwargs['confounders']) and v == 1]):40}{p['message']}")
```


```
PREDICTIONS                             TEXT
positive, taste                         Greatest burgers I ever tasted
suggestion, taste                       Your fries could be a bit more salty
positive, service                       Good service at the drive-in, those people should be given a raise
spam                                    This is spam
taste                                   I've never tasted such awful fries
positive, service                       Thanks for helping me to get around with my wheelchair!
suggestion, taste                       Maybe upgrade your burger buns, the current ones are dry and boring
positive, service                       Is this how you handle your customers?
suggestion, taste                       The salad tasted a bit different, I think you should put some oil on it.
positive, service                       This is the best place in town, and the staff are customer-oriented
suggestion, taste                       Too much pepper, please tell your chef to be more conservative with the seasoning
positive, service                       I appreciate your help
suggestion                              I think you should install security cameras on the parking lot
service, taste                          I don't like their french fries, but the burgers are ok and the staff are usually helpful
service, suggestion, taste              There should be more suitable options for vegans who are also keto and paleo crossfitters!
service, taste                          Big Burger is trying to POISON us all with vegetable oils
positive, service, taste                I complained about my burger and got a new one with just the right amount of seasoning - great customer service!
service                                 Poor excuse for an establishment, why don't you just shut down?
positive, taste                         Great food, great location
spam                                    More spam
positive, service, suggestion, taste    Great service, but not so great food
```

</details>



<details>
<summary>Validate</summary>
<br>

If there is validation data with the right answers, you can assess model performance using _predict(..., validation = True)_. For this example, we're going to cheat by re-using the training data listed above. The _Classifier.error()_ method will print the error by target variable. Validation also produces a list of prediction errors per row stored at _Classifier.error\_rate\_by\_row_. In the code below, note the [generator unpacking](https://www.geeksforgeeks.org/convert-generator-object-to-list-in-python/) of the prediction results to force the computation and produce the error rate. 
```python
_ = [*C.predict(data, validation = True)]
C.error()
```

As expected, we get an artificially low error rate since the model is overfitted to our training data. Rows 14 and 15 were predicted as taste. You could argue these messages are food related (keto/paleo and vegetable oils respectively), even though they were not manually coded as such. 

```bash
TARGETS             OVERALL             FALSE POSITIVE      FALSE NEGATIVE      THRESHOLD
----------------------------------------------------------------------------------------------------
positive            0.05                0.05                0.0                 0.5
service             0.0                 0.0                 0.0                 0.5
suggestion          0.05                0.0                 0.05                0.5
taste               0.1                 0.1                 0.0                 0.5

CONFOUNDERS         OVERALL             FALSE POSITIVE      FALSE NEGATIVE      THRESHOLD
----------------------------------------------------------------------------------------------------
spam                0.0                 0.0                 0.0                 0.5
```

</details>




#### Classifier from training\_data - pre-encoded

<details>
<summary>Expand</summary>
<br>

Although sentence transformers are fast compared to other neural networks, the encoding of text is the most time consuming part of the Classifier model and especially for large datasets. For the training stage, you can get around this by using a smaller sample. Another obvious way to save time and computation is if you have pre-existing embeddings in a vector database, such as [Elasticsearch](https://www.elastic.co/). You can then skip the encoding altogether by calling _train()_ and/or _predict()_ with the argument _pre\_encoded = True_. Ideally, you should never have to encode text more than once in a data pipeline. When using existing vectors, you need to specify _vector\_column_ instead of _text\_column_. These two sources are kept separate to allow for use cases where both raw text and embeddings are processed.  

In the following example, we encode the training data outside of the Classifier, but then we go back to using raw text from the prediction data. For raw text, use the _text\_column_ argument (if omitted, the default value being "text"). For vectors, use __either__ the _text\_column_ or the optional _vector\_column_ argument. The reason these two arguments exist side by side is you might want to predict using vectors while at the same time concatenating text - more on this in the section on pipelines. 
The code below uses an optional [pytorch](https://github.com/pytorch/pytorch) check to run the sentence transformer on the GPU. 

```python
from distinction import Classifier
from sentence_transformers import SentenceTransformer
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = SentenceTransformer(model_name_or_path = 'sentence-transformers/all-MiniLM-L6-v2',
                            device = device,
                            tokenizer_kwargs = {'clean_up_tokenization_spaces': True}) # this setting gets rid of a warning in Transformers 4.45.1

training_data = [dict(text='I am the great Cornholio', beavis=1), 
                 dict(text='I have seen the top of the mountain, and it is good', beavis=0)]

vectors = model.encode([r['text'] or '' for r in training_data], show_progress_bar = False)

for i,_ in enumerate(training_data):
    training_data[i]['vector'] = vectors[i] # Merge sentence embeddings into the original data


C = Classifier(text_column = 'vector', show_progress_bar = False) # Set to read encoded "text" for prediction. Optionally use vector_column = 'vector' instead, if you want to. 
C.train(training_data, pre_encoded = True) # Tell the classifier that the text column is an embedding

# Let's assume the prediction data is supplied as raw text
C.text_column = 'text' # Set to read proper text from the "text" column, as we predict with pre_encoded = False on the next line
results = C.predict([dict(text = 'I am Cornholio!')], pre_encoded = False)
print(next(results))
```

Output:
```bash
❯ python3 pre_encoded.py
Using device: cuda
Done encoding prediction data

{'text': 'I am Cornholio!', 'beavis': 1}
```


</details>

</details>



### Optimize the Classifier

<details>
<summary>Expand</summary>

This section can be run with the data and settings from the restaurant reviews example [above](#set-up-and-use-a-binary-classifier-for-independent-variables). Again, we will have to re-use the training data for the purpose of demonstration. The very small sample size means less chance of convergence, meaning the results are somewhat sketchy if not useless. Tuning the model uses repeated validation, which means the prediction data has to contain the right answers. 

Currently, the optimal similarity cutoff is determined to be the sweetspot between type 1 and type 2 errors. Minimizing the overall error works well for most targets, but for rare ones the model will just assign 0 to everything to achieve the best accuracy. The concept of [_imbalanced datasets_](https://discuss.pytorch.org/t/model-predictions-are-all-tensors-full-of-zeros/155381/11) is a known problem even for neural networks. Minimizing the difference between false positive and false negative rates does not yield the absolute minimum error, but it does seem to avoid having the model apply the null hypothesis to everything. It's on my todo list to improve on this, by putting some constraint on the optimization. In the meantime, you can use the plotting function to see whether there is room for improvement or not. 


<details>
<summary>Tune similarity</summary>
<br>

To find an estimate of the optimal similarity threshold, run the __Classifier.tune()__ method with default settings. The simulation will abort for each target when finding the optimal value, if not using plots (explained below), hence you might see the computation speed up towards the end. 

```python
C = Classifier(**kwargs)
C.train(data)
C.tune(data) # Tuning similarity within the default range of 0.01 - 1 in 0.01 increments, until optimal values are found

```

</details>

<details>
<summary>Tune selection</summary>
<br>

Add the argument __param\_name = 'selection'__ to the __Classifier.tune()__ method. To speed up the process, reduce the range with __param\_range = (start, stop, step)__. In this example, experience tells me the optimal value is somewhere in the range of 0.01 - 0.2 and so we don't have to spend time looking beyond that. You estimate one parameter at a time, hence the two separate calls to __tune()__. Only the first call needs to provide data to the Classifier. 

```python
C = Classifier(**kwargs)
C.train(data)
C.tune(data) # Tuning similarity within the default range of 0.01 - 1 in 0.01 increments
C.tune(param_name = 'selection', param_range = (0.01, 0.2, 0.01)) # Tuning selection within a reduced range, re-uses data from the previous call
```
</details>

<details>
<summary>Tune with plots</summary>
<br>

To plot the error curve, set __plot = True__. This requires external library [Plotext](https://github.com/piccolomo/plotext). Plots will be written to individual .html files in the _plots_ subfolder under your working directory. When plotting, the simulation will not end prematurely as it has to run through the entire range. You can still work in a reduced range with __param\_range__. 

```python
C.tune(data, plot = True) # Tuning similarity in the range of 0.01 - 1 in 0.01 increments and plotting the error curves
```
</details>


<details>
<summary>Use optimized criteria from tune()</summary>
<br>

Since __Classifier.tune()__ is a class method, the resulting "__criteria__" is saved to the class object. It's easy enough to export if you need to. 
```python
C = Classifier(**kwargs)
C.train(data)
C.tune(data) # Tuning similarity within the default range of 0.01 - 1 in 0.01 increments
C.tune(param_name = 'selection', param_range = (0.01, 0.2, 0.01)) # Tuning selection within a reduced range, re-uses data from the previous call
print(C.criteria)
```

This is the output. Again, only for the purpose of demonstration as it does not converge given the small sample size. 
```python
{'positive': {'similarity': 0.14, 'selection': 0.01}, 'service': {'similarity': 0.01, 'selection': 0.01}, 'suggestion': {'similarity': 0.37, 'selection': 0.04}, 'taste': {'similarity': 0.8, 'selection': 0.01}, 'spam': {'similarity': 1.0, 'selection': 0.01}}
```

</details>

<details>
<summary>Manually setting the thresholds</summary>
<br>

The concept of tuning the model relies on a validation dataset to compare the predictions with. If there is no validation data, you can optionally re-use the training data with the risk of overfitting. As a last resort however, you can also _predict_ the raw similarities, then sort the results descending with your favourite [spreadsheet software](https://github.com/saulpw/visidata), read the texts and manually identify the decision boundary. As you scroll through the results, there will be a dropoff point where texts are no longer relevant to the category of interest. You will need to consider both type 1 and type 2 errors (false positive and false negative respectively). Whether you want to minimize one of these or both will depend on your particular use case: "There are no solutions, only trade-offs". 

Exporting similarities:
```python
C = Classifier(**kwargs)
[*C.train(training_data)]
similarities = [*C.predict(prediction_data, discrete = False)] # Save similarities to variable
C.write_csv('my_similarities.csv', discrete = False) # Write similarities to disk
```

Using custom thresholds:
```python
custom_thresholds = { 'target1': { 'similarity': 0.64, 'selection': 0.1 }, 
                      'target2': { 'similarity': 0.7 }, # target2 uses the default selection
                      'target3': { 'selection': 0.2 } } # target3 uses the default cutoff (similarity)
kwargs = { 'targets': 'target1 target2 target3'.split(),
           'criteria': custom_thresholds }

C = Classifier(**kwargs)
print(C)
```

Output:
```bash
❯ python3 testrun4.py
Classifier(model='sentence-transformers/all-MiniLM-L6-v2', text_column='text', targets=['target1', 'target2', 'target3'], id_columns=[], confounders=[], ignore=[], default_selection=0.01, default_cutoff=0.5, criteria={'target1': {'similarity': 0.64, 'selection': 0.1}, 'target2': {'similarity': 0.7}, 'target3': {'selection': 0.2}}, mutually_exclusive=False, n_decimals=2, n_dims=384, trust_remote_code=False, show_progress_bar=True)
```


</details>

</details>


### Portability: save and load models

<details>
<summary>Expand</summary>

Saved models are typically a few kilobytes on disk. At the time of writing, only the training stage is saved. If you tune the model, the resulting __criteria__ will have to be stored elsewhere, eg in your python code. 

Saving:

```python
C = Classifier(**kwargs)
C.train(data)
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

Sentences are units of meaning, so splitting text into sentences will improve our predictions. If there is little punctuation in the text however, you may split by a fixed number of tokens and optionally with some overlap. These use cases are described below. The token splitting uses [regular expressions](https://docs.python.org/3/library/re.html) and word boundaries (\\b).

```python
from distinction import Classifier, split_records, combine_records

example_text = [{'text': 'This is the first sentence. Is this the second? Sentence number 3', 'binary_variable': 1},
                {'text': 'This text is a single sentence.', 'binary_variable': 0}] 
```
Note: in the code above we are hard-coding a binary variable from the start, where you would otherwise use the classifier for one or more targets once the text has been split. 

<details open>
<summary>Default settings</summary>
<br>

#### Split
By default, texts are split into sentences and then split into chunks when the sentence exceeds 384 tokens (the maximum for current sentence transformer models). The chunks are numbered by chunk\_id, with the last one being -1. These default settings should be used for semantic classifier pipelines, although a couple of other parameters are available to tamper with. 

__Note__: _split()_ assumes your data is not pre-encoded vectors. You cannot split an embedding the way you would split raw text. By the same token, _combine()_ automatically drops the vector column. 

```python
sentences = [*split_records(example_text)]
for sentence in sentences:
    print(sentence)
```

```bash
{'text': 'This is the first sentence. ', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 0}
{'text': 'Is this the second? ', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 1}
{'text': 'Sentence number 3', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 2}
{'text': 'This text is a single sentence.', 'binary_variable': 0, 'doc_id': 1, 'sentence_id': 0}
```

#### Combine

```python
results = [*combine_records(sentences, binary_targets = ['binary_variable'])]
for result in results:
    print(result)
```

Back to the original shape, with a document\_id added. 
```bash
{'doc_id': 0, 'text': 'This is the first sentence. Is this the second? Sentence number 3', 'binary_variable': 1}
{'doc_id': 1, 'text': 'This text is a single sentence.', 'binary_variable': 0}
```

</details>


<details>
<summary>Max sequence length</summary>
<br>

#### Split
If needed, you can set a different max number of tokens. Since whitespace and punctuation count as tokens, set max\_sequence\_length to double the number of words you want. This can also be used to produce n-grams. The two examples below differ with respect to the argument __per\_sentence__. 

##### Option 1: Fixed number of tokens
```python
sentences = [*split_records(example_text, per_sentence = False, max_sequence_length = 8)]
for sentence in sentences:
    print(sentence)
```

```bash
{'text': 'This is the first', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 0}
{'text': ' sentence. Is this the', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 1}
{'text': ' second? Sentence number 3', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 2}
{'text': '', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 3}
{'text': 'This text is a', 'binary_variable': 0, 'doc_id': 1, 'sentence_id': 0}
{'text': ' single sentence.', 'binary_variable': 0, 'doc_id': 1, 'sentence_id': 1}
```

##### Option 2: Max number of tokens per sentence
```python
sentences = [*split_records(example_text, per_sentence = True, max_sequence_length = 8)]
for sentence in sentences:
    print(sentence)
```

```bash
{'text': 'This is the first', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 0, 'chunk_id': 0}
{'text': ' sentence. ', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 0, 'chunk_id': -1}
{'text': 'Is this the second', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 1, 'chunk_id': 0}
{'text': '? ', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 1, 'chunk_id': -1}
{'text': 'Sentence number 3', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 2}
{'text': 'This text is a', 'binary_variable': 0, 'doc_id': 1, 'sentence_id': 0, 'chunk_id': 0}
{'text': ' single sentence.', 'binary_variable': 0, 'doc_id': 1, 'sentence_id': 0, 'chunk_id': -1}
```

#### Combine

There are no special considerations for combining records with respect to max\_sequence\_length. The code below uses default settings, as in the previous section. 
```python
results = [*combine_records(sentences, binary_targets = ['binary_variable'])]
for result in results:
    print(result)
```

```bash
{'doc_id': 0, 'text': 'This is the first sentence. Is this the second? Sentence number 3', 'binary_variable': 1}
{'doc_id': 1, 'text': 'This text is a single sentence.', 'binary_variable': 0}
```

</details>

<details>
<summary>Overlap</summary>
<br>

#### Split
An alternative strategy, possibly inferior to splitting by punctuation, is to split text by number of tokens with overlap. The example code below splits the text into chunks of no more than 10 tokens (5 words), with an overlap of 3 tokens (typically 2 words and one whitespace/punctuation). If using the overlap parameter, you must remember to use it when combining the texts back together again (see below). 
```python
sentences = [*split_records(example_text, per_sentence = False, max_sequence_length = 8, overlap = 3)]
for sentence in sentences:
    print(sentence)
```
```bash
{'text': 'This is the first', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 0}
{'text': 'the first sentence. Is ', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 1}
{'text': '. Is this the second', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 2}
{'text': 'the second? Sentence number ', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 3}
{'text': ' number 3', 'binary_variable': 1, 'doc_id': 0, 'sentence_id': 4}
{'text': 'This text is a', 'binary_variable': 0, 'doc_id': 1, 'sentence_id': 0}
{'text': 'is a single sentence.', 'binary_variable': 0, 'doc_id': 1, 'sentence_id': 1}
{'text': ' sentence.', 'binary_variable': 0, 'doc_id': 1, 'sentence_id': 2}
```

#### Combine

Note the _overlap_ argument below. 
```python
results = [*combine_records(example_text, overlap = 3, binary_targets = ['binary_variable'])]
for result in results:
    print(result)
```

```bash
{'doc_id': 0, 'text': 'This is the first sentence. Is this the second? Sentence number 3', 'binary_variable': 1}
{'doc_id': 1, 'text': 'This text is a single sentence.', 'binary_variable': 0}
```

</details>

</details>


### Aggregation options for binary targets

<details>
<summary>Expand</summary>
<br>

Depending on your use case, you may want to pick up the main tendency of the text or transient themes within it. For example, suggestions might be hidden in a sentence that is part of a longer text, whereas the general sentiment of a text might be better inferred by looking at the entire text. You can control this by how the binary variables are aggregated. 
Let's start with three examples relating to the restaurant reviews example above. The first text has 1 positive sentence, the second one has 2 and in the third one all 3 sentences are predicted to be positive. So which of these texts should we consider positive? 

```python

C = Classifier(**kwargs)
C.targets = ['positive'] # Let's look at one variable only, for brevity
C.train(data)

example_texts = [
    dict(id = 200, message = "Fries are expensive and not that good to be honest. Please do mashed potatoes instead and bigger plates! I am so incredibly happy."),
    dict(id = 201, message = "The location is great. I love the food. I don't like the staff that much."),
    dict(id = 202, message = "Food is nice. Staff is nice. I love it!")
]

split_examples = [*split_records(example_texts, text_column = 'message')]
predictions = [*C.predict(split_examples, discrete = True)]
for p in predictions:
    print(p)
```

```bash
{'doc_id': 0, 'id': 200, 'message': 'Fries are expensive and not that good to be honest. ', 'sentence_id': 0, 'positive': 0, 'spam': 0}
{'doc_id': 0, 'id': 200, 'message': 'Please do mashed potatoes instead and bigger plates! ', 'sentence_id': 1, 'positive': 0, 'spam': 0}
{'doc_id': 0, 'id': 200, 'message': 'I am so incredibly happy.', 'sentence_id': 2, 'positive': 1, 'spam': 0}
{'doc_id': 1, 'id': 201, 'message': 'The location is great. ', 'sentence_id': 0, 'positive': 1, 'spam': 0}
{'doc_id': 1, 'id': 201, 'message': 'I love the food. ', 'sentence_id': 1, 'positive': 1, 'spam': 0}
{'doc_id': 1, 'id': 201, 'message': "I don't like the staff that much.", 'sentence_id': 2, 'positive': 0, 'spam': 0}
{'doc_id': 2, 'id': 202, 'message': 'Food is nice. ', 'sentence_id': 0, 'positive': 1, 'spam': 0}
{'doc_id': 2, 'id': 202, 'message': 'Staff is nice. ', 'sentence_id': 1, 'positive': 1, 'spam': 0}
{'doc_id': 2, 'id': 202, 'message': 'I love it!', 'sentence_id': 2, 'positive': 1, 'spam': 0}
```

Note that __split\_records()__ changes the original data in place, in this case _example\_texts_ (printed again below). The __doc\_id__ key is added to each record, to be able to use this data later on when combining the split records back together. It's not strictly necessary to use the _original\_data_ argument of __combine\_records()__ below, it's just slightly better in terms of computational performance. 
```bash
{'id': 200, 'message': 'Fries are expensive and not that good to be honest. Please do mashed potatoes instead and bigger plates! I am so incredibly happy.', 'doc_id': 0}
{'id': 201, 'message': "The location is great. I love the food. I don't like the staff that much.", 'doc_id': 1}
{'id': 202, 'message': 'Food is nice. Staff is nice. I love it!', 'doc_id': 2}
```


##### Aggregation "any" (default)
```python
results = [*combine_records(predictions, 
                            text_column = 'message', 
                            binary_targets = C.targets, 
                            original_data = example_texts, 
                            aggregation = 'any')]
for result in results:
    print(result)
```

This is the default aggregation strategy, which sets the bar quite low. The first text should not be considered positive. We don't know why the person is happy, although we can assume it has nothing to do with the food. Consider the fact that sometimes people will express irony as well. 
```bash
{'doc_id': 0, 'message': 'Fries are expensive and not that good to be honest. Please do mashed potatoes instead and bigger plates! I am so incredibly happy.', 'positive': 1, 'spam': 0, 'id': 200}
{'doc_id': 1, 'message': "The location is great. I love the food. I don't like the staff that much.", 'positive': 1, 'spam': 0, 'id': 201}
{'doc_id': 2, 'message': 'Food is nice. Staff is nice. I love it!', 'positive': 1, 'spam': 0, 'id': 202}
```


##### Aggregation "most" / "majority"
```python
results = [*combine_records(predictions, text_column = 'message', 
                            binary_targets = C.targets, 
                            original_data = example_texts, 
                            aggregation = 'most')]
for result in results:
    print(result)
```

Looking at the majority of sentences, texts 2 and 3 become positive. 
```bash
{'doc_id': 0, 'message': 'Fries are expensive and not that good to be honest. Please do mashed potatoes instead and bigger plates! I am so incredibly happy.', 'positive': 0, 'spam': 0, 'id': 200}
{'doc_id': 1, 'message': "The location is great. I love the food. I don't like the staff that much.", 'positive': 1, 'spam': 0, 'id': 201}
{'doc_id': 2, 'message': 'Food is nice. Staff is nice. I love it!', 'positive': 1, 'spam': 0, 'id': 202}
```

##### Aggregation "all"
```python
results = [*combine_records(predictions, 
                            text_column = 'message', 
                            binary_targets = C.targets, 
                            original_data = example_texts, 
                            aggregation = 'all')]
for result in results:
    print(result)
```
With the constraint that all sentences have to be 1, only the third text becomes positive. 
```bash
{'doc_id': 0, 'message': 'Fries are expensive and not that good to be honest. Please do mashed potatoes instead and bigger plates! I am so incredibly happy.', 'positive': 0, 'spam': 0, 'id': 200}
{'doc_id': 1, 'message': "The location is great. I love the food. I don't like the staff that much.", 'positive': 0, 'spam': 0, 'id': 201}
{'doc_id': 2, 'message': 'Food is nice. Staff is nice. I love it!', 'positive': 1, 'spam': 0, 'id': 202}
```


##### Aggregation "relative" / "share"
```python
results = [*combine_records(predictions, 
                            text_column = 'message', 
                            binary_targets = C.targets, 
                            original_data = example_texts, 
                            aggregation = 'relative')]
for result in results:
    print(result)

```

After concatenation, the texts are found to be 1/3, 2/3 and 100% positive. 
```bash
{'doc_id': 0, 'message': 'Fries are expensive and not that good to be honest. Please do mashed potatoes instead and bigger plates! I am so incredibly happy.', 'positive': 0.3333333333333333, 'spam': 0.0, 'id': 200}
{'doc_id': 1, 'message': "The location is great. I love the food. I don't like the staff that much.", 'positive': 0.6666666666666666, 'spam': 0.0, 'id': 201}
{'doc_id': 2, 'message': 'Food is nice. Staff is nice. I love it!', 'positive': 1.0, 'spam': 0.0, 'id': 202}
```

##### Aggregation "mutually\_exclusive"

Use this when working with mutually exclusive data. The most common prediction wins. When two or more targets are equally common, the score becomes a tiebreaker. 
The actual code is shown as part of the section on [classifying mutually exclusive data](#set-up-and-use-a-Classifier-for-mutually-exclusive-binary-variables) below. 


</details>


### Set up a prediction pipeline for continuous data streams

<details>
<summary>Expand</summary>

When your model is finalized, you can turn it into an end-to-end prediction function using the _Classifier.to_pipeline()_ method. Provide it with the same keyword arguments used to initiate the Classifier before. The pipeline provides a convenient way to split the text, predict at the sentence level, and then combine the records together again. This code continues from the restaurant reviews example above. 

There are three use cases to consider, with respect to the __pre\_encoded__ argument. 

1. A raw text column is used for prediction. The text is split by sentence and then glued back together again. The original document level data is available to join in when combining the records after prediction. 
2. Pre-encoded vectors are used for prediction. These cannot be split into sentences or chunks like the raw text. It's assumed that these vectors are encoded at the sentence level, meaning they will be combined into coherent texts (using a _doc\_id\_column_ as key) after prediction. If not, set _combine = False_. Vectors are dropped from the output.
3. As #2, except there is also raw text to be combined. Remember the _text\_column_ is dual use: it can point to raw text or a pre-encoded vector. For this edge case, use _pre\_encoded = True_ and the extra argument _raw\_text\_column_. The vector, specified in _text\_column_ will be used for prediction. The raw text column will be glued together, assuming there are id columns in the data. 

<details>
<summary>Case 1: raw text as input</summary>

```python
kwargs = {
    'targets': 'positive suggestion taste service'.split(),
    'confounders': ['spam'],
    'id_columns': ['id'],
    'text_column': 'message',
    'default_selection': 0.05,
    'model': 'sentence-transformers/all-MiniLM-L6-v2',
    'show_progress_bar': False
}

C = Classifier(**kwargs)
C.train(data)
pipeline = C.to_pipeline(**kwargs) # Use the same settings as for the original classifier

new_data = [{"message": "I really love the taste of these burgers. This place is great."},
            {"message": "The staff was really helpful"},
            {"message": "This is definitely spam, even though I mention that the burgers taste good and the staff are helpful."}]

results = pipeline(new_data) # Returns a generator by default
for result in results:
    print(result)
```

```bash
{'doc_id': 0, 'message': 'I really love the taste of these burgers. This place is great.', 'positive': 1, 'service': 0, 'suggestion': 0, 'taste': 1, 'spam': 0}
{'doc_id': 1, 'message': 'The staff was really helpful', 'positive': 1, 'service': 1, 'suggestion': 0, 'taste': 0, 'spam': 0}
{'doc_id': 2, 'message': 'This is definitely spam, even though I mention that the burgers taste good and the staff are helpful.', 'positive': 0, 'service': 0, 'suggestion': 0, 'taste': 0, 'spam': 1}
```
<details>


<details>
<summary>Case 2: embeddings as input</summary>

Below we use embeddings for both training and prediction. 

```python
kwargs = {
    'targets': 'positive suggestion taste service'.split(),
    'confounders': ['spam'],
    'id_columns': ['id'],
    'text_column': 'embedding',
    'default_selection': 0.05
}

model = SentenceTransformer(model_name_or_path = 'sentence-transformers/all-MiniLM-L6-v2',
                            tokenizer_kwargs = {'clean_up_tokenization_spaces': True})

vectors = model.encode([r['message'] or '' for r in data], show_progress_bar = False)

for i,_ in enumerate(data):
    data[i]['embedding'] = vectors[i] # Put embedding into data
    data[i].pop('message') # Remove original text

C = Classifier(**kwargs)
C.train(data, pre_encoded = True)


kwargs.update(dict(pre_encoded = True, doc_id_column = 'id', split = False, combine = True)) 
# pre_encoded is an argument for .to_pipeline() but not to Classifier
pipeline = C.to_pipeline(**kwargs)


new_data = [{"id": 101, "message": "I really love the taste of these burgers. This place is great."},
            {"id": 102, "message": "The staff was really helpful"},
            {"id": 103, "message": "This is definitely spam, even though I mention that the burgers taste good and the staff are helpful."}]

new_vectors = model.encode([r['message'] or '' for r in new_data], show_progress_bar = False)
for i,_ in enumerate(new_data):
    new_data[i]['embedding'] = new_vectors[i]
    new_data[i].pop('message')

predictions = pipeline(new_data) # Returns a generator by default
for p in predictions:
    print(p)
```

```bash
{'id': 101, 'positive': 0, 'service': 0, 'suggestion': 0, 'taste': 1, 'spam': 0}
{'id': 102, 'positive': 1, 'service': 1, 'suggestion': 0, 'taste': 0, 'spam': 0}
{'id': 103, 'positive': 0, 'service': 0, 'suggestion': 0, 'taste': 0, 'spam': 1}
```

<details>


<details>
<summary>Case 3: embeddings as input, raw text as output</summary>

This is very similar to the previous example, except we specify a _text\_column_ for the raw text and put the vector in the _vector\_column_ argument. The vector will be dropped (unless _keep\_vector = True_) and the actual text will be concatenated. 

```python
kwargs = {
    'targets': 'positive suggestion taste service'.split(),
    'confounders': ['spam'],
    'id_columns': ['id'],
    'text_column': 'message',
    'vector_column': 'embedding',
    'default_selection': 0.05
}

model = SentenceTransformer(model_name_or_path = 'sentence-transformers/all-MiniLM-L6-v2',
                            tokenizer_kwargs = {'clean_up_tokenization_spaces': True})

vectors = model.encode([r['message'] or '' for r in data], show_progress_bar = False)

for i,_ in enumerate(data):
    data[i]['embedding'] = vectors[i]

C = Classifier(**kwargs)
C.train(data, pre_encoded = True)


kwargs.update(dict(pre_encoded = True, doc_id_column = 'id', split = False, combine = True)) # pre_encoded is an argument for .to_pipeline() but not to Classifier
pipeline = C.to_pipeline(**kwargs)


new_data = [{"id": 101, "message": "I really love the taste of these burgers. This place is great."},
            {"id": 102, "message": "The staff was really helpful"},
            {"id": 103, "message": "This is definitely spam, even though I mention that the burgers taste good and the staff are helpful."}]

new_vectors = model.encode([r['message'] or '' for r in new_data], show_progress_bar = False)
for i,_ in enumerate(new_data):
    new_data[i]['embedding'] = new_vectors[i]

predictions = pipeline(new_data) # Returns a generator by default
for p in predictions:
    print(p)
```

```bash
{'id': 101, 'message': 'I really love the taste of these burgers. This place is great.', 'positive': 0, 'service': 0, 'suggestion': 0, 'taste': 1, 'spam': 0}
{'id': 102, 'message': 'The staff was really helpful', 'positive': 1, 'service': 1, 'suggestion': 0, 'taste': 0, 'spam': 0}
{'id': 103, 'message': 'This is definitely spam, even though I mention that the burgers taste good and the staff are helpful.', 'positive': 0, 'service': 0, 'suggestion': 0, 'taste': 0, 'spam': 1}
```


<details>


<details>
<summary>Chaining pipelines</summary>

You can use one pipeline after another. Preferrably use the argument _keep\_vector = True_ to avoid the time-consuming step of encoding the text between the pipelines. Only the last pipeline should have _keep\_vector = False_ and _combine = True_ if prediction was done at the sentence level. If both of these arguments are True an exception is raised, as embeddings cannot be concatenated in a meaningful way. 

The example below consists of two separate pipelines. The first one categorizes a document as suggestion and/or taste if __any__ of the constituent sentences are predicted as such. Suggestions might be part of a longer text and not necessarily the main message. The second pipeline looks at the main tendency of the text, where __most__ of the sentences have to be positive or about service for these categories to be applied to the entire document. As the second text consists of one sentence, predictions are the same at the sentence and document levels.

The call to _pipeline\_chain([pipeline1, pipeline2], new\_data)_ is equal to _predictions = pipeline2(pipeline1(new\_data))_ except it's obviously more scalable. 

```python

from distinction import Classifier, count_used_keys, ones_to_int, pipeline_chain

# Omitted code from previous examples

kwargs = {
    'targets': 'positive suggestion taste service'.split(),
    'confounders': ['spam'],
    'id_columns': ['id'],
    'text_column': 'message',
    'vector_column': 'embedding',
    'default_selection': 0.05
}

C = Classifier(**kwargs)
C.train(data, pre_encoded = False)

kwargs.update(dict(keep_vector = True, targets = "suggestion taste".split(), aggregation = 'any', doc_id_column = 'id', split = True, combine = False)) 
# keep_vector makes sure we only need to encode the raw text once (time consuming)
# Split the text into sentences, but do not combine before applying the second pipeline
pipeline1 = C.to_pipeline(**kwargs)

kwargs.update(dict(pre_encoded = True, keep_vector = False, targets = "positive service".split(), aggregation = 'most', doc_id_column = 'id', split = False, combine = True))
# Combine sentences into the original documents, since this is the last prediction step
pipeline2 = C.to_pipeline(**kwargs)

new_data = [{"id": 101, "message": "I really love the taste of these burgers. This place is great."},
            {"id": 102, "message": "The staff was really helpful"},
            {"id": 103, "message": "This is definitely spam, even though I mention that the burgers taste good and the staff are helpful."}]

predictions = pipeline_chain([pipeline1, pipeline2], new_data)

for p in predictions:
    print(p)

```

```bash
{'id': 0, 'message': 'I really love the taste of these burgers. This place is great.', 'positive': 1, 'service': 0, 'suggestion': 0, 'taste': 1, 'spam': 0}
{'id': 1, 'message': 'The staff was really helpful', 'positive': 1, 'service': 1, 'suggestion': 0, 'taste': 0, 'spam': 0}
{'id': 2, 'message': 'This is definitely spam, even though I mention that the burgers taste good and the staff are helpful.', 'positive': 0, 'service': 0, 'suggestion': 0, 'taste': 0, 'spam': 1}
```



<details>



</details>


### Set up and use a Classifier for mutually exclusive binary variables

<details>
<summary>Expand</summary>

The following two examples use external datasets from [Kaggle](https://www.kaggle.com/datasets/). The Classifier is used in the same way as in the previous sections, except we add the argument __mutually\_exclusive = True__. This means only one of the targets can be true (1) and all others are false (0). 

There are no similarity thresholds for this kind of model, as the category with the max similarity is chosen (irrespective of whether these categories are equally well defined). As of this writing, there is also no tuning of "selection", ie how large percentage of the features to use. You will have to experiment. A reasonable assumption is that you will need a larger selection the more targets there are to choose from and the more general these decisions are. For the two analyses below, I settled on 0.5 and 0.65 respectively after some trial-and-error, which is substantially higher than what seems to work best for independent variables (as above). For some analyses, you might even set _default\_selection = 1_ meaning all the features are used, if that turns out to yield the lowest error. 

#### Reviews data

This example uses a subset of the [Reviews](https://www.kaggle.com/datasets/ahmedabdulhamid/reviews-dataset) dataset. Although the accuracy is not that good on individual sentences, after aggregating the result we get a typical accuracy of just below 90 %. This is despite the fact that the model is not trained on individual sentences.  

<details>
<summary>Code</summary>

```python
import sys
import csv
import random

from distinction import Classifier, split_records, combine_records

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

The confusion matrix below shows you the rate of misclassification (outside of the diagonal) on individual sentences. Then the overall accuracy is calculated on the original texts after putting them back together. The overall performance is obviously better than the per-sentence prediction, which highlights the value of seeing the bigger picture. 

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
sys		0m3,801s
```

</details>


#### News data

This next example uses a subset of the [AG's News Corpus](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) dataset. The typical accuracy is about 84 %. 
Each news article can belong to only one category. There is however some semantic overlap, such as sports metaphors being used for other kinds of news. Consider this title of a _world_ news article: "Rights group slams Iraqi trials". It was predicted to be a _sports_ article with a high degree of certainty. Slamming people is generally associated with contact sports like American football or hockey. The sentence transformer model lacks important contextual understanding of who the actors are in this sentence. 

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
print('Number of split sentences #2: ', len(split_prediction_data)) # After filtering texts with > 10 characters
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

print(f'First {n} predictions (entire text):')
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

In the very first sample, Wayne Rooney's arrival at Manchester United was originally and somewhat surprisingly in the _world_ category, although predicted as _sports_ with a score of 0.33. 
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

First 5 predictions (entire text):
{'doc_id': 0, 'text': "Wayne Rooney arrives at Man Utd's training ground for a medical ahead of a 25m move from Everton.", 'predicted': 'sports', 'score': [0.33], 'actual': 'world'}

{'doc_id': 1, 'text': 'There are many other treatment options for people with arthritis, and physicians are considering them patient by patient. Washington -- Physicians are pulling out their risk-versus-benefit calculators once ', 'predicted': 'business', 'score': [0.13, 0.35], 'actual': 'business'}

{'doc_id': 2, 'text': 'com - A new research station at the bottom of the world may give future Antarctica researchers some special treats, like the ability to live above ground and look out a window.', 'predicted': 'science', 'score': [0.33], 'actual': 'science'}

{'doc_id': 3, 'text': 'Namibian President Sam Nujoma #39;s chosen successor, Hifikepunye Pohamba, has won a landslide victory with 75 of the vote in the country #39;s third elections since independence, according to official results.', 'predicted': 'world', 'score': [0.37], 'actual': 'world'}

{'doc_id': 4, 'text': 'The planned acquisition of Sears, Roebuck and Co. by Kmart Holding Corp. highlights a changing retail environment that could soon eliminate the department store as we know it, analysts and consultants said on Friday.', 'predicted': 'business', 'score': [0.48, 0.51, 0.38], 'actual': 'business'}

Overall accuracy:
0.836

real	0m6,752s
user	0m6,727s
sys		0m3,597s

```


</details>

</details>
