# Installation

## From Github
```
pip3 install git+https://github.com/er1kb/distinction
```
or clone and install locally:
```
git clone https://github.com/er1kb/distinction.git && cd distinction && pip3 install .
```

## From PyPI
```
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

## Examples

Input data is an iterable (list/generator) of dicts. Export from your favourite dataset library using polars.DataFrame.[to\_dicts()](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.to_dicts.html) or pandas.DataFrame.[to\_dict('records')](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_dict.html).

<details>
<summary>Split records</summary>
<br>
TODO: example
```
pass
```
</details>

<details>
<summary>Combine records</summary>
<br>
TODO: example
```
pass
```
</details>

<details>
<summary>Classifier from training\_data - raw text</summary>
<br>
TODO: example
```
pass
```
</details>

<details>
<summary>Classifier from training\_data - pre-encoded</summary>
<br>
TODO: example
```
pass
```
</details>

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








# Swedish

## TODO

```
import distinction as ds
C = ds.Classifier(**kwargs)
[*C.train(training_data = ...)]
predictions = [*C.predict(...)]
```


