# Dependencies
Tested with:  
* [Numpy](https://www.python.org/) >= 1.25.0
* [SentenceTransformers](https://sbert.net/) >= 3.0.1

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


# English

## What is it
## Examples
### Split records
### Combine records
### Classifier from training\_data - raw text
### Classifier from training\_data - pre-encoded
### Tune similarity
### Tune selection
### Tune with plots
### Use optimized criteria from tune()
### Prediction pipeline


# Swedish

## TODO

```
import distinction as ds
C = ds.Classifier(**kwargs)
[*C.train(training_data = ...)]
predictions = [*C.predict(...)]
```


