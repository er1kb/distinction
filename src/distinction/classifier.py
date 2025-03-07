import os
import re
import sys
import math
import time
import warnings
from functools import partial, reduce
from dataclasses import dataclass, field
from itertools import zip_longest, cycle, combinations

import numpy as np
from sentence_transformers import SentenceTransformer


def split_records(records, text_column = 'text', max_sequence_length = 384, doc_id_column = 'doc_id', row_id_column = 'sentence_id', chunk_id_column = 'chunk_id', overlap = 0, per_sentence = True):
    """Splits dicts containing text into chunks of at most max_sequence_length tokens. Useful for constraints like the 512 word limit of BERT models."""
    k = 0
    if overlap >= max_sequence_length:
        raise Exception(f'Overlap ({overlap}) needs to be smaller than max_sequence_length ({max_sequence_length}), in this case {max_sequence_length-1} or less.')
    for record in records:
        text = record.get(text_column)
        record.update({doc_id_column: k}) # Add doc_id to original data
        i = max_sequence_length - int(overlap)
        j = 0
        if per_sentence:
            pattern = re.compile(r'[.?!]+\s*')
            first_hit = pattern.search(text)
            s = 0
            while bool(first_hit):
                end = first_hit.span()[1]
                tokens = re.split(pattern = r'\b', string = text[:end])

                # If max_sequence_length defined, split long sentences into chunks
                if max_sequence_length and len(tokens) > max_sequence_length:
                    c = 0
                    while len(tokens) > max_sequence_length:
                        yield { **record, **{ text_column: ''.join(tokens[:max_sequence_length]), 
                                doc_id_column: k, row_id_column: s, chunk_id_column: c } }
                        c += 1
                        tokens = tokens[i:]
                    yield { **record, **{ text_column: ''.join(tokens),
                            doc_id_column: k, row_id_column: s, chunk_id_column: -1 } } # mark the last chunk of the sentence by setting chunk_id = -1
                else:
                    yield { **record, **{ text_column: text[:end], 
                            doc_id_column: k, row_id_column: s } }
                text = text[end:]
                first_hit = pattern.search(text)
                s += 1
            # If remaining text, return it
            if len(text) > 0:
                yield { **record, **{ text_column: text, 
                        doc_id_column: k, row_id_column: s } }

        else:
            tokens = re.split(pattern = r'\b', string = text)
            while (len(tokens) // max_sequence_length) > 0:
                yield { **record, **{ text_column: ''.join(tokens[:max_sequence_length]), 
                                      doc_id_column: k, row_id_column: j } }
                j += 1
                tokens = tokens[i:]
            yield { **record, **{ text_column: ''.join(tokens[:max_sequence_length]), 
                    doc_id_column: k, row_id_column: j } } # TODO: Mark the last chunk here as well?
        k += 1


def combine_two_records(binary_targets, text_separator, text_column, doc_id_column, row_id_column, chunk_id_column, overlap, original_data, mutually_exclusive, drop, a = None, b = None):
    """ Merge dicts belonging to the same document, as indicated by the doc_id_column """
    from collections.abc import Sequence

    drop = set(drop)

    doc_ids = set([a[doc_id_column], b[doc_id_column]])
    if len(doc_ids) > 1:
        raise Exception("Cannot merge two documents with different doc ids.")

    common_keys = { text_column, doc_id_column }
    # Where available, use values from the original fulltext record
    if original_data:
        original_keys = set(original_data[a[doc_id_column]].keys()).union(set(original_data[a[doc_id_column]].keys())) - common_keys - set(binary_targets) - drop
    else:
        original_keys = set()

    remaining_keys = set(a.keys()).union(set(b.keys())) - common_keys - set(binary_targets) - original_keys - drop

    if row_id_column:
        remaining_keys = remaining_keys - { row_id_column }
    if chunk_id_column:
        remaining_keys = remaining_keys - { chunk_id_column }

    if text_column:
        # If the second text is an empty string, ignore overlap as it might be the last text of the document
        # if len(b[text_column]) == 0:
        if len(b.get(text_column)) == 0:
            overlap = 0

        if (overlap == 0):
            # a_text = a[text_column]
            a_text = a.get(text_column)
        else:
            # tokens = re.split(pattern = r'\b', string = a[text_column])
            tokens = re.split(pattern = r'\b', string = a.get(text_column))
            if (chunk_id_column in b and b[chunk_id_column] < 0):
                a_text = ''.join(tokens[:-overlap - 1])
            else:
                a_text =  ''.join(tokens)
        # record = { doc_id_column: a[doc_id_column],
        record = { doc_id_column: a.get(doc_id_column),
                   # text_column: text_separator.join([a_text, b[text_column]]) }
                   text_column: text_separator.join([a_text, b.get(text_column)]) }
    else:
        record = { doc_id_column: a.get(doc_id_column) }

    if mutually_exclusive:
        k = binary_targets[0]
        s = binary_targets[1]
        if not k in a and not k in b:
            raise Exception(f'Target "{k}" not found in dicts: \n\n{a} \n\n{b}')
        elif k in a and not k in b:
            record[k] = a[k] if isinstance(a[k], list) else [a[k]] # a[k] needs to be a list
            record[s] = a[s] if isinstance(a[s], list) else [a[s]] # a[k] needs to be a list
        elif not k in a and k in b: 
            record[k] = b[k] if isinstance(b[k], list) else [b[k]]
            record[s] = b[s] if isinstance(b[s], list) else [b[s]]
        else:
            record[k] = a[k] if isinstance(a[k], list) else [a[k]]
            record[k].append(b[k])
            record[s] = a[s] if isinstance(a[s], list) else [a[s]]
            record[s].append(b[s])
    else:
        for target in binary_targets:
            if not target in a and not target in b:
                raise Exception(f'Target "{target}" not found in dicts: \n\n{a} \n\n{b}')
            elif target in a and not target in b:
                record[target] = a[target]
            elif not target in a and target in b: 
                record[target] = b[target]
            else:
                record[target] = a[target] + b[target]

    for key in original_keys:
        record[key] = a.get(key) or b.get(key)

    for key in remaining_keys: # This part is used when original data is not supplied
        # If key present in just one of the records, grab that one
        if key in a and not key in b:
            record[key] = a[key]
        elif not key in a and key in b:
            record[key] = b[key]
        elif isinstance(a[key], str):
            if not isinstance(b[key], str):
                raise Exception(f'Cannot merge {type(a[key])} with {type(b[key])}.')
            record[key] = text_separator.join([a[key], b[key]])
        elif isinstance(a[key], Sequence):
            if isinstance(b[key], Sequence):
                # If a and b items are equal, keep a
                all_equal = all([x == y for x,y in zip_longest(a[key], b[key])])
                record[key] = a[key] if all_equal else a[key] + b[key]
            else:
                # If item b is not a sequence, append to a
                record[key] = list(a[key])
                record[key].append(b[key])
        else:
            # If single values at the first iteration, put both values in a list
            if a[key] == b[key]:
                record[key] = a[key]
            else:
                record[key] = [a[key], b[key]]

    return record


def combine_records(records, text_separator = '', text_column = None, doc_id_column = 'doc_id', row_id_column = 'sentence_id', chunk_id_column = 'chunk_id', original_data = None, binary_targets = None, vector_column = None, aggregation = 'any', overlap = 0, drop = list()):
    """ Combines dicts and concatenates text """

    records = iter(records)
    it = records.__iter__()
    previous_record = next(it)
    first_record = previous_record
    doc_id = previous_record[doc_id_column]
    current_doc = list()

    # strategies = 'any absolute sum most majority relative share mutually_exclusive'.split()

    if original_data:
        len1 = len(original_data)
        original_data = { (r[doc_id_column] if doc_id_column in r else i): r for i,r in enumerate(original_data) }
        len2 = len(original_data.keys())
        if len2 < len1:
            raise Exception(f'Doc ids missing or duplicate in at least one record of the original data. {len1} records contain {len2} unique keys.')

    mutually_exclusive = (aggregation == 'mutually_exclusive')
    if mutually_exclusive:
        k = 'predicted'
        s = 'score'
        binary_targets = [k, s]
        # if k in first_record:
        #     if not isinstance(first_record[k], list):
        #         first_record[k] = [first_record[k]] # Initial value needs to be a list
    else:
        for v in drop:
            if v in binary_targets:
                binary_targets.remove(v)

    if vector_column:
        drop.append(vector_column) # If text is pre-encoded vector, drop it

    glue = partial(combine_two_records, binary_targets, text_separator, text_column, doc_id_column, row_id_column, chunk_id_column, overlap, original_data, mutually_exclusive, drop)

    # Inspired by https://excamera.com/sphinx/article-islast.html - thanks!
    while True:
        try: 
            record = next(it)

            # If current record belongs to the same document as the previous record
            if record[doc_id_column] == doc_id:
                current_doc.append(record)
            else:
                # Make sure the reduce function gets used, even if the document is length 1
                if len(current_doc) == 0:
                    if text_column:
                        current_doc = [{ text_column: '', doc_id_column: doc_id, row_id_column: 1 }]
                    else:
                        current_doc = [{ doc_id_column: doc_id, row_id_column: 1 }]

                # Concat in order of row_id, not necessarily the order of the data (it might have been scrambled for some reason)
                if row_id_column:
                    current_doc.sort(key = lambda r: r[row_id_column])


                fulltext = reduce(glue, current_doc, first_record)
                n_rows = 1 + len(current_doc)

                # Aggregate targets
                if binary_targets and aggregation:
                    if mutually_exclusive:
                        target = binary_targets[0] # Single target: the predicted label column
                        score = binary_targets[1]
                        counter = list()
                        for value in set(fulltext[target]):
                            scores = fulltext.get(score)
                            scores = [ scores[i] for i,prediction in enumerate(fulltext[target]) if prediction == value ]
                            counter.append((value, fulltext[target].count(value), sum(scores) / len(scores)))
                        counter.sort(key = lambda x: (x[1], x[2]), reverse = True)
                        fulltext[target] = counter[0][0]

                    else:
                        for target in binary_targets: 
                            if not isinstance(fulltext[target], (int, np.integer)):
                                if text_column:
                                    obj = { text_column: fulltext[text_column], target: fulltext[target] }
                                else:
                                    obj = { target: fulltext[target] }
                                print()
                                print('problematic object:')
                                print(obj)
                                print()
                                raise Exception(f'Target variable is not integer (Python int or numpy.integer) but of type {type(fulltext[target])}: \n {obj}')
                            if not aggregation or aggregation in 'absolute sum mutually_exclusive'.split():
                                pass # keep the summed number or list of predicted labels
                            elif aggregation == 'any':
                                fulltext[target] = 1 if fulltext[target] > 0 else 0
                            elif aggregation == 'all':
                                fulltext[target] = 1 if (fulltext[target] / n_rows) == 1 else 0
                            elif aggregation in 'most majority'.split():
                                fulltext[target] = 1 if (fulltext[target] / n_rows) >= 0.5 else 0
                            elif aggregation in 'relative share'.split():
                                fulltext[target] = fulltext[target] / n_rows
                            else:
                                raise Exception(f'Unknown aggregation_strategy: {aggregation}')


                yield fulltext

                current_doc.clear()
                first_record = record
                doc_id = record[doc_id_column]

            previous_record = record

        except StopIteration: 

            if len(current_doc) == 0:
                if text_column:
                    current_doc = [{ text_column: '', doc_id_column: doc_id, row_id_column: 1 }]
                else:
                    current_doc = [{ doc_id_column: doc_id, row_id_column: 1 }]
            # Repeat the above for the very last record
            if row_id_column:
                current_doc.sort(key = lambda r: r[row_id_column])
            fulltext = reduce(glue, current_doc, first_record)
            n_rows = 1 + len(current_doc)

            # Aggregate targets
            if binary_targets and aggregation:
                if mutually_exclusive:
                    target = binary_targets[0] # Single target: the predicted label column
                    score = binary_targets[1]
                    counter = list()
                    for value in set(fulltext[target]):
                        scores = fulltext.get(score)
                        scores = [ scores[i] for i,prediction in enumerate(fulltext[target]) if prediction == value ]
                        counter.append((value, fulltext[target].count(value), sum(scores) / len(scores)))
                    counter.sort(key = lambda x: (x[1], x[2]), reverse = True)
                    fulltext[target] = counter[0][0]

                else:
                    for target in binary_targets: 
                        if not aggregation or aggregation in 'absolute sum mutually_exclusive'.split():
                            pass # keep the summed number or list of predicted labels
                        elif aggregation == 'any':
                            fulltext[target] = 1 if fulltext[target] > 0 else 0
                        elif aggregation == 'all':
                            fulltext[target] = 1 if (fulltext[target] / n_rows) == 1 else 0
                        elif aggregation in 'most majority'.split():
                            fulltext[target] = 1 if (fulltext[target] / n_rows) >= 0.5 else 0
                        elif aggregation in 'relative share'.split():
                            fulltext[target] = fulltext[target] / n_rows
                        else:
                            raise Exception(f'Unknown aggregation_strategy: {aggregation}')


            yield fulltext
            break


# Helpers

def pipeline_chain(pipelines, input):
    return reduce(lambda a,f: f(a), pipelines, input)


def get_used_keys(records):
    """ Get list of keys in iterable of row-based dicts """
    return set().union(*(d.keys() for d in records))


def tally_ones(a,b):
    """ Count ones """
    for k in b.keys():
        a[k] += 1 if (b.get(k) and b.get(k) == 1) else 0
    return a


def count_used_keys(records, ignore = ''):
    """ Count ones, per target, ignoring named columns (separated by whitespace) """
    ignore = ignore.split()
    records = [*records]
    initial_counts = { k:0 for k in get_used_keys(records) }
    counts = reduce(tally_ones, records, initial_counts)
    for key in ignore:
        counts.pop(key, None)
    return counts


def filter_keys(records, keys):
    """ Keep only certain keys in iterable of row-based dicts """
    for r in records: 
        yield { key:r[key] for key in keys if key in r }


def ones_to_int(records, keys):
    """ Make sure ones are of type int, if they happen to be valid integer values as string """
    for record in records: 
        for key in keys:
            if key in record:
                try:
                    record[key] = int(record[key])
                except:
                    pass
        yield record


def remove_unused_keys(records, exceptions):
    """ Remove unused keys in iterable of row-based dicts """
    for r in records:
        yield { k:v for k,v in r.items() if k in exceptions or (v and not v == 0) }


def dict_to_records(d):
    """ Convert column-based dict to iterable of row-based dicts """
    for col in zip(*d.values()):
        yield dict(zip(d, col))


def records_to_dict(records, keys):
    """ Convert iterable of row-based dicts to a column-based dict """
    d = dict((key, None) for key in keys)
    for key in keys:
        d[key] = [None] * len(records)
        for i in range(len(records)):
            d[key][i] = records[i][key] if key in records[i] and records[i][key] else 0
    return d


def dict_to_matrix(d, keys):
    """Convert dict of lists to a numpy 2d array"""

    matrix = np.zeros((len(d[keys[0]]), len(keys)))
    i = 0
    for key in keys:
        matrix[:,i] = d[key]
        i += 1
    return matrix


def matrix_to_dict(matrix, keys):
    """Convert numpy 2d array to a dict of lists"""
    d = dict()
    i = 0
    for key in keys:
        d[key] = matrix[:,i]
        i += 1
    return d


def confusions(validation_matrix, similarity_matrix, keys, nrows, n_decimals):
    """ Produce a confusion matrix with the actual and predicted values """
    confusion_matrix = np.zeros((len(keys), len(keys)))
    for i,key1 in enumerate(keys):
        for j,key2 in enumerate(keys):
            confusion_matrix[i,j] = np.sum(validation_matrix[:,i] * similarity_matrix[:,j])
    actual_shares = np.round(np.sum(validation_matrix, axis=0) / nrows, n_decimals)
    predicted_shares = np.round(np.sum(similarity_matrix, axis=0) / nrows, n_decimals)
    return (keys, np.round(confusion_matrix / nrows, n_decimals), actual_shares, predicted_shares)


def clamp(n, min, max):
    """ Clamp a number to a range. """
    if n < min:
        return min
    elif n > max:
        return max
    else:
        return n


def if_undefined(arg, default):
    if arg is None:
        return default
    else:
        return arg


def contains_minimum(a,b,c):
    return ((b - a) <= 0) and ((c - b) >= 0)


def subdivide(a, b, center = False):
    if center:
        margin = (b - a) / 4
        mid = (a + b) / 2
        return (mid - margin, mid, mid + margin)
    else:
        return (a, (a + b) / 2, b)



@dataclass
class Classifier:
    training_data: list[dict] = field(default_factory = list, repr = False)
    # model: str = 'KBLab/sentence-bert-swedish-cased'
    model: str = 'sentence-transformers/all-MiniLM-L6-v2' 
    text_column: str = 'text'
    vector_column: str = None
    targets: list[str] = field(default_factory = list)
    id_columns: list[str] = field(default_factory = list)
    confounders: list[str] = field(default_factory = list)
    ignore: list[str] = field(default_factory = list)
    default_selection: float = 0.01
    default_cutoff: float = 0.5
    criteria: dict[dict] = field(default_factory = dict)
    mutually_exclusive: bool = False
    n_decimals: int = 2
    n_dims: int = None
    trust_remote_code: bool = False
    show_progress_bar: bool = True
    use_sample_probability = False

    """
    Parameters
    ----------
    training_data :list(dict)
        List of dicts containing a text column (raw text or pre-encoded vector) and one or more binary columns

    model : Optional[str]
        The sentence transformer model to be used (copy the name from Huggingface hub)

    text_column : str
        Name of the text column

    vector_column : str
        Optional name of the vector column. You can use either the text_column or vector_column to specify an embedding column, assuming you train and/or predict using pre_encoded = True. This argument exists because you may want a pipeline to predict using a vector and then concatenate the raw text. 

    targets : Optional[str]
        Names of the binary columns. Makes sense if there aren't many of these columns in your data. If there are more columns than you care to type and keep track of, specify the id_columns, confounders and ignore (explained below) and the targets will be inferred by exclusion.

    id_columns : Optional[list(str)]
        Names of id columns, in order to exclude these

    confounders : Optional[list(str)]
        Confounding variables. These are part of the target variables. Any row that is a confounder cannot belong to any other category that is not also a confounder, meaning all proper target variables are set to 0. 

    ignore : Optional[list(str)]
        Names of columns to be ignored.

    default_selection: Optional[float]
        The share of dimensions to use, if not otherwise specified. The default value of 0.01 means you select 1% of the vector's 768 dimensions, meaning 8 dimensions. 

    discrete: Optional[bool] --> Removed - use discrete = False in the predict() method to get raw similarities!
        Whether to round predictions to 0 or 1. False means the raw similarities are returned, which is handy if you want to evaluate the model's performance, eg by sorting the predictions in descending order or similarity score and reading through the texts. 

    default_cutoff: Optional[float]
        You should provide custom cutoffs for all the target variables (see the criteria parameter below and the tune method), but if not there is the default.

    criteria: Optional[dict(dict)]
        The cutoffs and selections for each variable. Best optimized by using the tune method. 

    mutually_exclusive: Optional[bool]
        Set to True if only one of the target variables can equal 1.

    n_decimals: Optional[int]
        The number of decimal points for rounding the output.

    trust_remote_code: Optional[bool]
        Setting for the sentence transformer library. Defaults to False.

    show_progress_bar: Optional[bool]
        Whether to show a progress bar or not while encoding text into embeddings.

    use_sample_probability: Optional[bool]
        As a fallback, use sample probability to categorize variables. Not recommended. 
    """


    def __post_init__(self):
        """Initialize everything else"""
        # Initialize all the internal dictionaries by name
        properties = "training_indices prediction_indices central_tendency dispersion dispersion_rank combined_rank filter similarities predictions cutoffs selections".split()
        for p in properties:
            self.__setattr__(p, dict())
        self.selection_key = 'selection'
        self.sim_key = 'similarity'
        self.prob_key = 'probability'
        # import torch
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print('Using device:', device)
        self.sentence_model = SentenceTransformer(model_name_or_path = self.model,
                                                  # device = device,
                                                  trust_remote_code = self.trust_remote_code,
                                                  tokenizer_kwargs = {'clean_up_tokenization_spaces': True}) # this setting gets rid of a warning in Transformers 4.45.1
        self.n_dims = self.n_dims or self.sentence_model.get_sentence_embedding_dimension()
        # placeholders
        for property in 'trained training_data prediction_data'.split():
            self.__dict__[property] = None


    # TRAINING

    def separate_training_data(self):
        """Separate the training data into the different categories and isolate the target columns"""
        if not self.training_data:
            raise Exception('No training data given')

        # Unpack training data, in case it's a generator
        self.training_data = [*ones_to_int(self.training_data, self.targets)]
        self.training_nrows = len(self.training_data)
        # self.training_data = [*remove_unused_keys(self.training_data, exceptions = [self.text_column])]
        self.training_data = [*remove_unused_keys(self.training_data, exceptions = [self.text_column, self.vector_column])]
        # Infer the target columns if not explicitly given
        if not self.targets:
            # self.targets = sorted(list(set(get_used_keys(self.training_data)) - set([self.text_column] + self.confounders + self.id_columns + self.ignore)))
            self.targets = sorted(list(set(get_used_keys(self.training_data)) - set([self.text_column, self.vector_column] + self.confounders + self.id_columns + self.ignore)))
        # Put confounders last
        self.targets = sorted(list(set(self.targets))) + sorted(list(set(self.confounders)))

        for t in self.targets:
            self.training_indices[t] = [i for i,v in enumerate(self.training_data) if t in v and int(v[t]) == 1]
        # self.use_sample_probability = any([self.prob_key in x and not x[self.prob_key] for x in self.criteria.values()])
        if self.use_sample_probability:
            if self.training_nrows < 100:
                sys.exit('Cannot predict using sample probabilities when the number of training samples is less than 100, meaning each percent corresponds to more than one sample. If you want your predictions to follow the same distribution as the training data (which should only be considered for large and unnaturally predictable data streams), add more training data. If not, remove all instances of { "probability": None } in your "criteria" or specify an exact number between 0 and 1. \nExited.')
            print('***~~~ CALCULATING SAMPLE PROBABILITIES ~~~***')
            self.sample_probabilities = { k:round(len(v) / self.training_nrows, self.n_decimals) for k,v in self.training_indices.items() }
        else:
            self.sample_probabilities = { k: 0 for k,v in self.training_indices.items() }

        # If any confounders are present, get those indices
        self.confounder_indices = {i for i,v in enumerate(self.training_data) for c in self.confounders if c in v and int(v[c]) == 1} if len(self.confounders) > 0 else set()


    def remove_target(self, target):
        """ Safely remove a target column """
        if target in self.targets:
            self.targets.remove(target)
        if target in self.confounders:
            self.confounders.remove(target)


    def encode_training_data(self, pre_encoded):
        if pre_encoded:
            # Text is pre-encoded embedding
            # if isinstance(self.training_data[0][self.text_column], str):
            #     warnings.warn(f'Argument pre_encoded is set to {pre_encoded} but the first row of your text column seems to be raw text.', stacklevel = 2)
            # self.training_embeddings = np.vstack([r[self.text_column] for r in self.training_data])
            vector_column = self.vector_column or self.text_column
            if isinstance(self.training_data[0][vector_column], str):
                warnings.warn(f'Argument pre_encoded is set to {pre_encoded} but the first row of your vector column "{vector_column}" seems to be raw text.', stacklevel = 2)
            self.training_embeddings = np.vstack([r[vector_column] for r in self.training_data])

        else:
            # Text is string
            if isinstance(self.training_data[0][self.text_column], (np.ndarray, list)):
                warnings.warn(f'Argument pre_encoded is set to {pre_encoded}, meaning column "{self.text_column}" should be raw text, but at least one of these values is a numpy array or a list. Are you using pre-encoded embeddings?', stacklevel = 2)
            self.training_embeddings = np.vstack(self.sentence_model.encode([r[self.text_column] or '' for r in self.training_data], show_progress_bar = self.show_progress_bar))
            print('Done encoding training data')
            print()


    def reduce(self):
        """Calculate median (typical vector) and standard deviation (dispersion) by category"""
        # Calculate central tendency and dispersion for the category
        for category in self.targets:
            indices = self.training_indices[category] if category in self.confounders else sorted(list(set(self.training_indices[category]) - self.confounder_indices))
            vectors = self.training_embeddings[indices, :]
            self.central_tendency[category] = np.apply_along_axis(np.median, 0, vectors)
            self.dispersion[category] = np.apply_along_axis(np.std, 0, vectors)

            # Calculate central tendency and dispersion for all the other rows.
            adversarial_indices = sorted(list(set([i for i in range(self.training_nrows) if i not in indices]) - self.confounder_indices))
            adversarial_vectors = self.training_embeddings[adversarial_indices, :]
            adversarial_central_tendency = np.apply_along_axis(np.median, 0, adversarial_vectors)

            dispersion_rank = self.dispersion[category].argsort().argsort()

            adversarial_ct_diff = np.abs(self.central_tendency[category] - adversarial_central_tendency)
            adversarial_ct_diff_rank = self.n_dims - adversarial_ct_diff.argsort().argsort()
            # Favour rows with a low dispersion and a high difference in central tendency toward other texts
            self.combined_rank[category] = dispersion_rank + adversarial_ct_diff_rank


    def assemble_filter(self):
        for category in self.targets:
            if category in self.criteria:
                if self.selection_key in self.criteria[category] and self.criteria[category][self.selection_key]:
                    self.selections[category] = self.criteria[category][self.selection_key]
                else:
                    self.selections[category] = self.default_selection
            else:
                self.selections[category] = self.default_selection
        for category in self.targets:
            if self.selections[category] == 1:
                # If all dimensions are to be used, assign all ones to the filter mask
                self.filter[category] = np.ones(self.n_dims, dtype = int)
            else:
                p = np.percentile(self.combined_rank[category], self.selections[category] * 100)
                self.filter[category] = np.where(self.combined_rank[category] < p, 1, 0)


    def train(self, data, pre_encoded = False):

        self.training_data = data or self.training_data
        if not self.training_data:
            raise Exception('Missing input data for training. You need to provide training data (list of dicts) with either a text column or pre-encoded embeddings.')

        # if not data:
        #     raise Exception('Missing input for training. You need to provide training data (list of dicts) with either a text column or pre-encoded embeddings.')
        # self.training_data = data
        self.separate_training_data()
        self.encode_training_data(pre_encoded)
        self.reduce()
        self.assemble_filter()
        self.trained = True


    def to_npz(self, filename, compressed = False):

        if not self.trained:
            raise Exception('You need to train the model using the .train() method before saving it to disk.')

        sample_probabilities = [self.sample_probabilities[t] for t in self.targets]

        (np.savez_compressed if compressed else np.savez) \
        (filename,
         central_tendency = dict_to_matrix(self.central_tendency, self.targets), 
         filter = dict_to_matrix(self.filter, self.targets),
         targets = self.targets,
         confounders = self.confounders,
         combined_rank = dict_to_matrix(self.combined_rank, self.targets),
         sample_probabilities = sample_probabilities)
        print(f'Saved model to {os.path.join(os.getcwd(), filename)}.')
        print()

    def from_npz(self, filename):

        filename = filename if filename.endswith('npz') else filename + '.npz'
        a = np.load(filename)

        # load the rest
        for property in 'targets confounders'.split():
            self.__dict__[property] = a[property].tolist() # Convert from numpy string array

        # sample_probabilities
        for property in 'central_tendency filter combined_rank'.split():
            self.__dict__[property] = matrix_to_dict(a[property], self.targets)

        self.__dict__['sample_probabilities'] = { key: a['sample_probabilities'][i] for i,key in enumerate(self.targets) }

        self.trained = True
        print(f'Loaded model from {os.path.join(os.getcwd(), filename)}.')
        print()


    # PREDICTION

    def encode_prediction_data(self, pre_encoded):
        if pre_encoded:
            vector_column = self.vector_column or self.text_column
            text_column_present = [vector_column in r for r in self.prediction_data]
            if not any([vector_column in r for r in self.prediction_data]):
                raise Exception(f'None of the records contains the given text field "{vector_column}"')
            if not all([vector_column in r for r in self.prediction_data]):
                raise Exception(f'At least one of the records does not contain the given text field "{vector_column}"')
        else:
            text_column_present = [self.text_column in r for r in self.prediction_data]
            if not any([self.text_column in r for r in self.prediction_data]):
                raise Exception(f'None of the records contains the given text field "{self.text_column}"')
            if not all([self.text_column in r for r in self.prediction_data]):
                raise Exception(f'At least one of the records does not contain the given text field "{self.text_column}"')

        if pre_encoded:
            # if isinstance(self.prediction_data[0][self.text_column], str):
            #     warnings.warn(f'Argument pre_encoded is set to {pre_encoded} but the first row of your text column seems to be raw text.', stacklevel = 2)
            # self.prediction_embeddings = np.vstack([r[self.text_column] for r in self.prediction_data])
            if isinstance(self.prediction_data[0][vector_column], str):
                warnings.warn(f'Argument pre_encoded is set to {pre_encoded} but the first row of your vector column "{vector_column}" seems to be raw text.', stacklevel = 2)
            self.prediction_embeddings = np.vstack([r[vector_column] for r in self.prediction_data])
        else:
            if isinstance(self.prediction_data[0][self.text_column], (np.ndarray)):
                warnings.warn(f'Argument pre_encoded is set to {pre_encoded}, meaning column "{self.text_column}" should be raw text, but at least one of these values is a numpy array. Are you using pre-encoded embeddings?', stacklevel = 2)
            self.prediction_embeddings = np.vstack(self.sentence_model.encode([r[self.text_column] or '' for r in self.prediction_data], show_progress_bar = self.show_progress_bar))
            print('Done encoding prediction data')
            print()


    def measure_similarity(self):
        if not self.prediction_data:
            raise Exception('No prediction data given')
        for category, typical_vector in self.central_tendency.items():
            # Dimensionality reduction by only using those dimensions with the lowest variance (precision) and the biggest difference in central tendency (accuracy)
            filter = self.filter[category].astype(bool)
            typical_vector = typical_vector[filter]
            pred_vectors = self.prediction_embeddings[:, filter]
            # Calculate similarity between all the rows in the prediction data and all the typical vectors
            self.similarities[category] = (pred_vectors @ typical_vector) / (np.linalg.norm(pred_vectors, axis=1) * np.linalg.norm(typical_vector))
            self.similarities[category] = np.round(self.similarities[category], decimals = self.n_decimals)


    def binarize(self):
        for t in self.targets:
            self.prediction_indices[t] = [i for i,v in enumerate(self.prediction_data) if t in v and v[t] and int(v[t]) == 1]
        # print()
        # print('self.prediction_indices["Funktion"]:')
        # print(self.prediction_indices['Funktion'])
        # print()
        self.test_probabilities = { k:round(len(v) / self.prediction_nrows, self.n_decimals) for k,v in self.prediction_indices.items() }
        # redundant_columns = set(self.criteria) - set(self.targets)
        # print(f'Redundant criteria not found in the data: {", ".join(redundant_columns)}' if len(redundant_columns) > 0 else '')
        self.sufficient_sample = { k:len(self.prediction_indices[k]) >= math.ceil(1 / self.sample_probabilities[k]) if self.sample_probabilities[k] > 0 else False for k in self.targets } if self.use_sample_probability else dict()
        for category in self.targets:
            if category in self.criteria: # if custom rule present
                criterion = self.criteria[category]
                if self.sim_key in criterion and criterion[self.sim_key]:
                    self.cutoffs[category] = round(criterion[self.sim_key], self.n_decimals)
                # Classifying documents by probability is a possible fallback for when you have a large and very predictable dataset. Avoid unless necessary.
                elif self.prob_key in criterion: # if probability specified
                    if criterion[self.prob_key]:
                        probability = criterion[self.prob_key]
                    else:
                        if self.sufficient_sample[category]:
                            probability = np.clip(cutoff[self.prob_key] or self.sample_probabilities[category], a_min = 0, a_max = 1) # Limit probabilities to (0,1)
                        else:
                            print()
                            warnings.warn(f'Key "{category}" has sample probability {self.sample_probabilities[category]:.2f} but only {len(self.prediction_indices[category])} rows in the prediction data, meaning there can be no positives. The prediction data needs to have at least math.ceil(1 / sample_probability) = {math.ceil(1/self.sample_probabilities[category])} rows where "{category}" = 1. Using default similarity cutoff ({self.default_cutoff}) instead. Consider using a custom similarity cutoff instead.', stacklevel = 2)
                            print()
                            self.cutoffs[category] = self.default_cutoff
                    self.cutoffs[category] = round(np.percentile(self.similarities[category], (1 - probability) * 100), self.n_decimals)
                else: # if no custom rule present
                    self.cutoffs[category] = self.default_cutoff
            else: # if no custom rule present
                self.cutoffs[category] = self.default_cutoff
            # Turn similarities into binary variables
            self.predictions[category] = np.where(self.similarities[category] >= self.cutoffs[category], 1, 0).astype(int)

        if self.confounders:
            predicted_confounders = np.vstack([self.predictions[key] for key in self.confounders])
            # Summarize to check if at least one confounder is predicted
            predicted_confounders = np.apply_along_axis(np.sum, 0, predicted_confounders)

            actual_targets = sorted(list(set(self.targets) - set(self.confounders)))
            # If any confounder = 1, all target variables are set to 0
            for t in actual_targets:
                self.predictions[t] = np.where(predicted_confounders > 0, 0, self.predictions[t])



    def max_category(self, pre_encoded, keep_vector):
        """Return the most likely category, out of all candidates"""
        self.output_fieldnames = [self.text_column] + self.targets
        # similarity_matrix = np.zeros((self.prediction_nrows, len(self.targets)))
        # i = 0
        # for target in self.targets:
        #     similarity_matrix[:,i] = self.similarities[target]
        #     i += 1
        similarity_matrix = dict_to_matrix(self.similarities, self.targets)
        self.index_predictions = np.argmax(similarity_matrix, axis = 1).astype(int)
        max_scores = np.max(similarity_matrix, axis = 1)

        # other_keys = sorted(list(get_used_keys(self.prediction_data) - set(self.targets)))
        other_keys = sorted(list(get_used_keys(self.prediction_data))) # Keep original variables
        if pre_encoded:
            other_keys.remove(self.vector_column or self.text_column)
        other_data = filter_keys(self.prediction_data, other_keys)
        predicted_data = dict_to_records({'predicted': [ self.targets[index] for index in self.index_predictions ],
                                          'score': [ np.round(score, self.n_decimals) for score in max_scores ]
                                         })

        if keep_vector:
            # vector_column = self.text_column if pre_encoded else self.vector_column
            vector_column = self.vector_column or self.text_column
            embeddings = ({ vector_column: e } for e in self.prediction_embeddings)
            self.output = zip_longest(other_data, predicted_data, embeddings)
        else:
            self.output = zip_longest(other_data, predicted_data)



    def update_predictions(self, discrete, pre_encoded, keep_vector):
        self.output_fieldnames = [self.text_column] + self.targets
        other_keys = sorted(list(get_used_keys(self.prediction_data) - set(self.targets)))
        if pre_encoded:
            other_keys.remove(self.vector_column or self.text_column)
        other_data = filter_keys(self.prediction_data, other_keys)
        # predicted_data = dict_to_records(self.predictions if self.discrete else self.similarities)
        predicted_data = dict_to_records(self.predictions if discrete else self.similarities)

        if keep_vector:
            # vector_column = self.text_column if pre_encoded else self.vector_column
            vector_column = self.vector_column or self.text_column
            embeddings = ({ vector_column: e } for e in self.prediction_embeddings)
            self.output = zip_longest(other_data, predicted_data, embeddings)
        else:
            self.output = zip_longest(other_data, predicted_data)


    def predict(self, data = None, discrete = True, pre_encoded = False, validation = False, keep_vector = False):

        if not self.trained:
            raise Exception('You need to train the model (or load a trained model from file) before calling the predict method')
        # if validation and not self.discrete:
        if validation and not discrete:
            sys.exit('Cannot do validation of continuous data, as there is nothing to compare with. Set discrete = True when validation = True.')
        if data:
            self.prediction_data = [*ones_to_int(data, self.targets)]
            self.prediction_nrows = len(self.prediction_data)
            self.encode_prediction_data(pre_encoded)
        if not self.prediction_data:
            raise Exception('Need to provide the classifier with prediction data.')
        self.measure_similarity()
        self.binarize()
        if self.mutually_exclusive:
            self.max_category(pre_encoded, keep_vector)
        else:
            self.update_predictions(discrete, pre_encoded, keep_vector)
        if validation:
            self.validate()
        if keep_vector:
            for a,b,c in self.output:
                yield {**a, **b, **c}
        else:
            for a,b in self.output:
                yield {**a, **b}



    def validate(self):
        """Compare predicted classes to actual classes (where available)"""
        self.validation_data = self.prediction_data.copy()
        keys = get_used_keys(self.validation_data)

        # redundant_keys = sorted(list(keys - set(self.targets) - set(self.confounders + [self.text_column] + self.id_columns + self.ignore)))
        # print('Redundant keys in the prediction data: ', redundant_keys)
        missing_keys = sorted(list(set(self.targets) - keys))
        if len(missing_keys) > 0:
            warnings.warn(f'Missing keys in the prediction data: {missing_keys}', stacklevel = 2)

        keys = sorted(list(set(keys).intersection(self.targets)))

        # self.validation_data =  [*filter_keys(self.validation_data, [self.text_column] + keys)]
        self.validation_data =  [*filter_keys(self.validation_data, keys)]

        validation_matrix = records_to_dict(self.validation_data, keys)
        validation_matrix = dict_to_matrix(validation_matrix, keys)

        # Error rate by target variable
        if self.mutually_exclusive:

            similarity_matrix = dict_to_matrix(self.similarities, self.targets)
            similarity_matrix = (similarity_matrix == similarity_matrix.max(axis = 1, keepdims = True)).astype(int)

            diff_matrix = validation_matrix - similarity_matrix
            nrows = diff_matrix.shape[0]

            false_positive = np.round(np.sum(np.where(diff_matrix < 0, 1, 0), axis = 0) / nrows, decimals = self.n_decimals)
            false_negative = np.round(np.sum(np.where(diff_matrix > 0, 1, 0), axis = 0) / nrows, decimals = self.n_decimals)
            overall = np.round(np.sum(np.abs(diff_matrix), axis = 0) / nrows, decimals = self.n_decimals)
            self.error_rate = dict(false_positive = dict(zip(keys, false_positive)),
                                   false_negative = dict(zip(keys, false_negative)),
                                   overall = dict(zip(keys, overall)))

            # Confusion matrix
            # rows: validation, columns: predicted
            self.confusion = confusions(validation_matrix, similarity_matrix, keys, nrows, self.n_decimals)


            # Error rate by row
            # For mutually exclusive variables, the error for a single row is all or nothing

        else:
            prediction_matrix = dict_to_matrix(self.predictions, keys)
            diff_matrix = validation_matrix - prediction_matrix
            nrows = diff_matrix.shape[0]
            false_positive = np.round(np.sum(np.where(diff_matrix < 0, 1, 0), axis = 0) / nrows, decimals = self.n_decimals)
            false_negative = np.round(np.sum(np.where(diff_matrix > 0, 1, 0), axis = 0) / nrows, decimals = self.n_decimals)
            overall = np.round(np.sum(np.abs(diff_matrix), axis = 0) / nrows, decimals = self.n_decimals)
            abs_diff = np.abs(validation_matrix - prediction_matrix)
            self.error_rate = dict(false_positive = dict(zip(keys, false_positive)),
                                   false_negative = dict(zip(keys, false_negative)),
                                   overall = dict(zip(keys, overall)))

            # Error rate by row (share of misclassifications)
            ncols = len(keys)
            false_positive = np.round(np.sum(np.where(diff_matrix < 0, 1, 0), axis = 1) / ncols, decimals = self.n_decimals)
            false_negative = np.round(np.sum(np.where(diff_matrix > 0, 1, 0), axis = 1) / ncols, decimals = self.n_decimals)
            overall = np.round(np.sum(np.abs(diff_matrix), axis = 1) / ncols, decimals = self.n_decimals)
            self.error_rate_by_row = [ dict(false_positive=p, false_negative=n, overall=o)
                                       for (p,n,o) in zip(false_positive, false_negative, overall) ]


    def calculate_loss(self, all = False):
        self.assemble_filter()
        _ = [*self.predict(validation = True)]
        if all:
            return self.error_rate
        else:
            return { target: abs(self.error_rate.get('false_positive').get(target) - self.error_rate.get('false_negative').get(target)) for target in self.targets }



    def tune(self, data = None, pre_encoded = False, param_name = 'similarity', param_range = (0, 1, 0.01), plot = False, show_progress = True):

        if not data and not self.prediction_data:
            raise Exception('Need to provide the classifier with prediction data.')
        if data and not self.prediction_data:
            self.prediction_data = [*ones_to_int(data, self.targets)]
            self.prediction_nrows = len(self.prediction_data)
            self.encode_prediction_data(pre_encoded)

        unit = clamp(param_range[2], 0, 1)
        start = clamp(param_range[0], unit, 1)# - unit) 
        stop = clamp(param_range[1], unit, 1)# - unit)

        targets = self.targets.copy()
        confounders = self.confounders.copy()
        optimal_values = dict(((k, None) for k in targets))
        fp = dict(((k, []) for k in targets))
        fn = dict(((k, []) for k in targets))
        overall_errors = dict(((k, []) for k in targets))
        absdiff = dict(((k, []) for k in targets))

        lower_end = list()
        upper_end = list()

        parameter_multiplier = 1 / unit
        parameter_range = [x / parameter_multiplier for x in range(round(start * parameter_multiplier),
                                               round((stop + unit) * parameter_multiplier),
                                               round(unit * parameter_multiplier))]
        n_iterations = len(parameter_range)

        # LINE_UP = '\033[1A'
        # LINE_CLEAR = '\x1b[2K'
        start_time = time.time()
        icon = cycle(['  |   ', '  _   ', '  __  ', '   __ ', '    __', '     _', '     |', 
                      '     \u203e', '    \u203e\u203e', '   \u203e\u203e ', '  \u203e\u203e  ', '  \u203e   '])

        for i,iteration in enumerate(parameter_range):
            if i > 0:
                current_time = time.time()
                time_elapsed = current_time - start_time
                time_left = time_elapsed * ((n_iterations - i) / i)
                if show_progress:
                    # Display progress
                    # print(LINE_UP, end=LINE_CLEAR)
                    print(f'{next(icon)} Iteration {i + 1} of {n_iterations} estimating {param_name}, time left: {round(time_left)} seconds', end='\r', flush = True)

            for target in self.targets: 
                # # Merge parameter with existing criteria for each target
                if self.criteria.get(target):
                    self.criteria[target].update({ param_name: iteration })
                else:
                    self.criteria[target] = { param_name: iteration }

            current_error = self.calculate_loss(all = True)

            for target in self.targets.copy():
                fp[target].append(current_error['false_positive'][target])
                fn[target].append(current_error['false_negative'][target])
                overall_errors[target].append(current_error['overall'][target])
                absdiff[target].append(abs(current_error['false_positive'].get(target) - current_error['false_negative'].get(target)))

                if not plot and param_name == self.sim_key: 
                    if i == 0: # Need 2 iterations to compare
                        worsening = False
                    else: 
                        if param_name == self.sim_key:
                            worsening = (fp[target][-1] - fn[target][-1]) < 0
                        last_iteration = i == (n_iterations - 1)

                        if last_iteration or worsening:
                            # If optimal input value found, stop trying to optimize the target at hand
                            # If last iteration and no improvement, return the last value
                            if last_iteration and (not worsening):
                                upper_end.append(target)
                            optimal_values[target] = { param_name: parameter_range[i-1 if worsening else i] }
                            if i == 1:
                                lower_end.append(target)
                            self.remove_target(target)
                        if len(self.targets) == 0:
                            for target in targets:
                                # if target in self.criteria:
                                if self.criteria.get(target):
                                    self.criteria[target].update(optimal_values[target])
                                    # optimal_values[target] = {**self.criteria[target], 
                                    #                      **optimal_values[target]}
                                else:
                                    self.criteria[target] = optimal_values[target]
                            if upper_end:
                                print()
                                warnings.warn(f'Optimal {param_name} was found to be the very last value for the following targets: {", ".join(lower_end)}. Consider extending the lower end of the range. ', stacklevel = 2)
                                print()
                            if not last_iteration:
                                print()
                                print('returning early')
                                print()
                            if lower_end:
                                print()
                                warnings.warn(f'Optimal {param_name} was found to be the very first value for the following targets: {", ".join(lower_end)}. Consider extending the lower end of the range. ', stacklevel = 2)
                                print()

            if len(self.targets) == 0:
                break

        print()
        if plot or param_name == self.selection_key:

            min_error = { key: None for key in targets }
            min_error_index = { key: None for key in targets }
            for target in targets:
                if param_name == self.sim_key:
                    loss = absdiff[target]
                elif param_name == 'selection':
                    loss = overall_errors[target]
                min_error[target] = min(loss)
                min_error_index[target] = loss.index(min_error[target])
                optimal_values[target] = { param_name: parameter_range[min_error_index[target]] }

            for target in targets:
                if self.criteria.get(target):
                    self.criteria[target].update(optimal_values[target])
                else:
                    self.criteria[target] = optimal_values[target]

        if plot:
            try:
                import plotext as plt
            except ImportError as s:
                raise Exception('Optional dependency plotext (>=5.3.2) needed for plotting. Otherwise set plot = False.')

            for target in targets:
                plt.vline(min_error_index[target] + 1, 'black')
                plt.plot(overall_errors[target], label = f'{target} overall error')
                plt.plot(fp[target], label = f'{target} false positive')
                plt.plot(fn[target], label = f'{target} false negative')
                plt.xticks(list(range(n_iterations)), parameter_range)
                plt.title(f'{target} validation error by {param_name}')
                plt.build()
                cwd = os.getcwd()
                plotpath = os.path.join(cwd, 'plots')
                if not os.path.exists(plotpath):
                    os.mkdir(plotpath)
                plt.save_fig(f'{plotpath}/{target}.html')
                plt.clear_figure()

        # Update and run the tuned classifier once more to refresh the error rate and so on
        self.targets = targets
        self.confounders = confounders
        # self.criteria = optimal_values
        self.assemble_filter()
        _ = [*self.predict(validation = True)]



    def write_csv(self, filename, discrete = True):
        if not self.prediction_data:
            sys.exit('No prediction data')
        import csv

        try: # Independent variable might be self.vector_column and then self.text_column might not exist in the data
            data = {**{self.text_column: [d[self.text_column] for d in self.prediction_data]}, 
                    **(self.predictions if discrete else self.similarities)}
        except: # If text column not found, try writing just the predictions
            data = self.predictions if discrete else self.similarities

        data = dict_to_records(data)
        data = [*filter_keys(data, self.output_fieldnames)]
        with open(filename, 'w', newline = '') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = self.output_fieldnames)
            writer.writeheader()
            writer.writerows(data)


    def correlation(self, threshold = 0.1, w = 20):

        print('\n' * 2)
        keys = self.targets
        pairs = combinations(keys, 2)
        similarities = list()
        for a,b in pairs:
            s1 = np.sum(self.filter[a])
            s2 = np.sum(self.filter[b])
            shared_dims = np.sum(self.filter[a] * self.filter[b])
            shared_info = np.round((shared_dims * 2) / (s1 + s2), self.n_decimals)
            v1 = self.central_tendency[a] * self.filter[a]
            v2 = self.central_tendency[b] * self.filter[b]
            ndim_a = str(sum(self.filter[a]))
            ndim_b = str(sum(self.filter[b]))
            similarity = np.round((v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), self.n_decimals)
            if similarity >= threshold:
                similarities.append((a,b,similarity,shared_info,ndim_a,ndim_b))
        similarities.sort(key = lambda x: (x[2],x[3]))
        similarities.reverse()
        print(f'{"TARGET1":<{w}}{"TARGET2":<{w}}{"SIMILARITY":<{w}}{"SHARED DIMENSIONS":<{w}}{"N DIMENSIONS":<{w}}')
        print('-' * w * 5)
        for a,b,c,d,e,f in similarities:
            print(f'{a:<{w}}{b:<{w}}{c:<{w}}{d:<{w}}{e + " / " + f:<{w}}')
        return similarities


    def distribution(self, w = 20):
        print()
        for category in 'targets confounders'.split():
            xs = [x for x in self.targets if not x in self.confounders] if category == 'targets' else self.confounders
            print()
            if not self.test_probabilities:
                print(f'{category.upper():<{w}}{"% OF TRAINING SAMPLE":<{w}}')
                print('-' * (w * 2))
                for target in xs:
                    print(f'{self.sample_probabilities[target]:<{w}}')
            else:
                print(f'{category.upper():<{w}}{"% TRAIN":<{w}}{"% TEST":<{w}}')
                print('-' * (w * 3))
                for target in xs:
                    print(f'{target:<{w}}{self.sample_probabilities[target]:<{w}}{self.test_probabilities[target]:<{w}}')


    def error(self, w = 20, margin = 5):
        print('\n' * 2)
        if not self.error_rate:
            raise Exception('Error rate not calculated. Please run predict with validation = True, then try again.')
        for category in 'targets confounders'.split():
            xs = [x for x in self.targets if not x in self.confounders] if category == 'targets' else self.confounders
            print()
            if self.mutually_exclusive:
                # No confounders for mutually exclusive data
                if category == 'targets':
                    print(f'{category.upper():<{w}}{"OVERALL":<{w}}{"FALSE POSITIVE":<{w}}{"FALSE NEGATIVE":<{w}}')
                    print('-' * (w * 5))
                    for target in xs:
                        print(f'{target:<{w}}{self.error_rate["overall"][target]:<{w}}{self.error_rate["false_positive"][target]:<{w}}{self.error_rate["false_negative"][target]:<{w}}')
                    print('\n' * 2)
                    w = w // 2
                    keys, confusion_matrix, actual_shares, predicted_shares = self.confusion
                    keys = [key[:(w - margin)] + ('...' if len(key) > (w - margin) else '') for key in keys ]
                    print('CONFUSION MATRIX')
                    print('rows: validation/actual, columns: predicted, values sum to 1 (=100%)')
                    print('Actual/Predicted % (row and column sum resp.) are calculated before rounding')
                    print('-' * (w * (len(keys) + 3)))
                    print((" " * w) + ''.join([f"{key:<{w}}" for key in keys]) + ' ' * (w//2) + 'Actual %')
                    for i in range(confusion_matrix.shape[0]):
                        row = ''.join([f"{x:<{round(w)}}" for x in confusion_matrix[i,:]])
                        print(f'{keys[i]:<{w}}' + row + ' ' * (w//2) + f'{actual_shares[i]:<{w}}')

                    print()
                    row = ''.join([f"{x:<{round(w)}}" for x in predicted_shares])
                    print(f'{"Pred. %":<{w}}' + row + ' ' * (w//2) + f'{1:<{w}}')
                    #
                    # print('actual_shares:')
                    # print(actual_shares)

            else:
                print(f'{category.upper():<{w}}{"OVERALL":<{w}}{"FALSE POSITIVE":<{w}}{"FALSE NEGATIVE":<{w}}{"THRESHOLD":<{w}}')
                print('-' * (w * 5))
                for target in xs:
                    print(f'{target:<{w}}{self.error_rate["overall"][target]:<{w}}{self.error_rate["false_positive"][target]:<{w}}{self.error_rate["false_negative"][target]:<{w}}{self.cutoffs[target]:<{w}}')


    def error_by_row(self):
        # TODO
        pass


    def examples(self, targets = None, n = 10, w = 10, margin = 5):
        if self.mutually_exclusive:
            raise Exception('Classifier.examples() method not yet available for mutually exclusive data.')

        if isinstance(self.prediction_data[0][self.text_column], np.ndarray):
            raise Exception(f'Your text column "{self.text_column}" seems to be an embedding. The .examples() method is meant to display raw text.')

        targets = targets or self.targets
        sim_w = self.n_decimals + margin
        print('\n' * 2)

        ones = dict()
        zeros = dict()
        for t in targets:
            ones[t] = sorted([(i,v, self.prediction_data[i][self.text_column]) for i,v in enumerate(self.similarities[t]) if v >= self.cutoffs[t]], key = lambda x: x[1], reverse = True)
            zeros[t] = sorted([(i,v, self.prediction_data[i][self.text_column]) for i,v in enumerate(self.similarities[t]) if v < self.cutoffs[t]], key = lambda x: x[1], reverse = True)
            _n = min(n, len(ones[t]) // 2, len(zeros[t]) // 2)

            target_count = len(self.prediction_indices[t])
            if target_count < n:
                print(f'Target {t} not displayed. Count is {target_count} ({target_count / self.prediction_nrows * 100:.{self.n_decimals}f}% of the prediction data), which is less than n = {n}.')
            else:
                print(f'{t}, threshold = {self.cutoffs[t]}, count = {target_count} / {self.prediction_nrows} = {target_count / self.prediction_nrows * 100:.{self.n_decimals}f}%')
                print(f'{"TOP ONES (best matches)":<{w * 3 + sim_w}}{"BOTTOM ONES (look for false positives)":<{w * 3 + sim_w}}{"TOP ZEROS (look for false negatives)":<{w * 3 + sim_w}}')
                print(f'{"Text":<{w * 3}}{"Sim.":<{sim_w}}' * 3)
                print('-' * 3 * (3 * w + sim_w))
                m = len(ones[t]) - 1
                for i in range(_n):
                    print(f'{ones[t][i][2][:(w * 3) - margin] + ("..." if len(ones[t][i][2]) > (w * 3) - margin else ""):<{(w * 3)}}{ones[t][i][1]:<{sim_w},.{self.n_decimals}f}{ones[t][m - _n + i][2][:(w * 3) - margin] + ("..." if len(ones[t][m-n+i][2]) > (w * 3) - margin else ""):<{(w * 3)}}{ones[t][m-n+i][1]:<{sim_w},.{self.n_decimals}f}{zeros[t][i][2][:(w * 3) - margin] + ("..." if len(zeros[t][i][2]) > (w * 3) - margin else ""):<{(w * 3)}}{zeros[t][i][1]:<{sim_w},.{self.n_decimals}f}')
            print('\n' * 2)



    def to_pipeline(self, **kwargs):

        # Make sure our classifier is using the latest text column and possibly vector column for prediction, if given in kwargs
        if kwargs.get('text_column'):
            text_column = kwargs.get('text_column')
            if text_column != self.text_column:
                print(f'Using text column "{text_column}" for prediction instead of the previous one: "{self.text_column}".')
            self.text_column = text_column

        if kwargs.get('vector_column'):
            vector_column = kwargs.get('vector_column')
            if vector_column != self.vector_column:
                print(f'Using vector column "{vector_column}" for prediction instead of the previous one: "{self.vector_column}".')
            self.vector_column = vector_column

        kwargs['pre_encoded'] = if_undefined(kwargs.get('pre_encoded'), False) # Needs default value

        kwargs['drop'] = kwargs.get('drop') or list()

        if kwargs.get('pre_encoded'):
            kwargs['split'] = False # Cannot split records with pre-encoded vectors
            if self.vector_column:
                kwargs['text_column'] = self.text_column
                kwargs['vector_column'] = self.vector_column
            else:
                kwargs['text_column'] = None # Ignore and drop text_column if it's a pre-encoded vector
                kwargs['drop'].append(self.text_column)
        else:
            kwargs['split'] = if_undefined(kwargs.get('split'), True) # Needs default value

        kwargs['combine'] = if_undefined(kwargs.get('combine'), True) # Needs default value

        if kwargs.get('keep_vector') and kwargs.get('combine'):
            raise Exception(f'Arguments keep_vector and combine cannot simultaneously be set to True, as vectors cannot be concatenated the same way as raw text. Exiting!')

        predict_args = "pre_encoded validation keep_vector".split()
        predict_args = { k:v for k,v in kwargs.items() if k in predict_args }

        split_args = "text_column max_sequence_length doc_id_column row_id_column chunk_id_column overlap per_sentence".split()
        split_args = { k:v for k,v in kwargs.items() if k in split_args }

        # combine_args = "text_separator text_column doc_id_column row_id_column chunk_id_column original_data binary_targets aggregation overlap".split()
        combine_args = "text_separator text_column vector_column doc_id_column row_id_column chunk_id_column aggregation overlap drop".split()
        combine_args = { k:v for k,v in kwargs.items() if k in combine_args }

        from types import GeneratorType

        def inner_function(prediction_data):
            if isinstance(prediction_data, (list, tuple)):
                # prediction_data = iter(prediction_data)
                # prediction_data = prediction_data.__iter__()
                prediction_data = (r for r in prediction_data)

            if isinstance(prediction_data, GeneratorType):

                prediction_data = ones_to_int(prediction_data, self.targets)
                prediction_data = [*prediction_data] # Need to unpack, no way to copy a generator
                # original_data = prediction_data.copy()

                if kwargs.get('split'):
                    original_data = prediction_data.copy()
                    prediction_data = split_records(prediction_data, **split_args)
                predicted = self.predict(data = prediction_data, **predict_args)

                if kwargs.get('combine'):
                    if kwargs.get('split'):
                        predicted = combine_records(predicted, original_data = original_data, binary_targets = self.targets, **combine_args)
                    else:
                        predicted = combine_records(predicted, binary_targets = self.targets, **combine_args)
                yield from predicted
            else:
                raise Exception(f'Unknown/incompatible data type: {type(prediction_data)}')

        return inner_function


