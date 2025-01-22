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


def combine_two_records(binary_targets, text_separator, text_column, doc_id_column, row_id_column, chunk_id_column, overlap, original_data, a = None, b = None):
    """ Merge dicts belonging to the same document, as indicated by the doc_id_column """
    from collections.abc import Sequence

    binary_targets = binary_targets or [] # Cannot iterate over None or turn it into a set
    doc_ids = set([a[doc_id_column], b[doc_id_column]])
    if len(doc_ids) > 1:
        raise Exception("Cannot merge two documents with different doc ids.")

    common_keys = { text_column, doc_id_column }
    # Where available, use values from the original fulltext record
    if original_data:
        original_keys = set(original_data[a[doc_id_column]].keys()).union(set(original_data[a[doc_id_column]].keys())) - common_keys - set(binary_targets)
    else:
        original_keys = set()

    remaining_keys = set(a.keys()).union(set(b.keys())) - common_keys - set(binary_targets) - original_keys

    if row_id_column:
        remaining_keys = remaining_keys - { row_id_column }
    if chunk_id_column:
        remaining_keys = remaining_keys - { chunk_id_column }

    # If the second text is an empty string, ignore overlap as it might be the last text of the document
    if len(b['text']) == 0:
        overlap = 0

    # If overlap, remove it before joining
    # TODO: Overlap only if split by max_sequence_length (meaning chunk_id is present in a), not if split by punctuation
    # b_text = b[text_column] if (overlap == 0 or chunk_id_column not in b) else ''.join(re.split(pattern = r'\b', string = b[text_column])[:-overlap])
    # record = { doc_id_column: a[doc_id_column],
    #            text_column: text_separator.join([a[text_column], b_text]) }


    # a_text = a[text_column] if (overlap == 0 or chunk_id_column not in a) else ''.join(re.split(pattern = r'\b', string = a[text_column])[:-overlap - 1])
    # record = { doc_id_column: a[doc_id_column],
    #            text_column: text_separator.join([a_text, b[text_column]]) }

    # TODO: Error when last record of doc is an empty string
    if (overlap == 0):
        a_text = a[text_column]
    else:
        tokens = re.split(pattern = r'\b', string = a[text_column])
        if (chunk_id_column in b and b[chunk_id_column] < 0):
            a_text = ''.join(tokens[:-overlap - 1])
        else:
            a_text =  ''.join(tokens)
    record = { doc_id_column: a[doc_id_column],
               text_column: text_separator.join([a_text, b[text_column]]) }

    for target in binary_targets:
        if not target in a and not target in b:
            raise Exception(f'Target {target} not found in dicts: \n\n{a} \n\n{b}')
        elif target in a and not target in b:
            record[target] = a[target]
        elif not target in a and target in b: record[target] = b[target]
        else:
            record[target] = a[target] + b[target]

    for key in original_keys:
        record[key] = a[key] if key in a else b[key]

    for key in remaining_keys:
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


def combine_records(records, text_separator = '', text_column = 'text', doc_id_column = 'doc_id', row_id_column = 'sentence_id', chunk_id_column = 'chunk_id', original_data = None, binary_targets = None, aggregation = 'any', overlap = 0):
    """ Combines dicts and concatenates text """

    records = iter(records)
    it = records.__iter__()
    previous_record = next(it)
    first_record = previous_record
    doc_id = previous_record[doc_id_column]
    current_doc = list()

    # strategies = 'any absolute sum most majority relative share'.split()

    if original_data:
        len1 = len(original_data)
        original_data = { (r[doc_id_column] if doc_id_column in r else i): r for i,r in enumerate(original_data) }
        len2 = len(original_data.keys())
        if len2 < len1:
            raise Exception(f'Doc ids missing or duplicate in at least one record of the original data. {len1} records contain {len2} unique keys.')

    glue = partial(combine_two_records, binary_targets, text_separator, text_column, doc_id_column, row_id_column, chunk_id_column, overlap, original_data)

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
                    current_doc = [{ text_column: '', doc_id_column: doc_id, row_id_column: 1 }]

                # Concat in order of row_id, not necessarily the order of the data (it might have been scrambled for some reason)
                if row_id_column:
                    current_doc.sort(key = lambda r: r[row_id_column])


                fulltext = reduce(glue, current_doc, first_record)
                n_rows = 1 + len(current_doc)

                # Aggregate targets
                if binary_targets and aggregation:
                    for target in binary_targets: 
                        if not type(fulltext[target]) in (int, np.integer):
                            obj = { text_column: fulltext[text_column], target: fulltext[target] }
                            raise Exception(f'Target variable is not integer (Python int or numpy.integer): \n {obj}')
                        if not aggregation or aggregation in 'absolute sum'.split():
                            pass # keep the summed number
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
                current_doc = [{ text_column: '', doc_id_column: doc_id, row_id_column: 1 }]
            # Repeat the above for the very last record
            if row_id_column:
                current_doc.sort(key = lambda r: r[row_id_column])
            fulltext = reduce(glue, current_doc, first_record)
            n_rows = 1 + len(current_doc)

            # Aggregate targets
            if binary_targets and aggregation:
                for target in binary_targets: 
                    if not aggregation or aggregation in 'absolute sum'.split():
                        pass # keep the summed number
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

def get_used_keys(records):
    return set().union(*(d.keys() for d in records))


def filter_keys(records, keys):
    for r in records: 
        yield { key:r[key] for key in keys if key in r }


def remove_unused_keys(records, exceptions):
    for r in records:
        yield { k:v for k,v in r.items() if k in exceptions or (v and not v == 0) }


def dict_to_records(d):
    for col in zip(*d.values()):
        yield dict(zip(d, col))


def records_to_dict(records, keys):
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



@dataclass
class Classifier:
    training_data: list[dict] = field(default_factory = list, repr = False)
    sentence_transformer: str = 'KBLab/sentence-bert-swedish-cased'
    text_column: str = 'text'
    targets: list[str] = field(default_factory = list)
    id_columns: list[str] = field(default_factory = list)
    confounders: list[str] = field(default_factory = list)
    ignore: list[str] = field(default_factory = list)
    default_selection: float = 0.01
    discrete: bool = False
    default_cutoff: float = 0.5
    criteria: dict[dict] = field(default_factory = dict)
    mutually_exclusive: bool = False
    n_decimals: int = 2
    n_dims: int = None
    trust_remote_code: bool = False
    show_progress_bar: bool = True
    use_sample_probability = True

    """
    Parameters
    ----------
    training_data :list(dict)
        List of dicts containing a text column (raw text or pre-encoded vector) and one or more binary columns

    sentence_transformer : Optional[str]
        The sentence_transformer model to be used (copy the name from Huggingface hub)

    text_column : str
        Name of the text column

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

    discrete: Optional[bool]
        Whether to round predictions to 0 or 1. False means the raw similarities are returned, which is handy if you want to evaluate the model's performance, eg by sorting the predictions in descending order or similarity score and reading through the texts. 

    default_cutoff: Optional[float]
        You should provide custom cutoffs for all the target variables (see the criteria parameter below and the tune function), but if not there is the default.

    criteria: Optional[dict(dict)]
        The cutoffs and selections for each variable. Best optimized by using the tune function. 

    mutually_exclusive: Optional[bool]
        Set to True if only one of the target variables can equal 1.

    n_decimals: Optional[int]
        The number of decimal points for rounding the output.

    trust_remote_code: Optional[bool]
        Setting for the sentence transformer library. Defaults to False.

    show_progress_bar: Optional[bool]
        Whether to show a progress bar or not while encoding text into embeddings.

    use_sample_probability: Optional[bool]
        As a fallback, use sample probability to categorize variables.
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
        self.sentence_model = SentenceTransformer(model_name_or_path = self.sentence_transformer,
                                                  # device = device,
                                                  trust_remote_code = self.trust_remote_code,
                                                  tokenizer_kwargs = {'clean_up_tokenization_spaces': True}) # this setting gets rid of a warning in Transformers 4.45.1
        self.n_dims = self.n_dims or self.sentence_model.get_sentence_embedding_dimension()
        # placeholders
        for property in 'trained training_data prediction_data'.split():
            self.__dict__[property] = None

        # Dereference list arguments, as these are modified by the tune() function
        self.targets = self.targets.copy()
        self.confounders = self.confounders.copy()


    # TRAINING

    def separate_training_data(self):
        """Separate the training data into the different categories and isolate the target columns"""
        if not self.training_data:
            raise Exception('No training data given')

        # Unpack training data, in case it's a generator
        self.training_data = [*self.training_data]
        self.training_nrows = len(self.training_data)
        self.training_data = [*remove_unused_keys(self.training_data, exceptions = [self.text_column])]
        # Infer the target columns if not explicitly given
        if not self.targets:
            self.targets = sorted(list(set(get_used_keys(self.training_data)) - set([self.text_column] + self.confounders + self.id_columns + self.ignore)))
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
            if isinstance(self.training_data[0][self.text_column], str):
                warnings.warn(f'Argument pre_encoded is set to {pre_encoded} but the first row of your text column seems to be raw text.', stacklevel = 2)
            self.training_embeddings = np.vstack([r[self.text_column] for r in self.training_data])

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

            if self.default_selection == 1:
                # If all dimensions are to be used, assign all ones to the filter mask
                self.filter[category] = np.ones(vectors.shape[1], dtype = int)
                # TODO: calculate combined_rank dummy variable
            else:
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
            p = np.percentile(self.combined_rank[category], self.selections[category] * 100)
            self.filter[category] = np.where(self.combined_rank[category] < p, 1, 0)


    def train(self, training_data, pre_encoded = False):
        if not training_data:
            raise Exception('Missing input for training. You need to provide training data (list of dicts) with either a text column or pre-encoded embeddings.')
        self.training_data = training_data
        self.separate_training_data()
        self.encode_training_data(pre_encoded)
        self.reduce()
        self.assemble_filter()
        self.trained = True


    def to_npz(self, filename, compressed = False):

        # self.assemble_filter()

        (np.savez_compressed if compressed else np.savez) \
        (filename,
         central_tendency = dict_to_matrix(self.central_tendency, self.targets), 
         filter = dict_to_matrix(self.filter, self.targets),
         targets = self.targets,
         confounders = self.confounders,
         combined_rank = dict_to_matrix(self.combined_rank, self.targets),
         sample_probabilities = [self.sample_probabilities[t] for t in self.targets])
        print(f'Saved model to {os.path.join(os.getcwd(), filename)}.')
        print()


    def from_npz(self, filename):

        filename = filename if filename.endswith('npz') else filename + '.npz'
        a = np.load(filename)

        # load the rest
        for property in 'targets confounders'.split():
            self.__dict__[property] = a[property].tolist() # Convert from numpy string array

        for property in 'sample_probabilities'.split():
            self.__dict__[property] = a[property]

        for property in 'central_tendency filter combined_rank'.split():
            self.__dict__[property] = matrix_to_dict(a[property], self.targets)
        self.sample_probabilities = dict([(t, a['sample_probabilities'][i,]) for i,t in enumerate(self.targets)])

        self.trained = True
        print(f'Loaded model from {os.path.join(os.getcwd(), filename)}.')
        print()


    # PREDICTION

    def encode_prediction_data(self, pre_encoded):
        text_column_present = [self.text_column in r for r in self.prediction_data]
        if not any([self.text_column in r for r in self.prediction_data]):
            raise Exception(f'None of the records contains the given text field "{self.text_column}"')
        if not all([self.text_column in r for r in self.prediction_data]):
            raise Exception(f'At least one of the records does not contain the given text field "{self.text_column}"')

        if pre_encoded:
            if isinstance(self.prediction_data[0][self.text_column], str):
                warnings.warn(f'Argument pre_encoded is set to {pre_encoded} but the first row of your text column seems to be raw text.', stacklevel = 2)
            self.prediction_embeddings = np.vstack([r[self.text_column] for r in self.prediction_data])
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
            self.prediction_indices[t] = [i for i,v in enumerate(self.prediction_data) if t in v and v[t] == 1]
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



    def max_category(self):
        """Return the most likely category, out of all candidates"""
        self.output_fieldnames = [self.text_column] + self.targets
        similarity_matrix = np.zeros((self.prediction_nrows, len(self.targets)))
        i = 0
        for target in self.targets:
            similarity_matrix[:,i] = self.similarities[target]
            i += 1
        self.index_predictions = np.argmax(similarity_matrix, axis = 1).astype(int)
        max_scores = np.max(similarity_matrix, axis = 1)

        other_keys = sorted(list(get_used_keys(self.prediction_data) - set(self.targets)))
        other_data = filter_keys(self.prediction_data, other_keys)
        predicted_data = dict_to_records({'label': [ self.targets[index] for index in self.index_predictions ],
                                          'score': [ score for score in max_scores ]
                                         })
        self.output = zip_longest(other_data, predicted_data)



    def update_predictions(self):
        self.output_fieldnames = [self.text_column] + self.targets
        other_keys = sorted(list(get_used_keys(self.prediction_data) - set(self.targets)))
        other_data = filter_keys(self.prediction_data, other_keys)
        predicted_data = dict_to_records(self.predictions if self.discrete else self.similarities)
        self.output = zip_longest(other_data, predicted_data)


    def predict(self, prediction_data = None, pre_encoded = False, validation = False):
        if not self.trained:
            raise Exception('You need to train the model (or load a trained model from file) before calling the predict method')
        if validation and not self.discrete:
            sys.exit('Cannot do validation of continuous data, as there is nothing to compare with. Set discrete = True when validation = True.')
        # self.assemble_filter() # moved to the train method and called separately by tune()
        if prediction_data:
            # self.prediction_data = prediction_data
            self.prediction_data = [*prediction_data]
            self.prediction_nrows = len(self.prediction_data)
            self.encode_prediction_data(pre_encoded)
        self.measure_similarity()
        # if self.discrete:
        self.binarize()
        if self.mutually_exclusive:
            self.max_category()
        else:
            self.update_predictions()
        if validation:
            self.validate()
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

        self.validation_data =  [*filter_keys(self.validation_data, [self.text_column] + keys)]

        validation_matrix = records_to_dict(self.validation_data, keys)
        validation_matrix = dict_to_matrix(validation_matrix, keys)

        # Error rate by variable
        if self.mutually_exclusive:
            validation_matrix = np.argmax(validation_matrix, axis = 1)
            diff_matrix = np.abs(validation_matrix - self.index_predictions)
            nrows = diff_matrix.shape[0]
            self.error_rate = dict(overall = np.sum(diff_matrix) / nrows)

            # Error rate by row
            # For mutually exclusive variables, the error for a single row is all or nothing
            overall = np.round(np.sum(np.abs(diff_matrix), axis = 1) / 2, decimals = self.n_decimals)
            self.error_rate_by_row = [dict(overall=o) for o in overall]

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


    def write_csv(self, filename):
        if not self.prediction_data:
            sys.exit('No prediction data')
        import csv
        data = {**{self.text_column: [d[self.text_column] for d in self.prediction_data]}, 
                **(self.predictions if self.discrete else self.similarities)}
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


    def error(self, w = 20):
        print('\n' * 2)
        if not self.error_rate or not self.error_rate_by_row:
            raise Exception('Error rate not calculated. Please run predict with validation = True, then try again.')
        # w = 20
        for category in 'targets confounders'.split():
            xs = [x for x in self.targets if not x in self.confounders] if category == 'targets' else self.confounders
            print()
            print(f'{category.upper():<{w}}{"OVERALL":<{w}}{"FALSE POSITIVE":<{w}}{"FALSE NEGATIVE":<{w}}{"THRESHOLD":<{w}}')
            print('-' * (w * 5))
            for target in xs:
                print(f'{target:<{w}}{self.error_rate["overall"][target]:<{w}}{self.error_rate["false_positive"][target]:<{w}}{self.error_rate["false_negative"][target]:<{w}}{self.cutoffs[target]:<{w}}')


    def error_by_row(self):
        # TODO
        pass


    def examples(self, targets = None, n = 10, w = 10, margin = 5):
        targets = targets or self.targets
        sim_w = self.n_decimals + margin
        print('\n' * 2)

        ones = dict()
        zeros = dict()
        for t in targets:
            ones[t] = sorted([(i,v, self.prediction_data[i][self.text_column]) for i,v in enumerate(self.similarities[t]) if v >= self.cutoffs[t]], key = lambda x: x[1], reverse = True)
            zeros[t] = sorted([(i,v, self.prediction_data[i][self.text_column]) for i,v in enumerate(self.similarities[t]) if v < self.cutoffs[t]], key = lambda x: x[1], reverse = True)
            _n = min(n, len(ones[t]) // 2, len(zeros[t]) // 2)

            print(f'{t}, threshold = {self.cutoffs[t]}')
            print(f'{"TOP ONES (best matches)":<{w * 3 + sim_w}}{"BOTTOM ONES (look for false positives)":<{w * 3 + sim_w}}{"TOP ZEROS (look for false negatives)":<{w * 3 + sim_w}}')
            print(f'{"Text":<{w * 3}}{"Sim.":<{sim_w}}' * 3)
            print('-' * 3 * (3 * w + sim_w))
            m = len(ones[t]) - 1
            for i in range(_n):
                print(f'{ones[t][i][2][:(w * 3) - margin] + ("..." if len(ones[t][i][2]) > (w * 3) - margin else ""):<{(w * 3)}}{ones[t][i][1]:<{sim_w},.{self.n_decimals}f}{ones[t][m - _n + i][2][:(w * 3) - margin] + ("..." if len(ones[t][m-n+i][2]) > (w * 3) - margin else ""):<{(w * 3)}}{ones[t][m-n+i][1]:<{sim_w},.{self.n_decimals}f}{zeros[t][i][2][:(w * 3) - margin] + ("..." if len(zeros[t][i][2]) > (w * 3) - margin else ""):<{(w * 3)}}{zeros[t][i][1]:<{sim_w},.{self.n_decimals}f}')
            print('\n' * 2)



def tune(training_data, prediction_data, parameter = None, param_name = None, param_range = None, criteria = None, default_cutoff = None, default_selection = None, plot = False, return_errors = False, show_progress = True, **kwargs):
    if kwargs['mutually_exclusive']: 
        raise Exception(f'Argument mutually_exclusive needs to be "False". There is no per-variable parameter to optimize for mutually exclusive variables, as the assigned category is simply the one with the maximum similarity. However, if the right answers are available you can "validate" mutually exclusive data.')
    if not parameter and not param_name:
        raise Exception('You need to supply a parameter (dict: name, start, stop, step) OR param_name (str: similarity or selection) and param_range (tuple: (start, stop, step))')

    # For shorthand, use parameter = 'similarity' and range = (0, 1, 0.01)
    if param_name and param_range:
        start = param_range[0]
        stop = param_range[1] if len(param_range) == 2 else 1
        step = param_range[2] if len(param_range) == 3 else 0.01
        parameter = dict(name = param_name, start = start, stop = stop, step = step)

    # C = Classifier(n_decimals = kwargs['n_decimals'] if 'n_decimals' in kwargs else 5, **kwargs)
    C = Classifier(n_decimals = kwargs.get('n_decimals') or 5, **kwargs)
    C.default_cutoff = default_cutoff or C.default_cutoff
    C.default_selection = default_selection or C.default_selection
    C.train(training_data = training_data)
    targets = C.targets.copy()
    criteria = criteria or dict()
    _criteria = { target: {} for target in targets }

    parameter['start'] = parameter['start'] if parameter['start'] > 0 else 0.01
    parameter_name = parameter['name']
    parameter_multiplier = 1 / parameter['step'] 
    parameter_range = [x / parameter_multiplier for x in range(round(parameter['start'] * parameter_multiplier),
                                           round((parameter['stop'] + parameter['step']) * parameter_multiplier),
                                           round(parameter['step'] * parameter_multiplier))]
    n_iterations = len(parameter_range)
    overall_errors = { key: list() for key in targets }
    fp = { key: list() for key in targets }
    fn = { key: list() for key in targets }
    previous_loss = { key: 1 for key in targets }
    optimal_values = { key: { parameter_name: None } for key in targets }
    min_errors = { key: None for key in targets }
    lower_end = list()
    upper_end = list()
    _ = [*C.predict(prediction_data)]

    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    start_time = time.time()
    icon = cycle(['  |   ', '  _   ', '  __  ', '   __ ', '    __', '     _', '     |', 
                  '     \u203e', '    \u203e\u203e', '   \u203e\u203e ', '  \u203e\u203e  ', '  \u203e   '])
    for i,iteration in enumerate(parameter_range):
        if i > 0:
            current_time = time.time()
            time_elapsed = current_time - start_time
            time_left = time_elapsed * ((n_iterations - i) / i)
            if show_progress:
                print(LINE_UP, end=LINE_CLEAR)
                print(f'{next(icon)} Iteration {i + 1} of {n_iterations} estimating {parameter_name}, time left: {round(time_left)} seconds', end='\r', flush = True)
        for target in C.targets: 
            if target in criteria:
                _criteria[target] = {**criteria[target], 
                                     **{ parameter_name: iteration}}
            else:
                _criteria[target] = { parameter_name: iteration }
        C.criteria = _criteria
        C.assemble_filter()
        _ = [*C.predict(validation = True)]
        for target in C.targets.copy():
            overall_errors[target].append(C.error_rate['overall'][target])
            fp[target].append(C.error_rate['false_positive'][target])
            fn[target].append(C.error_rate['false_negative'][target])
            if not plot:
                if parameter_name == 'similarity':
                    current_loss = abs(C.error_rate['false_positive'][target] - C.error_rate['false_negative'][target])
                elif parameter_name == 'selection':
                    current_loss = overall_errors[target][-1]
                worsening = current_loss > previous_loss[target]
                last_iteration = i == (n_iterations - 1)
                if last_iteration or worsening:
                    # If optimal input value found, stop trying to optimize the target at hand
                    # If last iteration and no improvement, return the last value
                    index = i-1 if worsening else i
                    if last_iteration and (not worsening):
                        upper_end.append(target)
                    optimal_values[target] = { parameter_name: parameter_range[index] }
                    if i == 1 and n_iterations > 2:
                        lower_end.append(target)
                    C.remove_target(target)
                if len(C.targets) == 0:
                    for t in targets:
                        if t in criteria:
                            optimal_values[t] = {**criteria[t], 
                                                 **optimal_values[t]}
                    if upper_end:
                        print()
                        warnings.warn(f'Optimal {parameter_name} was found to be the very last value for the following targets: {", ".join(lower_end)}. Consider extending the lower end of the range. ', stacklevel = 2)
                        print()
                    if not last_iteration:
                        print()
                        print('returning early')
                        print()
                    if lower_end:
                        print()
                        warnings.warn(f'Optimal {parameter_name} was found to be the very first value for the following targets: {", ".join(lower_end)}. Consider extending the lower end of the range. ', stacklevel = 2)
                        print()
                    return optimal_values
                previous_loss[target] = current_loss
        print()
    if plot: 
        try:
            import plotext as plt
        except ImportError as s:
            raise Exception('Optional dependency plotext (>=5.3.2) needed for plotting. Otherwise set plot = False.')
        for target in C.targets:
            if parameter_name == 'similarity':
                loss = [abs(a-b) for a,b in zip(fp[target], fn[target])]
            elif parameter_name == 'selection':
                loss = overall_errors[target]
            min_errors[target] = min(loss)
            min_error_index = loss.index(min_errors[target])
            optimal_values[target] = { parameter_name: parameter_range[min_error_index] }

            # plt.vline(min_error_index + 1, 'black')
            plt.vline(min_error_index, 'black')
            plt.plot(overall_errors[target], label = f'{target} overall error')
            plt.plot(fp[target], label = f'{target} false positive')
            plt.plot(fn[target], label = f'{target} false negative')
            plt.xticks(list(range(n_iterations)), parameter_range)
            plt.title(f'{target} validation error by {parameter_name}')
            plt.build()
            cwd = os.getcwd()
            plotpath = os.path.join(cwd, 'plots')
            if not os.path.exists(plotpath):
                os.mkdir(plotpath)
            plt.save_fig(f'{plotpath}/{target}.html')
            plt.clear_figure()

    for t in targets:
        if t in criteria:
            optimal_values[t] = {**criteria[t], 
                                 **optimal_values[t]}

    if return_errors:
        return optimal_values, min_errors, errors
    else:
        return optimal_values



def prediction_pipeline(pretrained_classifier, n = 100, split = True, timeout = 10, **kwargs):

    text_column = kwargs.get('text_column') or 'text'
    aggregation = kwargs.get('aggregation')
    if not getattr(pretrained_classifier, 'discrete'):
        if not aggregation or aggregation in "absolute sum relative share".split():
            setattr(pretrained_classifier, 'discrete', True)
            print()
            print(f'Classifier set to produce discrete values, given that argument aggregation = "{aggregation}".')
            print()
            # TODO: Re-use this to raise an exception in combine_records()

    # Make sure our classifier is using the latest text column for prediction
    if 'text_column' in kwargs: 
        setattr(pretrained_classifier, 'text_column', kwargs['text_column']) 

    predict_args = "pre_encoded validation".split()
    predict_args = { k:v for k,v in kwargs.items() if k in predict_args }

    split_args = "text_column max_sequence_length doc_id_column row_id_column chunk_id_column overlap per_sentence".split()
    split_args = { k:v for k,v in kwargs.items() if k in split_args }

    combine_args = "text_separator text_column doc_id_column row_id_column chunk_id_column original_data binary_targets aggregation overlap".split()
    combine_args = { k:v for k,v in kwargs.items() if k in combine_args }

    from types import GeneratorType

    def inner_function(prediction_data):
        if isinstance(prediction_data, (list, tuple)):
            prediction_data = iter(prediction_data)
            prediction_data = prediction_data.__iter__()

        if isinstance(prediction_data, GeneratorType):

            prediction_data = [*prediction_data] # Need to unpack, no way to copy a generator
            original_data = prediction_data.copy()
            prediction_data = split_records(prediction_data, **split_args)
            predicted = pretrained_classifier.predict(prediction_data = prediction_data, **predict_args)
            if split:
                predicted = combine_records(predicted, original_data = original_data, binary_targets = pretrained_classifier.targets, **combine_args)
            yield from predicted
        else:
            raise Exception(f'Unknown/incompatible data type: {type(prediction_data)}')

    return inner_function


