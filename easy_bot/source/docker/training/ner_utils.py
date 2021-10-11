# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Data utilities for the named entity recognition task."""

import logging
from collections import namedtuple

import numpy as np
import mxnet as mx
import gluonnlp as nlp
from mxnet.gluon import Block, nn, HybridBlock
from contextlib import ExitStack
import traceback
import random
random.seed(111)

TaggedToken = namedtuple('TaggedToken', ['text', 'tag'])
PredictedToken = namedtuple('PredictedToken', ['text', 'true_tag', 'pred_tag'])

NULL_TAG = 'X'

def bio_bioes(tokens):
    """Convert a list of TaggedTokens in BIO(2) scheme to BIOES scheme.

    Parameters
    ----------
    tokens: List[TaggedToken]
        A list of tokens in BIO(2) scheme

    Returns
    -------
    List[TaggedToken]:
        A list of tokens in BIOES scheme
    """
    ret = []
    for index, token in enumerate(tokens):
        if token.tag == 'O':
            ret.append(token)
        elif token.tag.startswith('B'):
            # if a B-tag is continued by other tokens with the same entity,
            # then it is still a B-tag
            if index + 1 < len(tokens) and tokens[index + 1].tag.startswith('I'):
                ret.append(token)
            else:
                ret.append(TaggedToken(text=token.text, tag='S' + token.tag[1:]))
        elif token.tag.startswith('I'):
            # if an I-tag is continued by other tokens with the same entity,
            # then it is still an I-tag
            if index + 1 < len(tokens) and tokens[index + 1].tag.startswith('I'):
                ret.append(token)
            else:
                ret.append(TaggedToken(text=token.text, tag='E' + token.tag[1:]))
    return ret


def read_bio_as_bio2(data_path):
    """Read CoNLL-formatted text file in BIO scheme in given path as sentences in BIO2 scheme.

    Parameters
    ----------
    data_path: str
        Path of the data file to read

    Returns
    -------
    List[List[TaggedToken]]:
        List of sentences, each of which is a List of TaggedTokens
    """

    with open(data_path, 'r') as ifp:
        sentence_list = []
        current_sentence = []
        prev_tag = 'O'

        for line in ifp:
            if len(line.strip()) > 0:
                try:
                    line_items = line.rstrip().split()
                    if len(line_items) == 4:
                        word, _, _, tag = line_items
                except Exception as e:
                    traceback.print_exc()
                    print(f"error: got '{str(e)}' given line='{line}'", flush=True)

                # convert BIO tag to BIO2 tag
                if tag == 'O':
                    bio2_tag = 'O'
                else:
                    if prev_tag == 'O' or tag[2:] != prev_tag[2:]:
                        bio2_tag = 'B' + tag[1:]
                    else:
                        bio2_tag = tag
                current_sentence.append(TaggedToken(text=word, tag=bio2_tag))
                prev_tag = tag
            else:
                # the sentence was completed if an empty line occurred; flush the current sentence.
                sentence_list.append(current_sentence)
                current_sentence = []
                prev_tag = 'O'

        # check if there is a remaining token. in most CoNLL data files, this does not happen.
        if len(current_sentence) > 0:
            sentence_list.append(current_sentence)
        return sentence_list


def remove_docstart_sentence(sentences):
    """Remove -DOCSTART- sentences in the list of sentences.

    Parameters
    ----------
    sentences: List[List[TaggedToken]]
        List of sentences, each of which is a List of TaggedTokens.
        This list may contain DOCSTART sentences.

    Returns
    -------
        List of sentences, each of which is a List of TaggedTokens.
        This list does not contain DOCSTART sentences.
    """
    ret = []
    for sentence in sentences:
        current_sentence = []
        for token in sentence:
            if token.text != '-DOCSTART-':
                current_sentence.append(token)
        if len(current_sentence) > 0:
            ret.append(current_sentence)
    return ret


def bert_tokenize_sentence(sentence, bert_tokenizer):
    """Apply BERT tokenizer on a tagged sentence to break words into sub-words.
    This function assumes input tags are following IOBES, and outputs IOBES tags.

    Parameters
    ----------
    sentence: List[TaggedToken]
        List of tagged words
    bert_tokenizer: nlp.data.BertTokenizer
        BERT tokenizer

    Returns
    -------
    List[TaggedToken]: list of annotated sub-word tokens
    """
    ret = []
    for token in sentence:
        # break a word into sub-word tokens
        sub_token_texts = bert_tokenizer(token.text)
        try:
            # only the first token of a word is going to be tagged
            ret.append(TaggedToken(text=sub_token_texts[0], tag=token.tag))
        except Exception:
            # print(len(sub_token_texts), sub_token_texts, token.tag)
            pass
        ret += [TaggedToken(text=sub_token_text, tag=NULL_TAG)
                for sub_token_text in sub_token_texts[1:]]

    return ret


def load_segment(file_path, bert_tokenizer):
    """Load CoNLL format NER datafile with BIO-scheme tags.

    Tagging scheme is converted into BIOES, and words are tokenized into wordpieces
    using `bert_tokenizer`.

    Parameters
    ----------
    file_path: str
        Path of the file
    bert_tokenizer: nlp.data.BERTTokenizer

    Returns
    -------
    List[List[TaggedToken]]: List of sentences, each of which is the list of `TaggedToken`s.
    """
    logging.info('Loading sentences in %s...', file_path)
    bio2_sentences = remove_docstart_sentence(read_bio_as_bio2(file_path))
    bioes_sentences = [bio_bioes(sentence) for sentence in bio2_sentences]
    subword_sentences = [bert_tokenize_sentence(sentence, bert_tokenizer)
                         for sentence in bioes_sentences]

    logging.info('load %s, its max seq len: %d',
                 file_path, max(len(sentence) for sentence in subword_sentences))

    return subword_sentences


class BERTTaggingDataset:
    """

    Parameters
    ----------
    text_vocab: gluon.nlp.Vocab
        Vocabulary of text tokens/
    train_path: Optional[str]
        Path of the file to locate training data.
    dev_path: Optional[str]
        Path of the file to locate development data.
    test_path: Optional[str]
        Path of the file to locate test data.
    seq_len: int
        Length of the input sequence to BERT.
    is_cased: bool
        Whether to use cased model.
    """

    def __init__(self, text_vocab, train_path, dev_path, seq_len, is_cased,
                 tag_vocab=None):
        self.text_vocab = text_vocab
        self.seq_len = seq_len

        self.bert_tokenizer = nlp.data.BERTTokenizer(vocab=text_vocab, lower=not is_cased)

        train_sentences = [] if train_path is None else load_segment(train_path,
                                                                     self.bert_tokenizer)
        dev_sentences = [] if dev_path is None else load_segment(dev_path, self.bert_tokenizer)
        all_sentences = train_sentences + dev_sentences

        print("===== BEGIN TOKENIZED SAMPLES =====")
        print(' '.join([tokens.text for tokens in train_sentences[0]]))
        print(' '.join([tokens.tag for tokens in train_sentences[0]]))
        print()
        print(' '.join([tokens.text for tokens in train_sentences[1]]))
        print(' '.join([tokens.tag for tokens in train_sentences[1]]))
        print()
        print(' '.join([tokens.text for tokens in train_sentences[2]]))
        print(' '.join([tokens.tag for tokens in train_sentences[2]]))
        print("=====  END  TOKENIZED SAMPLES =====")

        if tag_vocab is None:
            logging.info('Indexing tags...')
            tag_counter = nlp.data.count_tokens(token.tag
                                                for sentence in all_sentences for token in sentence)
            self.tag_vocab = nlp.Vocab(tag_counter, padding_token=NULL_TAG,
                                       bos_token=None, eos_token=None, unknown_token=None)
            print([token.tag for token in all_sentences[0]])
            print(self.tag_vocab.idx_to_token)
        else:
            self.tag_vocab = tag_vocab
        self.null_tag_index = self.tag_vocab[NULL_TAG]

        self.train_inputs = [self._encode_as_input(sentence) for sentence in train_sentences]
        self.dev_inputs = [self._encode_as_input(sentence) for sentence in dev_sentences]

        print("===== BEGIN TOKENIZED SAMPLES =====")
        print(self.train_inputs[0][0])
        print(self.train_inputs[0][3])
        print()
        print(self.train_inputs[1][0])
        print(self.train_inputs[1][3])
        print()
        print(self.train_inputs[2][0])
        print(self.train_inputs[2][3])
        print("=====  END  TOKENIZED SAMPLES =====")

        logging.info('tag_vocab: %s', self.tag_vocab)
        print("train inputs: {}, dev inputs: {}". \
              format(len(self.train_inputs), len(self.dev_inputs)))
        self.dev_sentences = dev_sentences

    def _encode_as_input(self, sentence):
        """Enocde a single sentence into numpy arrays as input to the BERTTagger model.

        Parameters
        ----------
        sentence: List[TaggedToken]
            A sentence as a list of tagged tokens.

        Returns
        -------
        np.array: token text ids (batch_size, seq_len)
        np.array: token types (batch_size, seq_len),
                which is all zero because we have only one sentence for tagging.
        np.array: valid_length (batch_size,) the number of tokens until [SEP] token
        np.array: tag_ids (batch_size, seq_len)
        np.array: flag_nonnull_tag (batch_size, seq_len),
                which is simply tag_ids != self.null_tag_index

        """
        # check whether the given sequence can be fit into `seq_len`.
        # print(''.join([t.text[0] for t in sentence]))
        if len(sentence) > self.seq_len - 2:
            sentence = sentence[:self.seq_len-3]
        assert len(sentence) <= self.seq_len - 2, \
            'the number of tokens {} should not be larger than {} - 2. offending sentence: {}' \
            .format(len(sentence), self.seq_len, ''.join([t.text for t in sentence]))

        text_tokens = ([self.text_vocab.cls_token] + [token.text for token in sentence] +
                       [self.text_vocab.sep_token])
        padded_text_ids = (self.text_vocab.to_indices(text_tokens)
                           + ([self.text_vocab[self.text_vocab.padding_token]]
                              * (self.seq_len - len(text_tokens))))

        tags = [NULL_TAG] + [token.tag for token in sentence] + [NULL_TAG]
        padded_tag_ids = (self.tag_vocab.to_indices(tags)
                          + [self.tag_vocab[NULL_TAG]] * (self.seq_len - len(tags)))

        assert len(text_tokens) == len(tags)
        assert len(padded_text_ids) == len(padded_tag_ids)
        assert len(padded_text_ids) == self.seq_len

        valid_length = len(text_tokens)

        # in sequence tagging problems, only one sentence is given
        token_types = [0] * self.seq_len

        np_tag_ids = np.array(padded_tag_ids, dtype='int32')
        # gluon batchify cannot batchify numpy.bool? :(
        flag_nonnull_tag = (np_tag_ids != self.null_tag_index).astype('int32')

        return (np.array(padded_text_ids, dtype='int32'),
                np.array(token_types, dtype='int32'),
                np.array(valid_length, dtype='int32'),
                np_tag_ids,
                flag_nonnull_tag)

    @staticmethod
    def _get_data_loader(inputs, shuffle, batch_size):
        return mx.gluon.data.DataLoader(inputs, batch_size=batch_size, shuffle=shuffle,
                                        num_workers=2, last_batch='keep')

    def get_train_data_loader(self, batch_size):
        return self._get_data_loader(self.train_inputs, shuffle=True, batch_size=batch_size)

    def get_dev_data_loader(self, batch_size):
        return self._get_data_loader(self.dev_inputs, shuffle=False, batch_size=batch_size)

    @property
    def num_tag_types(self):
        """Returns the number of unique tags.

        Returns
        -------
        int: number of tag types.
        """
        return len(self.tag_vocab)


class BERTTagger(HybridBlock):
    """Model for sequence tagging with BERT

    Parameters
    ----------
    bert_model: BERTModel
        Bidirectional encoder with transformer.
    num_tag_types: int
        number of possible tags
    dropout_prob: float
        dropout probability for the last layer
    prefix: str or None
        See document of `mx.gluon.Block`.
    params: ParameterDict or None
        See document of `mx.gluon.Block`.
    """

    def __init__(self, bert_model, num_tag_types, dropout_prob, prefix=None, params=None):
        super(BERTTagger, self).__init__(prefix=prefix, params=params)
        self.bert_model = bert_model
        with self.name_scope():
            self.tag_classifier = nn.Dense(units=num_tag_types, flatten=False)
            self.dropout = nn.Dropout(rate=dropout_prob)

    def hybrid_forward(self, F, token_ids, token_types, valid_length): # pylint: disable=arguments-differ
    # def hybrid_forward(self, F, x): # pylint: disable=arguments-differ
        """Generate an unnormalized score for the tag of each token

        Parameters
        ----------
        token_ids: NDArray, shape (batch_size, seq_length)
            ID of tokens in sentences
            See `input` of `glounnlp.model.BERTModel`
        token_types: NDArray, shape (batch_size, seq_length)
            See `glounnlp.model.BERTModel`
        valid_length: NDArray, shape (batch_size,)
            See `glounnlp.model.BERTModel`

        Returns
        -------
        NDArray, shape (batch_size, seq_length, num_tag_types):
            Unnormalized prediction scores for each tag on each position.
        """
        # token_ids, token_types, valid_length = x
        bert_output = self.dropout(self.bert_model(token_ids, token_types, valid_length))
        output = self.tag_classifier(bert_output)
        return output

def convert_arrays_to_text(text_vocab, tag_vocab,
                           np_text_ids, np_true_tags, np_pred_tags, np_valid_length):
    """Convert numpy array data into text

    Parameters
    ----------
    np_text_ids: token text ids (batch_size, seq_len)
    np_true_tags: tag_ids (batch_size, seq_len)
    np_pred_tags: tag_ids (batch_size, seq_len)
    np.array: valid_length (batch_size,) the number of tokens until [SEP] token

    Returns
    -------
    List[List[PredictedToken]]:

    """
    predictions = []
    for sample_index in range(np_valid_length.shape[0]):
        sample_len = np_valid_length[sample_index]
        entries = []
        for i in range(1, sample_len - 1):
            token_text = text_vocab.idx_to_token[np_text_ids[sample_index, i]]
            true_tag = tag_vocab.idx_to_token[int(np_true_tags[sample_index, i])]
            pred_tag = tag_vocab.idx_to_token[int(np_pred_tags[sample_index, i])]
            # we don't need to predict on NULL tags
            if true_tag == NULL_TAG:
                last_entry = entries[-1]
                entries[-1] = PredictedToken(text=last_entry.text + token_text,
                                             true_tag=last_entry.true_tag,
                                             pred_tag=last_entry.pred_tag)
            else:
                entries.append(PredictedToken(text=token_text,
                                              true_tag=true_tag, pred_tag=pred_tag))

        predictions.append(entries)
    return predictions

def attach_prediction(data_loader, net, ctx, is_train):
    """Attach the prediction from a model to a data loader as the last field.

    Parameters
    ----------
    data_loader: mx.gluon.data.DataLoader
        Input data from `bert_model.BERTTaggingDataset._encode_as_input`.
    net: mx.gluon.Block
        gluon `Block` for making the preciction.
    ctx:
        The context data should be loaded to.
    is_train:
        Whether the forward pass should be made with `mx.autograd.record()`.

    Returns
    -------
        All fields from `bert_model.BERTTaggingDataset._encode_as_input`,
        as well as the prediction of the model.

    """
    for data in data_loader:
        text_ids, token_types, valid_length, tag_ids, flag_nonnull_tag = \
            [x.astype('float32').as_in_context(ctx) for x in data]

        with ExitStack() as stack:
            if is_train:
                stack.enter_context(mx.autograd.record())
            out = net(text_ids, token_types, valid_length)
        yield text_ids, token_types, valid_length, tag_ids, flag_nonnull_tag, out


def bosonnlp_to_bio2(origfile, trainfile, valfile):
    """Transform NER annotation in bosonnlp to bio2 format
    """

    val_ratio = 0.02

    traindata = []
    valdata = []
    with open(origfile, 'rt') as fp:
        lines = fp.readlines()
        random.shuffle(lines)

        val_samples = int(len(lines) * val_ratio)
        val_lines = lines[:val_samples]
        train_lines = lines[val_samples:]

        def transform(line):
            # print(line)

            it = 0
            document = ""
            annotations = []
            while True:
                start = line.find("{{", it)
                end = line.find("}}", start)
                next_start = line.find("{{", end)
                
                if end < 0:
                    break

                # print(start, end)
                labeltext = line[start+2:end]
                loc = labeltext.find(":")
                label = labeltext[:loc]
                text = labeltext[loc+1:]

                # label, text = line[start+2:end].split(":")
                prefix = line[it:start]
                if next_start > 0:
                    suffix = line[end+2:next_start]
                else:
                    suffix = line[end+2:]
                
                tic = len(prefix) + len(document)
                toc = len(prefix) + len(document) + len(text)

                annotations.append([tic, toc, label])
                document += prefix + text + suffix

                it = next_start
            document = document.replace(' ', '_')
            document = document.replace('，', ',')
            document = document.replace('“', '"')
            document = document.replace('”', '"')
            document = document.replace('：', ':')
            document = document.replace('（', '(')
            document = document.replace('）', ')')
            document = document.replace('\t', '_')
            return annotations, document

        documents = []

        def strip_suffix(tag):
            if tag.lower().endswith("name") and len(tag) > len("_name"):
                tag = tag[:-4]
            return tag.upper().strip("-_")

        with open(trainfile, 'w') as fp:
            for idx, line in enumerate(train_lines):
                annotations, document = transform(line)

                # print(annotations)
                # print(document)
                document = document.strip()
                documents.append(document)

                count = 0
                word = ""
                for i, c in enumerate(document):
                    if c.isalpha():
                        if len(document) > i+1:
                            if document[i+1].isalpha():
                                word += c
                                continue
                            else:
                                word += c
                                c = word.lower()
                                word = ""
                    elif c == "_":
                        continue

                    label = "O"
                    for a in annotations:
                        if i >= a[0] and i < a[1]:
                            label = "I-"+strip_suffix(a[2])
                    fp.write("{} X X {}\n".format(c, label))

                    # limit sequence length to 128 - 2
                    if (count % 125) == 124 or c in ["。", "；"]:
                        fp.write("\n")
                        count = 0
                    else:
                        count += 1

                fp.write("\n")

        with open(valfile, 'w') as fp:
            for idx, line in enumerate(val_lines):
                annotations, document = transform(line)

                # print(annotations)
                # print(document)
                document = document.strip()
                documents.append(document)

                count = 0
                word = ""
                for i, c in enumerate(document):
                    if c.isalpha():
                        if len(document) > i+1:
                            if document[i+1].isalpha():
                                word += c
                                continue
                            else:
                                word += c
                                c = word.lower()
                                word = ""
                    elif c == "_":
                        continue

                    label = "O"
                    for a in annotations:
                        if i >= a[0] and i < a[1]:
                            label = "I-"+strip_suffix(a[2])
                    fp.write("{} X X {}\n".format(c, label))

                    # limit sequence length to 128 - 2
                    if (count % 125) == 124 or c in ["。", "；"]:
                        fp.write("\n")
                        count = 0
                    else:
                        count += 1

                fp.write("\n")

        with open("documents.txt", "w") as fp:
            fp.write("\n".join(documents))
