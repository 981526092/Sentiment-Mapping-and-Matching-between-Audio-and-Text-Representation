import numpy as np
import re
import itertools
from collections import Counter

class DataPrepare:
    def __init__(self) -> None:
        pass

    def clean_str(self, string):
        """
        @Description: Tokenization/string cleaning for datasets.
        @Return: string.strip().lower()
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def load_data_and_labels(self):
        """
        @Description: Loads and preprocessed data for the dataset.
        @Return: input vectors, labels, vocabulary, and inverse vocabulary
        """
        # Load data from files
        positive_examples = list(open("./data/rt-polarity.pos", "r", encoding='latin-1').readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open("./data/rt-polarity.neg", "r", encoding='latin-1').readlines())
        negative_examples = [s.strip() for s in negative_examples]
        # Split by words
        x_text = positive_examples + negative_examples
        x_text = [self.clean_str(sent) for sent in x_text]
        x_text = [s.split(" ") for s in x_text]
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)
        return [x_text, y]

    def pad_sentences(self, sentences, padding_word="<PAD/>", sequence_length=56):
        """
        @Description: Pads all sentences to the same length.
        @Return: padded_sentences
        """
        padded_sentences = []
        for i in range(len(sentences)):
            sentence = sentences[i]
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
        return padded_sentences

    def build_vocab(self, sentences):
        """
        @Description: Builds a vocabulary mapping from word to index based on the sentences.
        @Return: vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = Counter(itertools.chain(*sentences))
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv = list(sorted(vocabulary_inv))
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]


    def build_input_data(self, sentences, labels, vocabulary):
        """
        @Description: Maps sentences and labels to vectors based on a vocabulary.
        """
        x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
        y = np.array(labels)
        return [x, y]


    def load_data(self):
        """
        @Description: Loads and preprocessed data for the dataset.
        @Return: input vectors, labels, vocabulary, and inverse vocabulary
        """
        # Load and preprocess data
        sentences, labels = self.load_data_and_labels()
        sequence_length = max(len(x) for x in sentences)
        #If it can't be divisible by 8, you need to 
        # change to the bigger nearest number that can be divisible by 8
        if (sequence_length%8 != 0):
            sequence_length = (sequence_length/8 + 1) * 8
        sentences_padded = self.pad_sentences(sentences, sequence_length)
        vocabulary, vocabulary_inv = self.build_vocab(sentences_padded)
        x, y = self.build_input_data(sentences_padded, labels, vocabulary)
        return [x, y, vocabulary, vocabulary_inv]


