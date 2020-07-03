# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:51:35 2019

@author: ns
"""
import re
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from string import punctuation
from nltk.stem import SnowballStemmer
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from warnings import warn
from sklearn.preprocessing import normalize
from pyspark import SparkContext

input_document_path = ''
output_document_path = ''
stopwords_filepath = []

class Tokenizer():

    def __init__(self, lang='english', stem=True, s_words=None):
        if stem:
            self._stem_obj = SnowballStemmer(lang)
        else:
            self._stem_obj = None

        if isinstance(s_words, list):
            self._s_words = s_words
        else:
            self._s_words = self.do_load_stopwords(s_words)

    @staticmethod
    def do_load_stopwords(fpath):
        try:
            fh = sc.binaryFiles(fpath).values().flatMap(lambda x: x.decode("utf-8").splitlines())
            stopwords_list = fh.collect()
            stopwords = [stopword.strip('\n') for stopword in stopwords_list]
        except IOError:
            stopwords = []
        return stopwords

    @property
    def set_stemmer(self):
        return self._stem_obj

    def rem_s_words(self, tok_list):
        if isinstance(tok_list, (list, tuple)):
            return [tok_instance for tok_instance in tok_list if tok_instance.lower() not in self._s_words]
        else:
            return ' '.join(
                [tok_instance for tok_instance in tok_list.split(' ') if tok_instance.lower() not in self._s_words]
            )

    def do_stem(self, word):
        if self.set_stemmer:
            return utoa(self._stem_obj.stem(word))
        else:
            return word

    def stem_tok(self, tok):
        return [self.do_stem(word) for word in tok]

    @staticmethod
    def do_punc_strip(text, exclude='', include=''):
        punc_charstostrip = ''.join(
            set(list(punctuation)).union(set(list(include))) - set(list(exclude))
        )
        return text.strip(punc_charstostrip)

    @staticmethod
    def remove_allpunctuation(text):
        return ''.join([char for char in text if char not in punctuation])

    def tok_words(self, text):
        return [
            self.do_punc_strip(word) for word in text.split(' ')
            if self.do_punc_strip(word)
        ]

    def text_sanitization(self, text):
        tokens = self.tok_words(text.lower())
        tokens = self.rem_s_words(tokens)
        tokens = self.stem_tok(tokens)
        sanitized_text = ' '.join(tokens)
        return sanitized_text

    @staticmethod
    def _rem_wh_space(text):
        notaspace = re.finditer(r'[^ ]', text)

        if not notaspace:
            return text

        f_notaspace = notaspace.__next__()
        f_notaspace = f_notaspace.start()

        l_notaspace = None
        for item in notaspace:
            l_notaspace = item

        if not l_notaspace:
            return text[f_notaspace:]
        else:
            l_notaspace = l_notaspace.end()
            return text[f_notaspace:l_notaspace]

    def tok_sentences(self, text, word_threshold=5):
        params_punkt = PunktParameters()
        params_punkt.abbrev_types = set([
            'vs', 'dr', 'mrs', 'mr', 'prof',
            'ms', 'inc', 'mt', 'e.g', 'i.e'
        ])
        punc_sen_tok = PunktSentenceTokenizer(params_punkt)

        not_processedtxt = text.replace('!"', '! "').replace('?"', '? "').replace('."', '. "')
        not_processedtxt = not_processedtxt.replace('\n', ' . ')
        notprocessed_sentences = punc_sen_tok.tokenize(not_processedtxt)

        for num, np_sentence in enumerate(notprocessed_sentences):
            np_sentence = utoa(np_sentence)
            np_sentence = np_sentence.replace('! " ', '!" ').replace('? " ', '?" ').replace('. " ', '." ')
            np_sentence = self._rem_wh_space(np_sentence)
            np_sentence = np_sentence[:-2] if (
                        np_sentence.endswith(' . ') or np_sentence.endswith(' .')) else np_sentence
            notprocessed_sentences[num] = np_sentence

        sentences_processed = [self.text_sanitization(sen) for sen in notprocessed_sentences]
        texts_filtered = [i for i in range(len(sentences_processed)) if
                          len(sentences_processed[i].replace('.', '').split(' ')) > word_threshold]
        sentences_processed = [sentences_processed[sentence_num] for sentence_num in texts_filtered]
        notprocessed_sentences = [notprocessed_sentences[sentence_num] for sentence_num in texts_filtered]

        return sentences_processed, notprocessed_sentences


class LSA():

    def __init__(self, t_nizer=Tokenizer('english', True, stopwords_filepath)):
        self._tnizer = t_nizer

    def _singular_vec_decomp(cls, matrix, n_concepts=5):
        uni_left, singular, vt_right = svds(matrix, k=n_concepts)
        return uni_left, singular, vt_right

    def do_validate_numberoftopics(cls, topics, sentences):
        sentences_collection = set([frozenset(sentence.split(' ')) for sentence in sentences])
        mat_rank_estimated = len(sentences_collection)

        if 1 >= mat_rank_estimated:
            raise SingularValueDecompositionRankException('there is no rank to compute SVD')

        if (mat_rank_estimated - 1) < topics:
            warn(
                'Number of Topics should be less than rank to avoid SVD comutation problem ',
                Warning
            )
            topics = mat_rank_estimated - 1
        return topics

    def do_LSAsummarize(self, text, length=5, topics=4, sigma_threshold_for_topic=0.5, bi_matrix=True):

        text = self.do_parseinput(text)

        sentences, notprocessed_sentences = self._tnizer.tok_sentences(text)

        length = self.do_parse_summarylength(length, len(sentences))
        if len(sentences) == length:
            return notprocessed_sentences

        topics = self.do_validate_numberoftopics(topics, sentences)

        weighing = 'b' if bi_matrix else 'fr'
        sent_matrix = self.do_matrix_computation(sentences, weighing=weighing)
        sent_matrix = sent_matrix.transpose()

        sent_matrix = sent_matrix.multiply(sent_matrix > 0)

        s, u, v = self._singular_vec_decomp(sent_matrix, n_concepts=topics)

        if 0 > sigma_threshold_for_topic >= 1:
            raise ValueError('Topic Sigma threshold should be b/w 0 and 1')

        sig_th_value = max(u) * sigma_threshold_for_topic
        u[u < sig_th_value] = 0

        s_vec = np.dot(np.square(u), np.square(v))

        t_sentences = s_vec.argsort()[-length:][::-1]
        t_sentences.sort()

        return [notprocessed_sentences[i] for i in t_sentences]

    def do_matrix_computation(cls, sentences, norm_value=None, weighing='frequency'):

        if norm_value not in (None, 'l1', 'l2'):
            raise ValueError('Normalization should be L1 or L2')

        if weighing.lower() == 'tfidf':
            vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1, stop_words=None)
        elif weighing.lower() == 'fr':
            vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=1, stop_words=None, binary=False)
        elif weighing.lower() == 'b':
            vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=1, stop_words=None, binary=True)
        else:
            raise ValueError('Allowed values TFIDF or BINARY or FREQUENCY')

        fr_matrix = vectorizer.fit_transform(sentences).astype(float)

        if norm_value in ('l1', 'l2'):
            fr_matrix = normalize(fr_matrix, norm=norm_value, axis=1)
        elif norm_value is not None:
            raise ValueError('Normalization should be L1 or L2')

        return fr_matrix

    def do_parseinput(cls, text):
        return parse_input_text(text)

    def do_parse_summarylength(cls, n_sentences, length):
        if 0 > length or not isinstance(length, (int, float)):
            raise ValueError('Summary Length should be a positive number')
        elif 1 > length > 0:
            return int(round(length * n_sentences))
        elif length >= n_sentences:
            return n_sentences
        else:
            return int(length)


class SingularValueDecompositionRankException(Exception):
    pass


def utoa(unicodestr):
    if isinstance(unicodestr, str):
        return unicodestr
    else:
        raise ValueError('I/P text should be String !')

def parse_input_text(text):
    if isinstance(text, str):
        if text.endswith('.txt'):
            textfile = open(text, 'rb')
            document = textfile.read()
            textfile.close()
            return utoa(document)
        else:
            return utoa(text)
    else:
        raise ValueError('I/P text should be String !')


txt = ''
fp = sc.binaryFiles(input_document_path).values().flatMap(lambda x: x.decode("utf-8").splitlines())
list_text = fp.collect()
for line in list_text:
    txt += line
print('\nLSA Summary:\n')

lsa_ref = LSA()
summary = lsa_ref.do_LSAsummarize(txt, length=3)
result = []
for sentence in summary:
    print(sentence)
    result.append(str(sentence))
sc.parallelize(result).coalesce(1).saveAsTextFile(output_document_path)
