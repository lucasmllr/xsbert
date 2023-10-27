from flair.data import Sentence
from flair.models import SequenceTagger
from typing import Tuple, Optional, List
import numpy as np

from models import XSTransformer


pos_tag_mapping = {
    'ADD': 'other',  # Email
    'AFX': 'other',  # Affix
    'CC': 'CC',  # Coordinating conjunction
    'CD': 'other',  # Cardinal number
    'DT': 'DT',  # Determiner
    'EX': 'other',  # Existential there
    'FW': 'other',  # Foreign word
    'HYPH': 'PT',  # Hyphen
    'IN': 'IN',  # Preposition or subordinating conjunction
    'JJ': 'JJ',  # Adjective
    'JJR': 'JJ',  # Adjective, comparative
    'JJS': 'JJ',  # Adjective, superlative
    'LS': 'PC',  # List item marker
    'MD': 'MD',  # Modal
    'NFP': 'PC',  # Superfluous punctuation
    'NN': 'NN',  # Noun, singular or mass
    'NNP': 'NN',  # Proper noun, singular
    'NNPS': 'NN',   # Proper noun, plural
    'NNS': 'NN',  # Noun, plural
    'PDT': 'DT',  # Predeterminer
    'POS': 'other',  # Possessive ending
    'PRP': 'PR',  # Personal pronoun
    'PRP$': 'PR',  # Possessive pronoun
    'RB': 'RB',  # Adverb
    'RBR': 'RB',  # Adverb, comparative
    'RBS': 'RB',  # Adverb, superlative
    'RP': 'other',  # Particle
    'SYM': 'other',  # Symbol
    'TO': 'other',  # to
    'UH': 'other',  # Interjection (often Particle)
    'VB': 'VB',  # Verb, base form
    'VBD': 'VB',  # Verb, past tense
    'VBG': 'VB',  # Verb, gerund or present participle
    'VBN': 'VB',  # Verb, past participle
    'VBP': 'VB',  # Verb, non-3rd person singular present
    'VBZ': 'VB',  # Verb, 3rd person singular present
    'WDT': 'DT',  # Wh-determiner
    'WP': 'PR',  # Wh-pronoun
    'WP$': 'PR',  # Possessive wh-pronoun
    'WRB': 'RB',  # Wh-adverb
    'XX': 'XX',  # Unknown
    '.': 'PT',
    ',': 'PT',
    '?': 'PT',
    '!': 'PT',
    ':': 'PT',
    ';': 'PT'
}


class Instance:

    def __init__(self, texts: Tuple[str], label: Optional[float] = None):

        self.texts = self._prep_texts(texts)
        self.label = label

        self.tokens = None
        self.attributions = None
        self.score = None

        self.words = None
        self.word_attributions = None

    def _prep_texts(self, texts: Tuple[str]):
        return_texts = []
        for text in texts:
            text = text.replace("n't", " not")
            return_texts.append(text)
        return tuple(return_texts)

    def tag(self, tagger: SequenceTagger, tag_mapping: Optional[dict] = pos_tag_mapping):
        tags = []
        for text in self.texts:
            s = Sentence(text)
            tagger.predict(s)
            results = []
            for t in s.tokens:
                tag = t.tag
                if tag_mapping is not None:
                    if tag in tag_mapping.keys():
                        tag = tag_mapping[tag]
                    else:
                        tag = 'other'
                results.append({'word': t.text, 'tag': tag})
            tags.append(results)
        self.words = tags

    def attribute_similiarity(self, model: XSTransformer):
        A, ta, tb, s = model.explain_similarity(
            text_a=self.texts[0],
            text_b=self.texts[1],
            verbose=False
        )
        self.attributions = A
        self.tokens = (ta, tb)
        self.score = s

    def match_tokens_to_words(self):
        assert self.words is not None
        assert self.tokens is not None
        for idx in range(2):
            words = iter(self.words[idx])
            current_word = next(words)
            next_word = next(words)
            current_tokens = []
            for i, t in enumerate(self.tokens[idx]):
                if t == 'CLS': 
                    continue
                if next_word['word'].startswith(t) or t == 'EOS':
                    current_word['tokens'] = current_tokens
                    current_tokens = [i]
                    current_word = next_word
                    try:
                        next_word = next(words)
                    except StopIteration:
                        continue
                else:
                    current_tokens.append(i)

    def make_word_attributions(self):
        word_attr = []
        for w0 in self.words[0]:
            if 'tokens' not in w0.keys():
                raise KeyError(f'Token matching error with sentence: {self.texts[0]}')
            t0 = w0['tokens']
            col = []
            for w1 in self.words[1]:
                if 'tokens' not in w1.keys():
                    raise KeyError(f'Token matching error with sentence: {self.texts[1]}')
                t1 = w1['tokens']
                a = self.attributions[t0[0]:t0[-1]+1, t1[0]:t1[-1]+1]
                col.append(a.mean())
            word_attr.append(col)
        self.word_attributions = np.array(word_attr)

    def get_sorted_word_attributions(self, top_percentile: Optional[float] = None):
        assert self.word_attributions is not None
        idxs = list(np.transpose(np.array(
            np.unravel_index(np.argsort(self.word_attributions, axis=None), self.word_attributions.shape)))
            )[::-1]
        pairs = []
        N = int(len(idxs) * top_percentile) if top_percentile is not None else len(idxs)
        for n, (i, j) in enumerate(idxs):
            a = self.word_attributions[i, j]
            w0 = self.words[0][i]
            w1 = self.words[1][j]
            pairs.append((a, w0, w1))
            if n >= N:
                break
        return pairs
    
    def get_filtered_word_attributions(self, types: List, return_cumul_score: bool = False):
        all_pairs = self.get_sorted_word_attributions()
        types = [t.split('-') for t in types]
        filtered_pairs = []
        for pair in all_pairs:
            for type in types:
                if pair[1]['tag'] == type[0] and pair[2]['tag'] == type[1] \
                    or pair[1]['tag'] == type[1] and pair[2]['tag'] == type[0]:
                    filtered_pairs.append(pair)
        if return_cumul_score:
            cscore = sum([p[0] for p in filtered_pairs])
            return filtered_pairs, cscore, self.score
        else:
            return filtered_pairs