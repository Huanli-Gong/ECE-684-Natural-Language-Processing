import numpy as np
import collections

import nltk
# nltk.download('brown')
# nltk.download('universal_tagset')


tagged_sents = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]

tags = set()
vocab = set()

for sent in tagged_sents:
    for word, tag in sent:
        tags.add(tag)
        vocab.add(word.lower())
vocab.add('UNK')

tag2idx = {tag: idx for idx, tag in enumerate(sorted(tags))}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

word2idx = {word: idx for idx, word in enumerate(sorted(vocab))}
idx2word = {idx: word for word, idx in word2idx.items()}

tag_unigram_counts = collections.Counter()
tag_bigram_counts = collections.Counter()
word_tag_counts = collections.Counter()
initial_tag_counts = collections.Counter()

for sent in tagged_sents:
    prev_tag = None
    for i, (word, tag) in enumerate(sent):
        word = word.lower()
        tag_unigram_counts[tag] += 1
        word_tag_counts[(tag, word)] += 1
        if i == 0:
            initial_tag_counts[tag] += 1
        if prev_tag is not None:
            tag_bigram_counts[(prev_tag, tag)] += 1
        prev_tag = tag

num_sentences = len(tagged_sents)
num_tags = len(tag2idx)
num_words = len(word2idx)

pi = np.zeros(num_tags)
A = np.zeros((num_tags, num_tags))
B = np.zeros((num_tags, num_words))

for tag in tag2idx:
    idx = tag2idx[tag]
    pi[idx] = (initial_tag_counts.get(tag, 0) + 1) / (num_sentences + num_tags)

for prev_tag in tag2idx:
    prev_idx = tag2idx[prev_tag]
    denom = tag_unigram_counts[prev_tag] + num_tags
    for curr_tag in tag2idx:
        curr_idx = tag2idx[curr_tag]
        count = tag_bigram_counts.get((prev_tag, curr_tag), 0)
        A[prev_idx, curr_idx] = (count + 1) / denom

for tag in tag2idx:
    tag_idx = tag2idx[tag]
    denom = tag_unigram_counts[tag] + num_words
    for word in word2idx:
        word_idx = word2idx[word]
        count = word_tag_counts.get((tag, word), 0)
        B[tag_idx, word_idx] = (count + 1) / denom

from viterbi import viterbi

test_sents = nltk.corpus.brown.tagged_sents(tagset='universal')[10150:10153]

for sent in test_sents:
    words = [word.lower() for word, tag in sent]
    true_tags = [tag for word, tag in sent]
    obs = []
    for word in words:
        obs.append(word2idx.get(word, word2idx['UNK']))

    qs, prob = viterbi(obs, pi, A, B)

    pred_tags = [idx2tag[q] for q in qs]

    print("Sentence:")
    print(" ".join(words))
    print("\nTrue Tags:")
    print(" ".join(true_tags))
    print("\nPredicted Tags:")
    print(" ".join(pred_tags))
    print("\n" + "-" * 50 + "\n")

