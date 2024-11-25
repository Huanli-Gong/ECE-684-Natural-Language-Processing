from collections import defaultdict
import random

def finish_sentence(sentence, n, corpus, randomize=False):
    ngram_count = defaultdict(lambda: defaultdict(int))

    for i in range(len(corpus) - n + 1):
        for length in range(1, n):
            prefix = tuple(corpus[i:i + length])
            next_word = corpus[i + length]
            ngram_count[prefix][next_word] += 1

    ngram_probs = defaultdict(dict)
    for prefix in ngram_count:
        total = sum(ngram_count[prefix].values())
        for next_word in ngram_count[prefix]:
            ngram_probs[prefix][next_word] = ngram_count[prefix][next_word] / total

    current_sentence = list(sentence)
    while len(current_sentence) < 10:
        found = False
        for i in range(n, 0, -1):
            prefix = tuple(current_sentence[-(i - 1):]) if len(current_sentence) >= i - 1 else ()

            if prefix in ngram_probs:
                candidates = ngram_probs[prefix]
                if randomize:
                    next_word = random.choices(list(candidates.keys()), weights=candidates.values())[0]
                else:
                    next_word = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[0][0]

                current_sentence.append(next_word)
                found = True
                break

            elif i > 1:
                shorter_prefix = prefix[1:]
                if shorter_prefix in ngram_probs:
                    candidates = {word: count * 0.4 for word, count in ngram_probs[shorter_prefix].items()}
                    if randomize:
                        next_word = random.choices(list(candidates.keys()), weights=candidates.values())[0]
                    else:
                        next_word = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[0][0]

                    current_sentence.append(next_word)
                    found = True
                    break

        if not found or current_sentence[-1] in {'.', '?', '!'}:
            break

    return current_sentence