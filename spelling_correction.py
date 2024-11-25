import csv
from collections import defaultdict

def load_error_model():
    deletion_counts = defaultdict(int)
    insertion_counts = defaultdict(int)
    substitution_counts = defaultdict(int)
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)

    with open('deletions.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for prefix, deleted_char, count in reader:
            key = prefix + deleted_char
            deletion_counts[key] += int(count)

    with open('additions.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for prefix, added_char, count in reader:
            key = prefix + added_char
            insertion_counts[key] += int(count)

    with open('substitutions.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for original_char, error_char, count in reader:
            key = error_char + original_char
            substitution_counts[key] += int(count)

    with open('bigrams.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for bigram, count in reader:
            bigram_counts[bigram] += int(count)

    with open('unigrams.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for char, count in reader:
            unigram_counts[char] += int(count)

    return deletion_counts, insertion_counts, substitution_counts, bigram_counts, unigram_counts

def load_language_model():
    word_counts = defaultdict(int)
    total_word_count = 0

    with open('count_1w.txt', 'r') as f:
        for line in f:
            try:
                word, count = line.strip().split('\t')
                word_counts[word] = int(count)
                total_word_count += int(count)
            except ValueError:
                continue

    return word_counts, total_word_count

def P_w(word, word_counts, total_word_count):
    word_count = word_counts.get(word, 0)
    if total_word_count == 0:
        return 0.0
    probability = word_count / total_word_count
    return probability

def P_edit(edit_type, edit_info, deletion_counts, insertion_counts, substitution_counts, bigram_counts, unigram_counts):
    if edit_type == 'deletion':
        wi_minus_1, wi = edit_info
        del_key = wi_minus_1 + wi
        deletion_count = deletion_counts.get(del_key, 0)
        bigram_count = bigram_counts.get(del_key, 0)
        if bigram_count == 0:
            return 0.0
        probability = deletion_count / bigram_count
        return probability
    elif edit_type == 'insertion':
        wi_minus_1, xi = edit_info
        ins_key = wi_minus_1 + xi
        insertion_count = insertion_counts.get(ins_key, 0)
        unigram_count = unigram_counts.get(wi_minus_1, 0)
        if unigram_count == 0:
            return 0.0
        probability = insertion_count / unigram_count
        return probability
    elif edit_type == 'substitution':
        xi, wi = edit_info
        sub_key = xi + wi
        substitution_count = substitution_counts.get(sub_key, 0)
        unigram_count = unigram_counts.get(wi, 0)
        if unigram_count == 0:
            return 0.0
        probability = substitution_count / unigram_count
        return probability
    else:
        return 0.0

def get_edits(word):
    edits = []
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    for L, R in splits:
        if len(R) > 0:
            edited_word = L + R[1:]
            wi_minus_1 = L[-1] if L else '#'
            wi = R[0]
            edits.append(('deletion', (wi_minus_1, wi), edited_word))

    for L, R in splits:
        wi_minus_1 = L[-1] if L else '#'
        for c in letters:
            edited_word = L + c + R
            xi = c
            edits.append(('insertion', (wi_minus_1, xi), edited_word))

    for L, R in splits:
        if len(R) > 0:
            for c in letters:
                if R[0] != c:
                    edited_word = L + c + R[1:]
                    xi = R[0]
                    wi = c
                    edits.append(('substitution', (xi, wi), edited_word))
    return edits

def correct(original: str) -> str:
    deletion_counts, insertion_counts, substitution_counts, bigram_counts, unigram_counts = load_error_model()
    word_counts, total_word_count = load_language_model()

    candidates = get_edits(original)
    max_probability = 0
    best_candidate = original

    for edit_type, edit_info, candidate in candidates:
        word_probability = P_w(candidate, word_counts, total_word_count)

        if word_probability == 0:
            continue

        edit_probability = P_edit(edit_type, edit_info, deletion_counts, insertion_counts, substitution_counts, bigram_counts, unigram_counts)
        if edit_probability == 0:
            continue

        total_probability = word_probability * edit_probability

        if total_probability > max_probability:
            max_probability = total_probability
            best_candidate = candidate

    if max_probability == 0:
        return original

    return best_candidate


misspelled_word = 'ther'
corrected_word = correct(misspelled_word)

print(f"Original word: {misspelled_word}")
print(f"Corrected word: {corrected_word}")
