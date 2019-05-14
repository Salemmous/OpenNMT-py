from os import listdir, mkdir
from os.path import isfile, join
import argparse
import re, collections
punctuation = re.compile(r"[\.\?\!\,\-\]\[\(\)]|\s", re.IGNORECASE)
spaces = re.compile(r"\s+")

parser = argparse.ArgumentParser(description='Compress the characters of extracted sentences.')
parser.add_argument('--lang', dest='lang',
                    help='set the language', required=True)
parser.add_argument('--num-merges', dest='merge',
                    help='set the num merges', default=10, type=int)

args = parser.parse_args()
LANGUAGE = args.lang

NUM_MERGES = args.merge

def to_words(string_to_words):
    string_to_words = string_to_words.replace("\n", "*")
    string_to_words = re.sub(punctuation, " ", string_to_words)
    string_to_words = re.sub(spaces, " ", string_to_words)
    return [x.strip() for x in string_to_words.split("*") if x != ""]

def to_sentences(string_to_sentences):
    sentences = string_to_sentences.split("\n")
    return [to_words(x) for x in sentences if len(x) != 0]

def count_to_dict(words):
    count_dict = dict()
    for word in words:
        if word in count_dict:
            count_dict[word] += 1
        else:
            count_dict[word] = 1
    return count_dict

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def bpe_vocab(vocab):
    for i in range(NUM_MERGES):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    return vocab

def bpe(words):
    vocab = count_to_dict(words)
    return bpe_vocab(vocab)

def table_bpe(bpe_words):
    bpe_dict = dict()
    for word, count in bpe_words.items():
        normal_word = word.replace(" ", "")
        bpe_dict[normal_word] = word
    return bpe_dict

def replace_with_bpe(sentences, bpe_words):
    bpe_dict = table_bpe(bpe_words)

    new_sentences = list()
    for sentence in sentences:
        new_sentence = list()
        for word in sentence:
            normal_word = word.replace(" ", "")
            new_sentence.append(bpe_dict[normal_word])
        new_sentences.append(new_sentence)
    return new_sentences

def to_string(sentences):
    return "\n".join([" * ".join(sentence) for sentence in sentences])


f_train_sign = open("data/sign/{}/train.sign".format(LANGUAGE), "r", encoding='utf8')
f_train_spoken = open("data/sign/{}/train.spoken".format(LANGUAGE), "r", encoding='utf8')
f_val_sign = open("data/sign/{}/val.sign".format(LANGUAGE), "r", encoding='utf8')
f_val_spoken = open("data/sign/{}/val.spoken".format(LANGUAGE), "r", encoding='utf8')

test_sign = f_val_sign.read()
test_sign_length = len(test_sign.split("\n"))-1
test_spoken = f_val_spoken.read()
test_spoken_length = len(test_spoken.split("\n"))-1

sign_words_chain = f_train_sign.read() + '\n' + test_sign
spoken_words_chain = f_train_spoken.read() + '\n' + test_spoken

f_train_sign.close()
f_train_spoken.close()
f_val_sign.close()
f_val_spoken.close()

sign_words = to_words(sign_words_chain)
spoken_words = to_words(spoken_words_chain)
sign_sentences = to_sentences(sign_words_chain)
spoken_sentences = to_sentences(spoken_words_chain)

byte_pair_encoded_sign = bpe(sign_words)
byte_pair_encoded_spoken = bpe(spoken_words)

replaced_sign = replace_with_bpe(sign_sentences, byte_pair_encoded_sign)
replaced_spoken = replace_with_bpe(spoken_sentences, byte_pair_encoded_spoken)

try:
    mkdir("data/sign-bpe-{}/".format(NUM_MERGES))
except OSError:  
    pass
try:
    mkdir("data/sign-bpe-{}/{}/".format(NUM_MERGES, LANGUAGE))
except OSError:  
    pass
f_train_sign = open("data/sign-bpe-{}/{}/train.sign".format(NUM_MERGES, LANGUAGE), "w+")
f_train_spoken = open("data/sign-bpe-{}/{}/train.spoken".format(NUM_MERGES, LANGUAGE), "w+")
f_val_sign = open("data/sign-bpe-{}/{}/val.sign".format(NUM_MERGES, LANGUAGE), "w+")
f_val_spoken = open("data/sign-bpe-{}/{}/val.spoken".format(NUM_MERGES, LANGUAGE), "w+")

f_train_sign.write(to_string(replaced_sign[0:-test_sign_length]))
f_train_spoken.write(to_string(replaced_spoken[0:-test_spoken_length]))
f_val_sign.write(to_string(replaced_sign[-test_sign_length:]))
f_val_spoken.write(to_string(replaced_spoken[-test_spoken_length:]))

f_train_sign.close()
f_train_spoken.close()
f_val_sign.close()
f_val_spoken.close()