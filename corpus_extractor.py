from os import listdir
from os.path import isfile, join
import random
import xml.etree.ElementTree as ET
import argparse

parser = argparse.ArgumentParser(description='Extract the sentences from eaf files to make them usable for OpenNMT.')
parser.add_argument('--corpus-folder', dest='name',
                    help='set the name of the corpus', required=True)
parser.add_argument('--glosses', dest='glosses',
                    help='set the name of the glosses', required=True)
parser.add_argument('--trans', dest='translation',
                    help='set the name of the translation', required=True)
parser.add_argument('--lang', dest='language',
                    help='set the name of the language', required=True)
parser.add_argument('--only-number', dest='only_number', action='store_true',
                    help='configure the output')

args = parser.parse_args()

ONLY_NUMBER = args.only_number

CORPUS_NAME = args.name

GLOSSES = args.glosses

TRANSLATION = args.translation

LANGUAGE = args.language

ANNOTATION_RATIO = 5

path = "{}".format(CORPUS_NAME)

if not ONLY_NUMBER:
    print("Looking in {} for glosses with tier id {} and translation {}".format(CORPUS_NAME, GLOSSES, TRANSLATION))

onlyfiles = [f for f in listdir(path) if (isfile(join(path, f)) and f.endswith('.eaf'))]

class Annotation:
    def __init__(self, start, end, word):
        self.start = start
        self.end = end
        self.word = word

sentences = dict()

for file in onlyfiles:
    complete_path = path + file
    time_slots = dict()
    sl_annotations = list()
    tr_sentences = list()
    with open(complete_path, "r") as file:
        file_content = file.read()
        root = ET.fromstring(file_content)
        for time_slot in root.iter("TIME_SLOT"):
            time_slots[time_slot.attrib["TIME_SLOT_ID"]] = time_slot.attrib["TIME_VALUE"]
        tiers = [tier for tier in root.iter("TIER") if tier.attrib["TIER_ID"] == GLOSSES]
        if len(tiers) > 0:
            for annotation in tiers[0].iter("ALIGNABLE_ANNOTATION"):
                start = time_slots[annotation.attrib["TIME_SLOT_REF1"]]
                end = time_slots[annotation.attrib["TIME_SLOT_REF2"]]
                word = annotation.find("ANNOTATION_VALUE").text
                if(word != None):
                    sl_annotations.append(Annotation(start, end, word))
        sl_annotations = sorted(sl_annotations, key=lambda x: x.start)
        tiers = [tier for tier in root.iter("TIER") if tier.attrib["TIER_ID"] == TRANSLATION]
        if len(tiers) > 0:
            for annotation in tiers[0].iter("ALIGNABLE_ANNOTATION"):
                start = time_slots[annotation.attrib["TIME_SLOT_REF1"]]
                end = time_slots[annotation.attrib["TIME_SLOT_REF2"]]
                word = annotation.find("ANNOTATION_VALUE").text
                if(word != None):
                    tr_sentences.append(Annotation(start, end, word))
        for tr_sentence in tr_sentences:
            sentence = list()
            for sl_annotation in sl_annotations:
                if (sl_annotation.end > tr_sentence.start and sl_annotation.start < tr_sentence.end) or (sl_annotation.start < tr_sentence.end and sl_annotation.end > tr_sentence.start):
                    sentence.append(sl_annotation.word)
            if len(tr_sentence.word) > 0 and len(sentence) > 0:
                sentence_str = " ".join(sentence)
                sentences[tr_sentence.word] = sentence_str

def convert_sentence(str):
    return ' '.join(list(str.replace(" ", "*").replace("\n", "")))

def append_to_files(spoken_lang, sign_lang, f_spoken, f_sign):
    f_spoken.write(convert_sentence(spoken_lang))
    f_spoken.write('\n')
    f_sign.write(convert_sentence(sign_lang))
    f_sign.write('\n')

def is_valid(spoken_lang, sign_lang):
    spoken_length = len(list(spoken_lang))
    sign_length = len(list(sign_lang))
    return sign_length / spoken_length < ANNOTATION_RATIO and spoken_length / sign_length < ANNOTATION_RATIO

i = 0

f_train_sign = open("data/sign/{}/train.sign".format(LANGUAGE), "a")
f_train_spoken = open("data/sign/{}/train.spoken".format(LANGUAGE), "a")
f_val_sign = open("data/sign/{}/val.sign".format(LANGUAGE), "a")
f_val_spoken = open("data/sign/{}/val.spoken".format(LANGUAGE), "a")


for (spoken_lang, sign_lang) in sentences.items():
    if is_valid(spoken_lang, sign_lang):
        i += 1
        if (random.randint(1, 10) == 5):
            append_to_files(spoken_lang, sign_lang, f_val_spoken, f_val_sign)
        else:
            append_to_files(spoken_lang, sign_lang, f_train_spoken, f_train_sign)
print(("{}" if ONLY_NUMBER else "{} valid sentences").format(i))

f_train_sign.close()
f_train_spoken.close()
f_val_sign.close()
f_val_spoken.close()