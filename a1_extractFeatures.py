import numpy as np
import sys
import argparse
import os
import json
import re
import csv

first_person = open("/u/cs401/Wordlists/First-person", "r")
second_person = open("/u/cs401/Wordlists/Second-person", "r")
third_person = open("/u/cs401/Wordlists/Third-person", "r")
coord_conj = open("/u/cs401/Wordlists/Conjunct", "r")
slang_one = open("/u/cs401/Wordlists/Slang", "r")
slang_two = open("/u/cs401/Wordlists/Slang2", "r")


first_person_words = []
for line in first_person:
    linesplit = line.split(" ")
    for word in linesplit:
        word = word.replace('\n', '')
        first_person_words.append(word)

second_person_words = []
for line in second_person:
    linesplit = line.split(" ")
    for word in linesplit:
        word = word.replace('\n', '')
        second_person_words.append(word)

third_person_words = []
for line in third_person:
    linesplit = line.split(" ")
    for word in linesplit:
        word = word.replace('\n', '')
        third_person_words.append(word)

conj_words = []
for line in coord_conj:
    linesplit = line.split(" ")
    for word in linesplit:
        word = word.replace('\n', '')
        conj_words.append(word)

slang_words = []
for line in slang_one:
    linesplit = line.split(" ")
    for word in linesplit:
        word = word.replace('\n', '')
        slang_words.append(word)
for line in slang_two:
    linesplit = line.split(" ")
    for word in linesplit:
        word = word.replace('\n', '')
        slang_words.append(word)

future_tense_tags = ["'ll", "will"]
common_noun_tags = ["NN", "NNS"]
proper_noun_tags = ["NNP", "NNPS"]
adverb_tags = ["RB", "RBR", "RBS"]

AoA_words = {}
IMG_words = {}
FAM_words = {}
with open('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv', 'r') as csvfile:
    for line in csvfile.readlines():
        line_content = line.split(",")
        try:
            AoA_words[line_content[1]] = int(line_content[3])
            IMG_words[line_content[1]] = int(line_content[4])
            FAM_words[line_content[1]] = int(line_content[5])
        except ValueError:
            continue

V_words = {}
A_words = {}
D_words = {}
with open('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv', 'r') as csvfile:
     for line in csvfile.readlines():
        line_content = line.split(",")
        try:
            V_words[line_content[1]] = int(line_content[2])
            A_words[line_content[1]] = int(line_content[5])
            D_words[line_content[1]] = int(line_content[8])
        except ValueError:
            continue

   

LIWC_ID_File = open('/u/cs401/A1/feats/' + type_comment + '_IDs.txt', 'r')

def match_helper( comment, regex_list ):
    count = 0
    for regex in regex_list:
        matches = re.findall(re.escape(regex + "/"), comment)
        count += len(matches)
    return (count)

def extract1( comment, type_comment, id_comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features
    '''
    print('TODO')
    string_array = []
    # TODO: your code here
    feats = np.zeros(173)

    # 1) FIRST PERSON
    feats[0] = match_helper(comment, first_person_words)

    # 2) SECOND PERSON
    feats[1] = match_helper(comment, second_person_words)

    # 3) THIRD PERSON
    feats[2] = match_helper(comment, third_person_words)

    # 4) CONJ
    feats[3] = match_helper(comment, conj_words)

    # 5) TAG PAST TENSE
    feats[4] = len(re.findall("/vdb", comment, flags=re.IGNORECASE))

    # 6) TAG FUTURE TENSE
    feats[5] = match_helper(comment, future_tense_tags)
    feats[5] += len(re.findall(r'[going/\S+ to/\S+ \S+/vb]{1}', comment, flags=re.IGNORECASE))

    # 7) NUMBER COMMA
    feats[6] = len(re.findall(" ,/", comment))

    # 8) MULTI PUNCTUATION
    feats[7] = len(re.findall(r'[ ' + re.escape(string.punctuation) + r'/' + re.escape(string.punctuation) + r']{2,}', comment))

    # 9) COMMON NOUNS
    for common_noun in common_noun_tags:
        feats[8] += len(re.findall(r'[/' + common_noun + r' ]{1}', comment, flags=re.IGNORECASE))

    # 10) PROPER NOUNS
    for proper_noun in proper_noun_tags:
        feats[9] += len(re.findall(r'[/' + proper_noun + r' ]{1}', comment, flags=re.IGNORECASE))

    # 11) ADVERBS
    for adverb in adverb_tags:
        feats[10] += len(re.findall(r'[/' + adverb + r" ]{1}", comment, flags=re.IGNORECASE))

    # 12) wh- WORDS
    feats[11] = len(re.findall(r'[ wh\S+]{1}', comment, flags=re.IGNORECASE))

    # 13) slang ACRO
    for slang in slang_words:
        feats[12] += len(re.findall(r' ' + re.escape(slang) + r'', comment, flags = re.IGNORECASE))

    # 14) UPPERCASE
    feats[13] = len(re.findall(r' [A-Z]{3,}[/]{1}', comment))

    # 17) NUMBER SENTENCES
    feats[16] = len(re.findall(r'' + re.escape("\n") + r'', comment))

    # 15) LENGTH
    feats[14] = len(re.findall(r'\S+/\S+', comment)) / feats[16]

    # 16) LENGTH (NOT PUNCT)
    toks = re.findall(r'( \w+/)', comment)
    num_toks = len(toks)
    total_length = 0
    for tok in toks:
        total_length += len(tok) - 1
    feats[15] = total_length / num_toks

    # 18, 19, 20) AVERAGE OF AoA, IMG, FAM
    toksFound = 0
    AoASum = 0
    IMGSum = 0
    FAMSum = 0
    for tok in toks:
        tok = tok.rstrip("/").lstrip(" ")
        if(tok in AoA_words):
            toksFound += 1
            AoASum += AoA_words[tok]
            IMGSum += IMG_words[tok]
            FAMSum += FAM_words[tok]
    if(toksFound > 0):
        feats[17] = AoASum/toksFound
        feats[18] = IMGSum/toksFound
        feats[19] = FAMSum/toksFound


    # 21, 22, 23) STD DEV OF AoA, IMG, FAM
    toksFound = 0
    AoASum = 0
    IMGSum = 0
    FAMSum = 0
    for tok in toks:
        tok = tok.rstrip("/").lstrip(" ")
        if(tok in AoA_words):
            toksFound += 1
            AoASum += (AoA_words[tok] - feats[17])**2
            IMGSum += (IMG_words[tok] - feats[18])**2
            FAMSum += (FAM_words[tok] - feats[19])**2
    if(toksFound > 0):
        feats[20] = (AoASum/toksFound)**(1/2)
        feats[21] = (IMGSum/toksFound)**(1/2)
        feats[22] = (FAMSum/toksFound)**(1/2)


    # 24, 25, 26) AVERAGE of V, A, D
    toksFound = 0
    VSum = 0
    ASum = 0
    DSum = 0
    for tok in toks:
        tok = tok.rstrip("/").lstrip(" ")
        if(tok in AoA_words):
            toksFound += 1
            VSum += V_words[tok]
            ASum += A_words[tok]
            DSum += D_words[tok]
    if(toksFound > 0):
        feats[23] = VSum/toksFound
        feats[24] = ASum/toksFound
        feats[25] = DSum/toksFound


    # 27, 28, 29) STD DEV OF V, A, D
    toksFound = 0
    VSum = 0
    ASum = 0
    DSum = 0
    for tok in toks:
        tok = tok.rstrip("/").lstrip(" ")
        if(tok in AoA_words):
            toksFound += 1
            VSum += (V_words[tok] - feats[23])**2
            ASum += (A_words[tok] - feats[24])**2
            DSum += (D_words[tok] - feats[25])**2
    if(toksFound > 0):
        feats[26] = (VSum/toksFound)**(1/2)
        feats[27] = (ASum/toksFound)**(1/2)
        feats[28] = (DSum/toksFound)**(1/2)

    # id_comment, type_comment
        for i, line in enumerate(LIWC_ID_File.readlines()):
            line = line.strip()
            if(line == type_ID): 


def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))

    # TODO: your co
    for i, line in enumerate(data):
        j = json.loads(line)
        comment = j['body']
        type_comment = j['cat']

        feats[i, 0:172] = extract1(comment, type_comment, j['id'])

        if(type_comment == "Left"):
            feats[i][173] = 0
        elif(type_comment == "Center"):
            feats[i][173] = 1
        elif(type_comment == "Right"):
            feats[i][173] = 2
        elif(type_comment == "Alt"):
            feats[i][173] = 3
        else:
            print("Unknown type: " + type_comment)
            sys.exit(-1)



    np.savez_compressed( args.output, feats)

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)

