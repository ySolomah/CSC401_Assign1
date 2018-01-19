import sys
import argparse
import os
import json
import html
from html.parser import HTMLParser
import re
import spacy
import string

indir = '/u/cs401/A1/data/';

stopwords = open("/u/cs401/Wordlists/StopWords", "r")
common_abbrev = open("/u/cs401/Wordlists/abbrev.english", "r")

def preproc1( comment , steps=range(1,11) ):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    
    comment_after_five = ""
    modComm = ''
#    print("Comment before: " + comment)
    if 1 in steps:
        comment = comment.replace('\n', '')
        comment = re.sub(r'[ ]+', " ", comment)
#        print("Comment after 1: " + comment)
    if 2 in steps:
        remove_html_escape = HTMLParser()
        comment = remove_html_escape.unescape(comment)
#        print("Comment after 2: " + comment)
    if 3 in steps:
        comment = re.sub(r'http\S*', '', comment)
        comment = re.sub(r'www\S*', '', comment)
#        print("Comment after 3: " + comment)
    if 4 in steps:
        punct = string.punctuation
        punct = punct.replace("'", "")
        comment = re.sub(r'([' + re.escape(punct) + r']+)', r' \1 ', comment)
#        print("Comment after 4.1: " + comment)
        comment = re.sub(r'([a-zA-Z] . [a-zA-Z . ]+)', r'\1'.replace(" ", ""), comment) 
#        print("Comment after 4.2: " + comment)
    if 5 in steps:
        comment = re.sub(r"([A-Za-z]{1}[']{1}[A-Za-z]{1})", r' \1', comment)
        comment = re.sub(r"([A-Za-z]{1}['] ])", r'\1'.replace("'", "") + " " + "'", comment)
        comment = re.sub(r'[ ]+', " ", comment)
        comment_after_five = comment
#        print("Comment after 5: " + comment)
    if 6 in steps:
        new_comment_temp = ""
        nlp = spacy.load('en', disable=['parser', 'ner'])
        utt = nlp(u"" + comment + "")
        for token in utt:
            new_comment_temp = new_comment_temp + token.text + "/" + token.tag_ + " "
        comment = new_comment_temp
#        print("Comment after 6: " + comment)
    if 7 in steps:
        comment = " " + comment + " "
        for line in stopwords:
            linesplit = line.split(" ")
            for stopword in linesplit:
                stopword = stopword.replace('\n', '')
                comment = re.sub(r' ' + re.escape(stopword) + r'[/]\S+ ', ' ', comment, flags=re.IGNORECASE)
#        print("Comment after 7: " + comment)
    if 8 in steps:
        nlp = spacy.load('en', disable=['parser', 'ner'])
        utt = nlp(u"" + comment_after_five + "")
        for token in utt:
            if(token.lemma_[0] == '-' and token.text[0] != '-'):
                continue
            else:
                comment = re.sub(r'' + re.escape(token.text) + r'', token.lemma_, comment)
#        print("Comment after 8: " + comment)
    if 9 in steps:
        split_comment = comment.split(" ")
        new_comment = ""
        for i in range(len(split_comment)):
            if(len(split_comment[i]) == 0):
                continue;
            elif(i == 0):
                new_comment = new_comment + split_comment[i] + " "
                continue;
            elif(split_comment[i][0] == '.'):
                abbrev_flag = False
                for line in common_abbrev:
                    if(abbrev_flag == True):
                        break;
                    linesplit = line.split(" ")
                    for abbrev in linesplit:
                        abbrev = abbrev.replace('.', '')
                        abbrev = abbrev.replace('\n', '')
                        if(bool(re.search(" " + abbrev + "/", " " + split_comment[i-1]))):
                            abbrev_flag = True
                if(abbrev_flag == False):
                    new_comment = new_comment + split_comment[i] + "\n"
                else:
                    new_comment = new_comment + split_comment[i] + " "
            else:
                new_comment = new_comment + split_comment[i] + " "
        comment = new_comment
#        print("Comment after 9: " + comment)

    if 10 in steps:
        comment = comment.lower()
#        print("Comment after 10: " + comment)

    modComm = comment
    return modComm

def main( args ):

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)
            print("File name: " + file)
            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            # TODO: read those lines with something like `j = json.loads(line)`
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            # TODO: append the result to 'allOutput'
            
            line_start = args.ID[0] % len(data)

            for i in range(line_start, line_start + args.max):
                print("i is: " + str(i))
                line = data[i]
                j = json.loads(line)
 #               print("j is: " + str(j))
                pre_processed_data = preproc1(j['body'])
                j['body'] = pre_processed_data
                j['cat'] = file
                allOutput.append(j)
                   
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000, type=int)
    args = parser.parse_args()

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)
                
    main(args)
