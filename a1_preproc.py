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


def preproc1( comment , steps=range(1,11) ):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''

    modComm = ''
    print("Comment before: " + comment)
    if 1 in steps:
        comment = comment.replace('\n', '')
        print("Comment after 1: " + comment)
    if 2 in steps:
        remove_html_escape = HTMLParser()
        comment = remove_html_escape.unescape(comment)
        print("Comment after 2: " + comment)
    if 3 in steps:
        comment = re.sub(r'http\S*', '', comment)
        comment = re.sub(r'www\S*', '', comment)
        print("Comment after 3: " + comment)
    if 4 in steps:
        punct = string.punctuation
        punct = punct.replace("'", "")
        comment = re.sub(r'([' + re.escape(punct) + r']+)', r' \1 ', comment)
        print("Comment after 4.1: " + comment)
        comment = re.sub(r'([a-zA-Z] . [a-zA-Z . ]+)', r'\1'.replace(" ", ""), comment) 
        print("Comment after 4.2: " + comment)
    if 5 in steps:
        comment = re.sub(r"([A-Za-z]{1}[']{1}[A-Za-z]{1})", r' \1', comment)
        comment = re.sub(r"([A-Za-z]{1}['] ])", r'\1'.replace("'", "") + " " + "'", comment)
        print("Comment after 5: " + comment)
    if 6 in steps:
        print('TODO')
        nlp = spacy.load('en', disable=['parser', 'ner'])
        utt = nlp(u"I know the best words")
        for token in utt:
           print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
    if 7 in steps:
        print('TODO')
    if 8 in steps:
        print('TODO')
    if 9 in steps:
        print('TODO')
    if 10 in steps:
        print('TODO')
        
    return modComm

def main( args ):

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

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
                print("j is: " + str(j))
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
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)
                
    main(args)
