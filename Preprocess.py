import time
import spacy
import numpy as np


def load_data(output, doc_name, nlp):
    f = open(doc_name, encoding="utf-8")

    offset = 0
    sent = []

    origin = []
    correct = []

    for line in f.readlines():

        if line[0] == 'S':

            offset = 0
            sent = nlp(line[1:].strip())
            origin = []
            for token in sent:
                origin.append(token.text)
            correct = []



        elif line[0] == 'A':

            annot = line[1:].strip().split('|||')
            start = int(annot[0].split()[0])
            end = int(annot[0].split()[1])

            replace = annot[2]

            if not start == -1:
                for i in range(offset, start):
                    correct.append(origin[i])

                for token in nlp(replace):
                    correct.append(token.text)

                offset = end

            else:
                offset = len(sent)





        else:

            for i in range(offset, len(sent)):
                correct.append(sent[i].text)

            output.write(' '.join(origin) + '\n')
            output.write(' '.join(correct) + '\n')


def load_files(file_list, nlp, output):
    with open(output, 'w', encoding="utf-8") as f:
        for file in file_list:
            load_data(f, file, nlp)


def load_preprocessed_data(file):
    data = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            # print(line)
            sent = line.strip().split()
            data.append(sent)
    return data
