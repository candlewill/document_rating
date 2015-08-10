__author__ = 'NLP-PC'
import csv
import pickle

def save_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE)
        writer.writerows(data)


def dump_picle(data, filename):
    pickle.dump(data, open(filename, "wb"))
