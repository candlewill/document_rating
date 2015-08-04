__author__ = 'NLP-PC'
import csv


def save_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE)
        writer.writerows(data)
