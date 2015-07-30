__author__ = 'NLP-PC'
# coding: utf-8
import goslate
from load_data import load_lexicon
from file_name import get_file_path
from load_data import load_anew
import os
os.chdir('..')
print(load_lexicon(get_file_path('lexicon')))
words,valence, arousal = load_anew(get_file_path('anew'))
gs = goslate.Goslate()
for tw_text in gs.translate(words, 'zh-tw'):
    print(tw_text)
print(gs.translate('Hi', 'zh-TW'))
# You could get all supported language list through get_languages
languages = gs.get_languages()
print(languages['zh-TW'])
