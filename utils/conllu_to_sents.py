#!/usr/bin/env python
#coding:utf8

import sys
import codecs
if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: conllu_to_sents.py conllu_file sents_file"
    sentences = []
    with codecs.open(sys.argv[1], "r", encoding="utf=8") as fi:
        sentences_list = fi.read().strip().split("\n\n")
        for sentence in sentences_list:
            sent = ""
            words_list = sentence.strip().split('\n')
            start_index = 0
            end_index = 0
            for word in words_list:
                fields = word.strip().split('\t')
                if fields[0][0] != '#' and u'.' not in fields[0] and (u'-' in fields[0] or (int(fields[0])<start_index or int(fields[0])>end_index)):
                    if u'-' in fields[0]:
                        start_index, end_index = fields[0].split('-')
                        start_index, end_index = int(start_index), int(end_index)
                    sent += fields[1]
                    if "SpaceAfter=No" not in fields[-1]:
                        sent += ' '
            sentences.append(sent)

    with codecs.open(sys.argv[2], "w", encoding="utf-8") as fo:
        for sent in sentences:
            fo.write(sent + '\n')
        
        
