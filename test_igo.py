# encoding: utf-8
from igo import Tagger

CHAR_CODE = "utf-8"
DIC_DIR = "~/workspace/tmp/igo_ipadic"



if __name__ == '__main__':


    text = u"すもももももももものうち"
    tagger = Tagger(DIC_DIR)

    words = tagger.parse(text)

    for word in words:
        print(word.surface, word.feature, word.start)
