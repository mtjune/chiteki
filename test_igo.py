# encoding: utf-8
import igo

CHAR_CODE = "utf-8"
DIC_DIR = '/home/yamajun/workspace/tmp/igo_ipadic'



if __name__ == '__main__':


    text = u"今日はとても良い天気ですね"
    tagger = igo.tagger.Tagger(DIC_DIR)

    words = tagger.parse(text)

    for word in words:
        print(word.surface, word.feature, word.start)
