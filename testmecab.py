# encoding: utf-8
import MeCab

CHAR_CODE = "utf-8"



def mecabparse(text):
    mt = MeCab.Tagger()
    # text = text.encode(CHAR_CODE)

    res = mt.parse(text)
    words = []
    features = []

    lines = res.split("\n")

    for line in lines[0:-2]	:
        word, feature = line.split("\t", 2)
        words.append(word)
        features.append(feature.split(","))

    print(words)
    print(features)

    return words, features



if __name__ == '__main__':
    text = u"本日は晴れです"
    mecabparse(text)


# res = mt.parseToNode(text)

# while res:
#     print(res.surface, res.feature)
#     res = res.next
