import sys
import os
import xml.etree.ElementTree as ET
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer


def ltf2conllu(in_dir, out_file):
    lemmatizer = WordNetLemmatizer()

    with open(out_file, 'w') as out:
        for file in os.listdir(in_dir):
            tree = ET.parse(os.path.join(in_dir, file))
            root = tree.getroot()
            doc = root[0]

            docid = file.split(".")[0]

            out.write("# newdoc id = %s\n" % docid)

            for text in doc:
                sent_id = 0

                for seg in text:
                    sent_id += 1
                    token_id = 0

                    words = []
                    spans = []
                    for tokens in seg:
                        if tokens.tag == 'TOKEN':
                            words.append(tokens.text)
                            begin = int(tokens.attrib['start_char']) - 1
                            end = int(tokens.attrib['end_char'])
                            spans.append(
                                "%d,%d" % (begin, end)
                            )

                    word_pos = nltk.pos_tag(words)

                    l_tokens = []
                    for (word, pos), span in zip(word_pos, spans):
                        token_id += 1
                        l_tokens.append(
                            [token_id, word, lemmatizer.lemmatize(word),
                             "_", pos, "_", "_", "_", "_", "_", span])

                    out.write("# sent_id = %d\n" % sent_id)

                    for tokens in l_tokens:
                        out.write("\t".join([str(t) for t in tokens]))
                        out.write('\n')
                    out.write("\n")


def txt2conllu(in_dir, out_file):
    lemmatizer = WordNetLemmatizer()

    with open(out_file, 'w') as out:
        for file in os.listdir(in_dir):
            docid = file.split(".")[0]
            out.write("# newdoc id = %s\n" % docid)

            with open(os.path.join(in_dir, file)) as file_content:
                sent_id = 0
                for line in file_content:
                    sent_id += 1
                    line = line.strip()

                    if not line:
                        continue

                    spans_gen = WhitespaceTokenizer().span_tokenize(line)
                    tokens = []
                    spans = []

                    for span in spans_gen:
                        tokens.append(line[span[0]: span[1]])
                        spans.append("%d,%d" % (span[0], span[1]))

                    word_pos = nltk.pos_tag(tokens)

                    out.write("# sent_id = %d\n" % sent_id)

                    token_id = 0
                    l_tokens = []
                    for (word, pos), span in zip(word_pos, spans):
                        token_id += 1
                        l_tokens.append(
                            [token_id, word, lemmatizer.lemmatize(word),
                             "_", pos, "_", "_", "_", "_", "_", span])

                    for tokens in l_tokens:
                        out.write("\t".join([str(t) for t in tokens]))
                        out.write('\n')
                    out.write("\n")


def main():
    in_dir = sys.argv[1]
    out_file = sys.argv[2]
    format_in = sys.argv[3]

    if format_in == 'txt':
        txt2conllu(in_dir, out_file)
    elif format_in == 'ltf':
        txt2conllu(in_dir, out_file)


if __name__ == '__main__':
    main()
