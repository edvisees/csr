import sys
import os
import xml.etree.ElementTree as ET
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer


def validate_sent(seg):
    unk = 0
    all = 0
    for token in seg:
        if token.tag == 'TOKEN':
            if token.attrib['pos'] == 'unknown':
                unk += 1
            all += 1

    if 1.0 * unk / all > 0.3:
        # print("Too many unknown tokens, maybe a dirty sentence.")
        return False

    return True


def ltf2txt(in_dir, out_dir):
    lemmatizer = WordNetLemmatizer()

    for file in os.listdir(in_dir):
        tree = ET.parse(os.path.join(in_dir, file))
        root = tree.getroot()
        doc = root[0]

        docid = file.split(".")[0]

        out_file = os.path.join(out_dir, docid + '.txt')
        sent_file = os.path.join(out_dir, docid + '.sent')

        with open(out_file, 'w') as out, open(sent_file, 'w') as sent_out:
            text = ""

            for entry in doc:
                bad_ending = False
                sents = []

                for seg in entry:
                    new_sent = True

                    seg_attr = seg.attrib
                    seg_id = seg_attr['id']
                    key_frame = seg_attr.get('keyframe', '-')

                    sents.append(
                        "{} {} {} {}".format(
                            int(seg_attr['start_char']) - 1,
                            int(seg_attr['end_char']) + 1,
                            seg_id,
                            key_frame,
                        )
                    )

                    is_valid = validate_sent(seg)

                    for token in seg:
                        if token.tag == 'TOKEN':
                            begin = int(token.attrib['start_char']) - 1

                            if new_sent:
                                if bad_ending and begin > len(text):
                                    text += '.'
                                if begin > len(text):
                                    text += ('\n' * (begin - len(text)))
                            else:
                                text += (' ' * (begin - len(text)))

                            if is_valid:
                                text += token.text
                            else:
                                text += ' ' * (len(token.text))

                            bad_ending = not token.attrib['pos'] == 'punct'
                            new_sent = False
            out.write(text)
            sent_out.write('\n'.join(sents))


def main():
    in_dir = sys.argv[1]
    out_file = sys.argv[2]
    format_in = sys.argv[3]

    if format_in == 'ltf':
        ltf2txt(in_dir, out_file)


if __name__ == '__main__':
    main()
