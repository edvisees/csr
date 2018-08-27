import sys
import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom


class LTF:
    def __init__(self, docid, language='eng'):
        self.docid = docid

        self.root = ET.Element('LCTL_TEXT')
        self.root.set('lang', language)

        self.doc_node = ET.SubElement(self.root, 'DOC')
        self.doc_node.set('id', docid)
        self.doc_node.set('lang', language)
        self.doc_node.set('grammar', 'none')

        self.text_node = ET.SubElement(self.doc_node, 'TEXT')
        self.text_offset = 0

        self.current_seg = None
        self.seg_text = []

        self.seg_id = 0
        self.token_id = 0

        self.total_len = 0

    def get_new_seg(self):
        sid = 'segment-{}'.format(self.seg_id)
        return sid

    def get_new_token(self):
        tid = 'token-{}-{}'.format(self.seg_id, self.token_id)
        self.token_id += 1
        return tid

    def begin_seg(self, keyframe=None):
        self.current_seg = ET.SubElement(self.text_node, 'SEG')
        self.current_seg.set('start_char', str(self.text_offset + 1))
        self.current_seg.set('id', self.get_new_seg())
        if keyframe:
            self.current_seg.set('keyframe', keyframe)

    def add_token(self, text):
        self.seg_text.append(text)

    def end_seg(self):
        sep = ''

        text_node = ET.SubElement(self.current_seg, 'ORIGINAL_TEXT')
        text_node.text = ' '.join(self.seg_text)

        for text in self.seg_text:
            self.text_offset += len(sep)
            sep = ' '
            token = ET.SubElement(self.current_seg, 'TOKEN')
            token.set('id', self.get_new_token())
            token.set('pos', 'word')
            token.set('morph', 'none')
            token.set('start_char', str(self.text_offset + 1))
            token.text = text
            self.text_offset += len(text)

            self.total_len += len(text) + len(sep)

            token.set('end_char', str(self.text_offset))

        self.current_seg.set('end_char', str(self.text_offset))

        assert len(text_node.text) == int(
            self.current_seg.attrib['end_char']) - int(
            self.current_seg.attrib['start_char']) + 1

        # Leave some space to add stuff in between.
        self.text_offset += 2

        self.seg_id += 1
        self.seg_text.clear()

    def write(self, out):
        self.doc_node.set('raw_text_char_length', str(self.total_len))

        rough_string = ET.tostring(self.root, 'UTF-8')
        reparsed = minidom.parseString(rough_string)
        out.write(reparsed.toprettyxml(indent='  '))


def text_to_ltf(in_file_path, out_file_path, docid):
    with open(in_file_path) as inf:
        ltf = LTF(docid)
        is_empty = True
        for line in inf:
            ltf.begin_seg()
            for word in line.split(' '):
                ltf.add_token(word)
            ltf.end_seg()
            is_empty = False

        if not is_empty:
            with open(out_file_path, 'w') as out:
                ltf.write(out)


def sausage_to_ltf(in_file_path, out_file_path, docid):
    with open(in_file_path) as inf, open(out_file_path, 'w') as out:
        audio_parsed = json.load(inf)

        ltf = LTF(docid)

        for keyframe, intervals in audio_parsed:
            for interval in intervals:
                best_words = []

                ltf.begin_seg(keyframe=keyframe)

                for hypos in interval['sausage']:
                    best_score = -1
                    best_word = ""

                    for hypo in hypos:
                        word = hypo['word']
                        confidence = hypo['confidence']

                        if confidence > best_score:
                            best_score = confidence
                            best_word = word

                    ltf.add_token(best_word)

                ltf.end_seg()

        ltf.write(out)


if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if len(sys.argv) > 3:
        suffix = sys.argv[3]
        input_format = sys.argv[4]
    else:
        suffix = '.json'
        input_format = 'sausage'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Converting data in {} to LTF, store to {}".format(
        input_dir, output_dir))

    for f in os.listdir(input_dir):
        file_path = os.path.join(input_dir, f)

        if not f.endswith(suffix):
            continue

        print("Converting {}".format(f))

        docid = f.strip(suffix)
        output_path = os.path.join(output_dir, f + '.ltf.xml')

        if input_format == 'sausage':
            sausage_to_ltf(file_path, output_path, docid)
        elif input_format == 'txt':
            text_to_ltf(file_path, output_path, docid)
