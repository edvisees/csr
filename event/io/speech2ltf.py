#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from xml.etree import ElementTree as ET
from xml.dom import minidom
import os, sys, argparse
from collections import defaultdict


def get_args():
   ap = argparse.ArgumentParser(description="Convert speech transcripts to LTF files")
   ap.add_argument("transcript_file", help="format: utt_id<TAB>lang_id<TAB>transcripts")
   ap.add_argument("output_ltf_dir", help="Output directory where LTF files should be writtern")
   return ap.parse_args()


def normalize_lang(lang):
   lang = lang.strip().lower()
   if lang.startswith("en"):
      return "eng"
   elif lang.startswith("ru"):
      return "rus"
   elif lang.startswith("uk"):
      return "ukr"
   return None


def write_ltf(outfile, lang, doc_id, segments):
   lctl_text = ET.Element("LCTL_TEXT", {"lang": lang})
   doc = ET.SubElement(lctl_text, "DOC", {"id": doc_id, "lang": lang, "media_type": "video"})
   text = ET.SubElement(doc, "TEXT")
   start_char = 1
   for utt_id in sorted(segments.keys()):
      words = [w for w in segments[utt_id].split() if w.lower() != "<unk>"]
      if len(words) == 0:
         continue
      segment_text = " ".join(words)
      seg = ET.SubElement(text, "SEG", {
         "id": utt_id,
         "start_char": str(start_char),
         "end_char": str(start_char + len(segment_text) - 1)
      })
      original_text = ET.SubElement(seg, "ORIGINAL_TEXT") 
      original_text.text = segment_text
      for i, word in enumerate(words):
         token = ET.SubElement(seg, "TOKEN", {
            "pos" : "word",
            "morph" : "none",
            "id": utt_id + "_" + str(i),
            "start_char": str(start_char),
            "end_char": str(start_char + len(word) - 1)
         })
         token.text = word
         start_char = start_char + len(word) + 1
      with open(outfile, 'w') as fout:
         fout.write(minidom.parseString(ET.tostring(lctl_text)).toprettyxml(indent="  ", encoding='UTF-8').decode())


def main():
   args = get_args()
   with open(args.transcript_file) as fin:
      lines = fin.readlines()
   utterances = defaultdict(lambda: defaultdict(dict))
   for line in lines:
      fields = line.split(None, 2)
      if len(fields) != 3:
         continue
      utt_id, lang, transcript = fields
      lang = normalize_lang(lang)
      if not lang:
         continue
      doc_id = utt_id.rsplit('_', 1)[0].upper()
      utterances[lang][doc_id][utt_id.upper()] = transcript.strip()
   for lang in utterances:
       if not os.path.exists(os.path.join(args.output_ltf_dir, lang)):
         os.makedirs(os.path.join(args.output_ltf_dir, lang))
       for doc_id in utterances[lang]:
         outfile = os.path.join(args.output_ltf_dir, lang, doc_id + ".ltf.xml")
         write_ltf(outfile, lang, doc_id, utterances[lang][doc_id])


if __name__ == "__main__":
   main()
