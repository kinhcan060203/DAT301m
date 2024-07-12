
import config as args
from models.image_to_text import Image2Text
from models.check_spell import CheckSpell
import re
import json
import numpy as np


def extract_phrases(text):
    text =text.split("Câu")
    if len(text)>1 and ':' in text[1]:
        text = 'Câu' + ''.join(text[1:])
    else:
        text = ''.join(text[:])
    for i in ['A.',"B.","C.","D."]:
        if i in text:
            text = i + ''.join(text.split(i)[1:])
            break
    return re.findall(r'[\w ]+', text)



if __name__ == '__main__': 
    checkspell = CheckSpell(args.SPELL_CONFIG)
    img2text = Image2Text(args.CRAFT_CONFIG)
    test_img = 'data/de_1.jpg' 
    raw_text = img2text(test_img)
    with open("demo_result/raw_text.json", "w", encoding='utf8') as f:
        json.dump(raw_text, f, indent=4, ensure_ascii=False)
    extracted = [extract_phrases(sentences) for sentences in raw_text]
     
    phrases = [[(p.strip()).strip(' \x00 ') for p in phrase if len(p.split()) >= 1] for phrase in extracted]
    with open("demo_result/final_result.json", "w", encoding='utf8') as f:
        json.dump(phrases, f, indent=4, ensure_ascii=False)

