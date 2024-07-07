import config as args
from models.image_to_text import CONCHOCON
from models.check_spell import CheckSpell
import gradio as gr
import re
import json
from PIL import Image

def extract_phrases(text):
    return re.findall(r'[\w ]+', text)

def out(data):
    try:
        ques = [' '.join(data[:data.index('A')])]
        ans = data[data.index('A'):]
        return ques + ans
    except ValueError:
        return [' '.join(data)]

def rewrite(phrases):
    questions = {}
    current_question = None

    for item in phrases:
        if item[0].startswith('Câu'):
            if current_question:
                questions[current_question['id']] = out(current_question['text']) #{
            current_question = {'id': item[0], 'text': item[1:]}
        else:
            if current_question:
                current_question['text'].extend(item)

    if current_question:
        questions[current_question['id']] = out(current_question['text'])
    return questions

def greet(image):
    
    img = Image.fromarray(image)
    path="demo_result/save_img.jpg"
    img.save(path)
    
    # model vs image
    checkspell = CheckSpell(args.SPELL_CONFIG)
    img2text = CONCHOCON(args.CRAFT_CONFIG)
    test_img = path
    raw_text = img2text(test_img)
    
    with open("demo_result/raw_text_2.json", "w", encoding='utf8') as f:
        json.dump(raw_text, f, indent=4, ensure_ascii=False)
    extracted = [extract_phrases(sentences) for sentences in raw_text]
    
    phrases = [[checkspell(p.strip()).strip(' \x00 ') for p in phrase if len(p.split()) >= 1] for phrase in extracted] #checked spelling
    with open("demo_result/final_result_2.json", "w", encoding='utf8') as f:
        json.dump(phrases, f, indent=4, ensure_ascii=False)
    
    question = rewrite(phrases)
    # with open("demo_result/final_result_2.json", "w", encoding='utf8') as f:
    #     json.dump(question, f, indent=4, ensure_ascii=False)
        
        
    # final_result = 'demo_result/final_result_2.json'
    # with open(final_result, 'r', encoding='utf8') as f: 
    #     data = json.load(f)
    
    ##########################################################################
    
    html_content = ""
    for k, v in question.items():
        html_content += f"<h3>{k}</h3><p>{'<br>'.join(v)}</p>"
        
    return html_content

# with gr.Blocks() as demo:
#     image_input = gr.Image()
#     greet(image_input)

demo=gr.Interface(
    fn=greet,
    inputs=gr.Image(),
    outputs=gr.HTML(label="Extracted Questions and Answers")
    #outputs=gr.Textbox()
)

if __name__ == "__main__":
   demo.launch()