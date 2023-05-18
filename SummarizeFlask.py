

from flask import Flask, render_template, request, redirect,url_for
import requests
from bs4 import BeautifulSoup
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import sentencepiece

def summarize(url):
    # make a GET request to the URL and extract the text
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))

    # initialize the T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    # prepare the input for the model
    input_text = 'summarize: ' + text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # generate the summary
    summary_ids = model.generate(input_ids, num_beams=4, max_length=100, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

app = Flask("_name__")

@app.route("/", methods=['POST', 'GET'])




def main():
    if request.method == 'POST':
        URL_input= request.form['URLtxt']
        url=URL_input
        summary=summarize(url)
        return render_template('index.html', summary=summary)
        
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run()