from flask import Flask, render_template, request
from newspaper import Article

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            article = Article(request.values.get('news_url'))
        except:
            return render_template("index.html", error="Error: Failed to fetch the news article.")
        article.download()
        article.parse()

        if request.values.get('technique') == "abstractive":
            news_summary = abs_summarize(article.text)
            news_summary = news_summary[5:]
            news_summary = news_summary[:-4]
            return render_template("index.html", title=article.title, content=news_summary, sumType="Abstractive")
        elif request.values.get('technique') == "extractive":
            news_summary = ext_summarize(article.text, 0.1)
            print(news_summary)
            return render_template("index.html", title=article.title, content=news_summary, sumType="Extractive")
    return render_template("index.html")

@app.route('/about.html')
def about():
    return render_template("about.html")

@app.route('/login.html')
def login():
    return render_template("login.html")

@app.route('/help.html')
def help():
    return render_template("help.html")

def abs_summarize(text):
    tokenizer_model = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    loaded_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
    tokens = tokenizer_model(text, truncation=True, padding="longest", return_tensors="pt")
    summary = loaded_model.generate(**tokens)
    return tokenizer_model.decode(summary[0])

def ext_summarize(text, per):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    tokens = [token.text for token in doc]
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency
    sentence_tokens = [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
    select_length = int(len(sentence_tokens) * per)
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    final_summary = [word.text for word in summary]
    summary = ''.join(final_summary)
    return summary

if __name__ == '__main__':
    app.run(debug=True)