"""Simple Web interface to spaCy entity recognition

To see the pages point your browser at http://127.0.0.1:5000.

"""

from flask import Flask, request, render_template

import spaCy_model

app = Flask(__name__)


# For the website we use the regular Flask functionality and serve up HTML pages.

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', input=open('sample_text.txt', encoding="utf-8").read())
    else:
        text = request.form['text']
        ner = spaCy_model.NerClass(text)
        dep = spaCy_model.DepClass(text)
        dep_l = dep.process_data()
        markup = ner.get_entities_with_markup()
        markup_paragraphed = ''
        for line in markup.split('\n'):
            if line.strip() == '':
                markup_paragraphed += '<p/>\n'
            else:
                markup_paragraphed += line
        return render_template('result.html', markup=markup_paragraphed)


# alternative where we use two resources

@app.get('/get')
def index_get():
    return render_template('index.html', input=open('sample_text.txt').read())


@app.post('/post')
def index_post():
    text = request.form['text']
    ner_doc = spaCy_model.NerClass(text)
    entity_markup = ner_doc.get_entities_with_markup()
    dep_doc = spaCy_model.DepClass(text)
    dep_l = dep_doc.process_data()
    dep_graph = dep_doc.dep_graph()
    ner_parse = ''
    dep_parse = ''
    for line in entity_markup.split('\n'):
        if line.strip() == '':
            ner_parse += '<p/>\n'
        else:
            ner_parse += line
    for item in dep_l:
        dep_parse += '<p style="text-align: center;">' + item[0] + " " + item[1] + " " + item[2] + '</p>\n'
    return render_template('result2.html', ner=ner_parse, dep=dep_parse, dep_graph=dep_graph)


if __name__ == '__main__':
    app.run(debug=True)