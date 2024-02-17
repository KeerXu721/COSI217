# Assignment 1 - Web Services 🌐

This assignment contains three part:

1. Create a RESTful API to access spaCy NER and spaCy dependency parsing
2. Create a Flask webserver to access spacy NER  and spaCy dependency parsing
3. Create a Streamlit application to access and spaCy dependency parsing

And in order to meet the requirement of making the API and websites independent, 
I created another part for spaCy model access. Each part is explained below:

### spaCy model access 🔑
All the spaCy access are contained in the spaCy_model.py file. This code contains two class: NerClass, and DepClass,
and they correspond to the Named Entity Recognition parsing and the Dependency parsing. 

#### Requirements if you want to run the code in this file:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```
### RESTful API with FastAPI 🔗
To run this section, open a terminal, and run:
```bash
uvicorn app_fastapi:app --reload
```

To access the NER:
```bash
curl http:/127.0.0.1:8000
curl -X POST http://127.0.0.1:8000/ner \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d@input.json
```

To access the dependency parser:
```bash
curl http:/127.0.0.1:8000
curl -X POST http://127.0.0.1:8000/dep \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d@input.json
```
### Flask webserver 🖥️
This part is in the app_flask.py file. 

#### Requirements if you want to run the code in this file:
```bash
pip install Flask
```

This is a simple web interface that allows users to input some texts, submit, and then display
the parsed Named Entity Recognition results and dependency parsing results. 

To run the code, simply run the *app_flask.py* file, and click the link: 
[[http://127.0.0.1:5000] ]([http://127.0.0.1:5000]), then you will be directed to the webpage. 

### Streamlit 🎬
#### Requirements if you want to run the code in this file:
```bash
pip install streamlit graphviz
```
In this section, I created a small Streamlit application that shows the result of spaCy NER and dependency parsing. 
It contains a side bar, which you can click open and close on the upper left corner. You can choose either the NER or
the dependency parsing result. Then you will be directed to the corresponding web page, where you can type in text to 
process. In the dependency parsing page, you can choose table or graph to display the parsed results. 

#### To run this application:
```bash
streamlit run app_streamlit.py
```

