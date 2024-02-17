"""spaCy_model.py

Run spaCy NER over an input string and insert XML tags for each entity.

"""
import io
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")


class NerClass:

    def __init__(self, text: str):
        self.text = text
        self.doc = nlp(text)

    def get_tokens(self) -> list:
        return [token.lemma_ for token in self.doc]

    def get_entities(self) -> str:
        entities = []
        for e in self.doc.ents:
            entities.append((e.start_char, e.end_char, e.label_, e.text))
        return entities

    def get_entities_with_markup(self) -> str:
        entities = self.doc.ents
        starts = {e.start_char: e.label_ for e in entities}
        ends = {e.end_char: True for e in entities}
        buffer = io.StringIO()
        for p, char in enumerate(self.text):
            if p in ends:
                buffer.write('</entity>')
            if p in starts:
                buffer.write('<entity class="%s">' % starts[p])
            buffer.write(char)
        markup = buffer.getvalue()
        return '<markup>%s</markup>' % markup


class DepClass:

    def __init__(self, text: str):
        self.text = text
        self.doc = nlp(text)

    def process_data(self):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(self.text)
        dep_l = []
        for chunk in doc.noun_chunks:
            dep_l.append((chunk.root.text, chunk.root.dep_, chunk.root.head.text))
        return dep_l

    def dep_graph(self):
        options = {
            'bg': '#ffffff',  # Background color
            'color': '#000000',  # Text color
            'font': 'Arial',  # Font family
            'distance': 120  # Distance between words
        }
        doc = nlp(self.text)
        dep_graph = displacy.render(doc, style="dep", jupyter=False)
        dep_graph = dep_graph.replace('widthL initial', 'width: 50%')
        dep_graph = dep_graph.replace('height: initial', 'height: 50%')
        return dep_graph

    def get_dep_parse(self):
        dependency_parse = []
        for token in self.doc:
            dependency_parse.append((token.text, token.dep_, token.head.text))
        return dependency_parse


if __name__ == '__main__':
    example = "Here is just a simple example to show the ner model results: " \
              "I did not go to school today because of the " \
              "predicted snow storm, but it does not snow heavily, which is disappointing."

    # doc = NerClass(example)
    # print(doc.get_tokens())
    # for entity in doc.get_entities():
    #     print(entity)
    # print(doc.get_entities_with_markup())
    dep = DepClass(example)
    dep.dep_graph()
