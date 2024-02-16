from collections import Counter
from operator import itemgetter

import streamlit as st
import pandas as pd
import altair as alt

import spaCy_model

example = "Sebastian Thrun started working on self-driving cars at Google in 2007."


# st.set_page_config(layout='wide')
def ner():
    st.markdown('## spaCy Named Entity Recognition')

    text = st.text_area('Text to process', value=example, height=100)

    doc = spaCy_model.NerClass(text)

    entities = doc.get_entities()
    tokens = doc.get_tokens()
    counter = Counter(tokens)
    words = list(sorted(counter.most_common(30)))

    # https://pandas.pydata.org
    chart = pd.DataFrame({
        'frequency': [w[1] for w in words],
        'word': [w[0] for w in words]})

    # https://pypi.org/project/altair/
    bar_chart = alt.Chart(chart).mark_bar().encode(x='word', y='frequency')

    st.markdown(f'Total number of tokens: {len(tokens)}<br/>'
                f'Total number of types: {len(counter)}', unsafe_allow_html=True)

    # https://docs.streamlit.io/library/api-reference/data/st.table
    st.table(entities)

    # https://docs.streamlit.io/library/api-reference/charts/st.altair_chart
    st.altair_chart(bar_chart)


def dep():
    st.markdown('## spaCy Dependency Parsing')
    text = st.text_area('Text to process', value=example, height=100)

    selection = st.radio("Select Display Option", ("Table", "Graph"))

    dep = spaCy_model.DepClass(example)
    dep_l = dep.process_data()

    if selection == "Table":
        column_names = ['Root Text', 'Root Dependency', 'Head Text']
        df = pd.DataFrame(dep_l, columns=column_names)

        st.table(df)

    elif selection == "Graph":
        dep_graph = dep.dep_graph()
        css_style = """
        <style>
        .graph-container {
            display: inline-block;
            vertical-align: top; /* Adjust as needed */
            margin-right: auto; /* Move to leftmost position */
        }
        </style>
        """
        st.markdown(css_style, unsafe_allow_html=True)
        st.image(dep_graph, use_column_width=True, output_format="PNG", caption="Dependency Graph")


option = st.sidebar.selectbox(
    'Please select the spaCy model',
    ('Named Entity Recognition', 'Dependency Parsing')
)

if option == "Named Entity Recognition":
    ner()
elif option == "Dependency Parsing":
    dep()
