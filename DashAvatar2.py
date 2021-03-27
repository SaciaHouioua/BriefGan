# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
# import
#dieu
import numpy as np
import pandas as pd
import string
from collections import defaultdict
from time import time
# Sklearn
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer , TfidfTransformer
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import recall_score , precision_score ,f1_score
from sklearn import metrics , svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB, CategoricalNB, ComplementNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC
#vis
# import matplotlib
# import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objs as go
import plotly as py
from plotly.offline import init_notebook_mode, iplot, plot
#NLP ISSUE WITH HEROKU
# # # import nltk   
# # # from nltk.corpus import stopwords , wordnet as wn    
# # # from nltk import wordpunct_tokenize , WordNetLemmatizer ,sent_tokenize ,  word_tokenize
# # # from nltk.stem import PorterStemmer , LancasterStemmer
# # # from nltk.stem.snowball import SnowballStemmer  
#Dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
from tensorflow import keras
import tensorflow as tf




import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
import plotly.graph_objs as go
from dash.dependencies import Input, Output


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

batch_size = 1
latent_dim = 128
generator = keras.models.load_model("generator.h5")
generator.compile()




app.layout = html.Div(children=[
    html.H1(children='Image Avatar'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),
    html.Button('Button 1', id='btn-nclicks-1', n_clicks=0),
    html.Div(id='container-button-timestamp'),
    
])

@app.callback(Output('container-button-timestamp', 'children'),
              Input('btn-nclicks-1', 'n_clicks'))

def displayClick(btn1):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-nclicks-1' in changed_id:

        image_random = np.random.normal(size=(batch_size,latent_dim))
        image_genereted = generator.predict(image_random)
        fig = go.Figure(px.imshow((image_genereted * 255)[0].astype(np.uint8)))
        
        return html.Div(dcc.Graph(
        id='Avatar', figure = fig))
    else:
        msg = 'None of the buttons have been clicked yet'
    return html.Div(msg)
    
if __name__ == '__main__':
    app.run_server(debug=True)
