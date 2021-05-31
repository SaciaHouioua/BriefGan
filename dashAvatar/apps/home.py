#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 16:31:30 2021

@author: sacia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:06:37 2020
@author: randon
"""

import plotly.express as px
import plotly.graph_objs as go
from tensorflow import keras
import tensorflow as tf



import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from app import app, server
import dash
import numpy as np

from app import app, server

from apps import home  #, page1

batch_size = 1
latent_dim = 128
generator = keras.models.load_model("generator.h5")
generator.compile()


layout = html.Div([
    dbc.Container([
        
        dbc.Row([
            dbc.Col(html.H1("Welcome to Avatar Generator", className="text-center")
                    , className="mb-4 mt-4")
        ]),
        
             
        
        dbc.Row([
            dbc.Col(html.H4(children='Avatar Generator'
                                     ))
            ]),
        dbc.Row([
            
           dbc.Col(html.H5(children='You can click on the button to generate a new avatar :')                     
                    , className="mb-4")
            ]),
        
        
        dbc.Button("Click me", id="avatar-click", className="mr-2"),
        
           
        html.Span(id="avatar-output", style={"vertical-align": "middle"}),
       
     
        
        ])])

@app.callback(
    Output("avatar-output", "children"), [Input("avatar-click", "n_clicks")]
)
def on_button_click(n):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'avatar-click' in changed_id:

        image_random = np.random.normal(size=(batch_size,latent_dim))
        image_genereted = generator.predict(image_random)
        fig = go.Figure(px.imshow((image_genereted * 255)[0].astype(np.uint8)))
        #hide axis
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        return html.Div(dcc.Graph(
        id='Avatar', figure = fig))
    else:
        msg = 'None of the buttons have been clicked yet'
    return html.Div(msg)
        
        
   
