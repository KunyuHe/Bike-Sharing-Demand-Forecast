from pathlib import Path

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output

import utils

INPUT_DIR = Path("../data/clean/")
CONFIG_DIR = Path("../configs/")

df = utils.readDF(INPUT_DIR)
num, cat = utils.getNumCat(df, target='cnt')
num.append('cnt')

type_var = {'num': num,
            'cat': cat}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(
        children="Bike Sharing Demand Forecast EDA",
        style={'textAlign': 'center',
               'color': "#00245D"}
    ),

    html.Div(children=[
        html.Label("Variable Type"),
        dcc.Dropdown(
            id="Variable Type",
            options=[
                {'label': "Numerical", 'value': "num"},
                {'label': "Categorical", 'value': "cat"}
            ],
            value="num"
        ),

        html.Label("Variable Name"),
        dcc.Dropdown(id="Variable Name")
    ]),

    html.Div(children=[
        dcc.Graph(id='Distribution'),
        dcc.Graph(id='Relation')
    ])
])

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


@app.callback(
    Output("Variable Name", 'options'),
    [Input("Variable Type", 'value')]
)
def setVarLst(type_):
    return [{'label': var, 'value': var} for var in type_var[type_]]


@app.callback(
    Output("Variable Name", 'value'),
    [Input("Variable Name", 'options')]
)
def setVar(options):
    return options[0]['value']


@app.callback(
    Output("Distribution", 'figure'),
    [Input("Variable Type", 'value'), Input("Variable Name", 'value')]
)
def plot_dist(type_, var):
    if type_ == "num":
        fig = px.histogram(df, x=var,
                           title=f"Distribution of {var}")
        return fig
    fig = px.bar(df, x=var,
                 title=f"Distribution of {var}")
    return fig


@app.callback(
    Output("Relation", 'figure'),
    [Input("Variable Type", 'value'), Input("Variable Name", 'value')]
)
def plot_relation(type_, var):
    if var == 'cnt':
        return

    if type_ == "num":
        fig = px.scatter(df, x=var, y='cnt')
        return fig

    fig = px.box(df, x=var, y="cnt")
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
