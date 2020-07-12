
import dash
import dash_core_components as dcc
import tensorflow as tf
import random
import dash_bootstrap_components as dbc
import dash_html_components as html
import pandas as pd
import pathlib
import plotly.express as px
import h5py
import numpy as np
import plotly.io as pio
from dash.dependencies import Input, Output, State
import dash_colorscales as dcs
import json


pio.templates.default = "plotly_dark"
# app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app = dash.Dash(external_stylesheets=[dbc.themes.CYBORG],
    external_scripts=['https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML',]
)


def NamedSlider(name, short, min, max, step, val, marks=None):
    if marks:
        step = None
    else:
        marks = {i: i for i in range(min, max + 1, step)}

    return html.Div(
        style={"margin": "15px 0px 0px 0px"},
        children=[
            # f"{name}:",
            html.Div(
                # style={"margin-left": "5px"},
                children=[
                    dcc.Slider(
                        id=f"slider-{short}",
                        min=min,
                        max=max,
                        marks=marks,
                        step=step,
                        value=val,
                    )
                ],
            ),
        ],
    )




navbar = dbc.NavbarSimple(
    children=[
        # dbc.NavItem(dbc.NavLink("Page 1", href="#")),
        dbc.NavItem(dcc.Dropdown(
                                        id="expt_name",
                                        searchable=False,
                                        clearable=False,
                                        options=[
                                            {
                                                "label": "Marmousi Example I",
                                                "value": "planewave_marm",
                                            },
                                            {
                                                "label": "Marmousi Example II",
                                                "value": "twitter_3000",
                                            },
                                            {
                                                "label": "Simple Expanding Perturbation I",
                                                "value": "simple1",
                                            },
                                            {
                                                "label": "Simple Expanding Perturbation II",
                                                "value": "simple2",
                                            },
                                        ],
                                        placeholder="Select a dataset",
                                        value="planewave_marm",
                                    ), style={'width': "40vh"}),
        dbc.DropdownMenu(
            children=[
                
                dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Page 2", href="#"),
                dbc.DropdownMenuItem("Page 3", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="SymAE",
    brand_href="#",
    color="primary",
    dark=True,
)

card_content = [
    dbc.CardHeader("Card heqefafwrfder"),
    dbc.CardBody(
        [
            html.H5("Card tiwfwrftle", className="card-title"),
            html.P(
                "This is some card content that we'll reuse",
                className="card-text",
            ),
        ]
    ),
]

card_encoder = [ dbc.CardBody( [ html.Img(src=app.get_asset_url('encoder.png'), style={"width": "7.5vh"}) ]), ]
card_decoder = [ dbc.CardBody( [ html.Img(src=app.get_asset_url('decoder.png'), style={"width": "7.5vh"}) ]), ]

card_medium1 = [
    dbc.CardHeader("Choose Baseline Medium"),
    dbc.CardBody(
        [
            dcc.Graph(id="fig_medium1", style={"height": "25vh", "width": "40vh"},),
            NamedSlider( name="Baseline Medium", short="medium1", min=0, max=2, step=None, val=0,
                                        marks={
                                            0: '0',
                                            1: '1',
                                            2: '2' 
                                        },),
        ]
    ),
]

card_medium2 = [
    dbc.CardHeader("Choose Monitor Medium"),
    dbc.CardBody(
        [
            dcc.Graph(id="fig_medium2", style={"height": "25vh", "width": "40vh"},),
            NamedSlider( name="Monitor Medium", short="medium2", min=0, max=2, step=None, val=2,
                                        marks={
                                            0: '0',
                                            1: '1',
                                            2: '2' 
                                        },),
        ]
    ),
]

card_Xi = [
    dbc.CardHeader(
        dbc.Tabs(
                [
                    dbc.Tab(label=r"$X_i$ Plot", tab_id="plot"),
                    dbc.Tab(label=r"$X_i$ Info", tab_id="info"),
                ],
                id="Xi_tabs",
                card=True,
                active_tab="plot",
            )
        ),
    dbc.CardBody(
        [
        html.Div(id="fig_Xi", style={"height": "28vh", "width": "23vh", 'marginTop': '0em',}), 
        dbc.Button('next instance', id='reload_i', n_clicks=1, className='mr-2',),
        html.Div(id='div-hidden_i', style={'display': 'none', 'marginBottom': '10em' })
        ]
    ),
]


card_Xj = [
    dbc.CardHeader(
        dbc.Tabs(
                [
                    dbc.Tab(label=r"$X_j$ Plot", tab_id="plot"),
                    dbc.Tab(label=r"$X_j$ Info", tab_id="info"),
                ],
                id="Xj_tabs",
                card=True,
                active_tab="plot",
            )
        ),
    dbc.CardBody(
        [
        html.Div(id="fig_Xj", style={"height": "28vh", "width": "23vh", 'marginTop': '0em',}), 
        dbc.Button('next instance', id='reload_j', n_clicks=1, className='mr-2',),
        html.Div(id='div-hidden_j', style={'display': 'none', 'marginBottom': '10em' })
        ]
    ),
]

card_Xihat = [
    dbc.CardHeader(
        dbc.Tabs(
                [
                    dbc.Tab(label=r"$\hat{X}_i$ Plot", tab_id="obs"),
                    dbc.Tab(label=r"$\hat{X}_{j,i}$ Plot", tab_id="hybrid"),
                ],
                id="Xihat_tabs",
                card=True,
                active_tab="obs",
            )
        ),
    dbc.CardBody(
        [
        dcc.Graph(id="fig_Xihat", style={"height": "28vh", "width": "23vh", 'marginTop': '0em',}), 
        dcc.RadioItems(
            id="toggle_hat_i",
            options=[
                {"label": r'$\hat{X}$ Plot', "value": "HAT"},
                {"label": r'$\hat{X}-{X}$ Plot', "value": "DIFF", },
            ],
            # style={'marginRight': '10em'},
            value="HAT",
            labelStyle={
                "display": "inline-block",
                "padding": "12px 12px 12px 0px",
            },
                ),
        ]
    ),
]

card_Xjhat = [
    dbc.CardHeader(
            dbc.Tabs(
                [
                    dbc.Tab(label=r"$\hat{X}_j$", tab_id="obs"),
                    dbc.Tab(label=r"$\hat{X}_{i,j}$", tab_id="hybrid"),
                ],
                id="Xjhat_tabs",
                card=True,
                active_tab="obs",
            )
            ),
    dbc.CardBody(
        [
        dcc.Graph(id="fig_Xjhat", style={"height": "28vh", "width": "23vh", 'marginTop': '0em',}), 
        dcc.RadioItems(
            id="toggle_hat_j",
            options=[
                {"label": r'$\hat{X}$', "value": "HAT"},
                {"label": r'$\hat{X}-{X}$', "value": "DIFF", },
            ],
            # style={'marginRight': '10em'},
            value="HAT",
            labelStyle={
                "display": "inline-block",
                "padding": "12px 12px 12px 0px",
            },
                ),
        ]
    ),
]

card_latent1 = [
    dbc.CardHeader("SymAE's Structured Latent Space"),
    dbc.CardBody(
        [
            dcc.Graph(id="fig_latent_space1", style={"height": "25vh", "width": "25vh", 'marginTop': '0em',}), 
            # html.P(latent_instruction, style={"width":"25vh"})
                # "This is some card content that we'll reuse",
                # className="card-text",
            # ),
        ]
    ),
]

card_latent2 = [
    dbc.CardHeader("SymAE's Structured Latent Space"),
    dbc.CardBody(
        [
            dcc.Graph(id="fig_latent_space2", style={"height": "25vh", "width": "25vh", 'marginTop': '0em',}), 
            # html.P(latent_instruction, style={"width":"25vh"})
                # "This is some card content that we'll reuse",
                # className="card-text",
            # ),
        ]
    ),
]


row_1 = dbc.Row(
    [
        dbc.Col(dbc.Card(card_medium1, color="dark", outline=True),width="auto"),
        dbc.Col(dbc.Card(card_Xi, color="dark", outline=True), width="auto"),
        dbc.Col(dbc.Card(card_encoder, color="light", outline=True), width="auto"),
        dbc.Col(dbc.Card(card_latent1, color="dark", outline=True),width="auto"),
        dbc.Col(dbc.Card(card_decoder, color="light", outline=True), width="auto"),
        dbc.Col(dbc.Card(card_Xihat, color="dark", outline=True),width="auto"),
    ],
    className="mb-4",
    align="center",
    justify="center",
)

row_2 = dbc.Row(
    [
        dbc.Col(dbc.Card(card_medium2, color="dark", outline=True),width="auto"),
        dbc.Col(dbc.Card(card_Xj, color="dark", outline=True), width="auto"),
        dbc.Col(dbc.Card(card_encoder, color="light", outline=True), width="auto"),
        dbc.Col(dbc.Card(card_latent2, color="dark", outline=True),width="auto"),
        dbc.Col(dbc.Card(card_decoder, color="light", outline=True), width="auto"),
        dbc.Col(dbc.Card(card_Xjhat, color="dark", outline=True),width="auto"),
    ],
    className="mb-4",
    align="center",
    justify="center",
)

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

with open(PATH.joinpath("demo_description.md"), "r") as file:
    demo_description_md = file.read()


data={}
expts=["planewave_marm"]
for expt in expts:
    DATA_PATH= PATH.joinpath(expt).resolve()
    mh5=h5py.File(DATA_PATH.joinpath("medium.h5"), 'r')
    m={}
    for key in mh5.keys():
        m[key]=np.array(mh5[key])
    X=h5py.File(DATA_PATH.joinpath("Xotestdash.h5"), 'r')
    Xot=np.array(X["data"][:,:,:,:,:])
    nsamp, ntau, nr, nt, nfield=Xot.shape
    with open(DATA_PATH.joinpath('Xotestdash.json')) as f:
        labels = json.load(f)
    labelindices={'base0': [], 'moni1' : [], 'moni2' : []}
    for isamp in range(nsamp):
        labelindices[labels[str(isamp+1)]["1"]["Medium"]].append(isamp)

    encoderr=tf.keras.models.load_model(DATA_PATH.joinpath("_TFencoderr"), compile=False)
    encodertau=tf.keras.models.load_model(DATA_PATH.joinpath("_TFencodertau"), compile=False)
    decoder=tf.keras.models.load_model(DATA_PATH.joinpath("_TFdecoder"), compile=False)

    data[expt] = {
    'Xot': Xot, 'm': m, 'labels': labels, 'labelindices': labelindices, 
    'encoderr': encoderr, 
    'encodertau': encodertau, 
    'decoder': decoder
    }
#


app.layout = html.Div([
    dbc.Container(html.Div([navbar,row_1,row_2]), fluid=True),
            dcs.DashColorscales(
                                    id="colorscale-picker",
                                     colorscale="RdBu",)])
    #, className='w-60')


def generate_medium(epsilon,expt):
    eps=["base0", "moni1", "moni2"][epsilon]
    m0=data[expt]['m'][eps]
    fig=px.imshow(m0.transpose(),  color_continuous_scale='Viridis',)
    fig.update_layout(margin={"l":0,"r":0,"b":0,"t":0})
    return fig
                # labels=dict(x="Day of Week", y="Time of Day", color="Productivity"),
                # x=['Monday', 'Thursday', 'Friday'],
                # y=['Morning', 'Afternoon', 'Evening']

def generate_gather(d,cs):
    fig=px.imshow(d.transpose(), aspect="auto", range_color=[-3,3], color_continuous_scale=cs,)
    fig.update_yaxes(title='time')
    fig.update_layout(margin={"l":0,"r":0,"b":0,"t":0})
    fig.update_xaxes(title='receiver')
    return fig




@app.callback(
        Output("fig_medium1", "figure"),
        [
            Input("slider-medium1", "value"),
            Input("expt_name", "value"),
        ],
)
def update_baseline(epsilon,expt):
    return generate_medium(epsilon,expt)

@app.callback(
        Output("fig_medium2", "figure"),
        [
            Input("slider-medium2", "value"),
            Input("expt_name", "value"),
        ],
)
def update_monitor(epsilon,expt):
    return generate_medium(epsilon,expt)

@app.callback(
        Output("div-hidden_i", "children"),
        [
            Input("reload_i", "n_clicks"),
            Input("slider-medium1", "value"),
            Input("expt_name", "value"),
        ],
)
def load_Xi(dummy,epsilon,expt):
    eps=["base0", "moni1", "moni2"][epsilon]
    return random.choice(data[expt]["labelindices"][eps])

@app.callback(
        Output("div-hidden_j", "children"),
        [
            Input("reload_j", "n_clicks"),
            Input("slider-medium2", "value"),
            Input("expt_name", "value"),
        ],
)
def load_Xj(dummy,epsilon,expt):
    eps=["base0", "moni1", "moni2"][epsilon]
    return random.choice(data[expt]["labelindices"][eps])

def reconstruct(i,j,expt,toggle,hybrid_flag):
    x=data[expt]["Xot"][[i,j]]
    zr=data[expt]["encoderr"].predict(x)
    ztau=data[expt]["encodertau"].predict(x)
    if(hybrid_flag):
        zr=np.flip(zr, axis=[0]);
    z=tf.concat([zr, ztau],1)
    xhat=data[expt]["decoder"].predict(z)
    if (toggle=="DIFF"):
        x=x[0,0,:,:,0]
        xhat=xhat[0,0,:,:,0]-x
    else:
        xhat=xhat[0,0,:,:,0]

    return xhat
 


@app.callback(
        Output("fig_Xi", "children"),
        [
            Input("div-hidden_i", "children"),
            Input("expt_name", "value"),
            Input("colorscale-picker", "colorscale"),
            Input("Xi_tabs", "active_tab")
        ],
)
def plot_Xi(i,expt,cs,tab):
        if(tab=="plot"):
            d=data[expt]["Xot"][i,0,:,:,0]
            return dcc.Graph(figure=generate_gather(d,cs),  style={"height": "28vh", "width": "23vh", 'marginTop': '0em',})
        else:
            return html.P(json.dumps(labels[str(i+1)]["1"]))

@app.callback(
        Output("fig_Xihat", "figure"),
        [
            Input("div-hidden_i", "children"),
            Input("div-hidden_j", "children"),
            Input("expt_name", "value"),
            Input("colorscale-picker", "colorscale"),
            Input("toggle_hat_i", "value"),
            Input("Xihat_tabs", "active_tab"),
        ],
)
def plot_Xihat(i,j,expt,cs,toggle,tab):
    if(tab == "obs"):
        hybrid_flag=False
    else:
        hybrid_flag=True
    return generate_gather(reconstruct(i,j,expt,toggle,hybrid_flag),cs)


 
@app.callback(
        Output("fig_Xj", "children"),
        [
            Input("div-hidden_j", "children"),
            Input("expt_name", "value"),
            Input("colorscale-picker", "colorscale"),
            Input("Xj_tabs", "active_tab")
        ],
)
def plot_Xj(j,expt,cs,tab):
    if(tab=="plot"):
        d=data[expt]["Xot"][j,0,:,:,0]
        return dcc.Graph(figure=generate_gather(d,cs),  style={"height": "28vh", "width": "23vh", 'marginTop': '0em',})
    else:
        return html.P(json.dumps(labels[str(j+1)]["1"]))

@app.callback(
        Output("fig_Xjhat", "figure"),
        [
            Input("div-hidden_i", "children"),
            Input("div-hidden_j", "children"),
            Input("expt_name", "value"),
            Input("colorscale-picker", "colorscale"),
            Input("toggle_hat_j", "value"),
            Input("Xjhat_tabs", "active_tab"),
        ],
)
def plot_Xjhat(i,j,expt,cs,toggle,tab):
    if(tab == "obs"):
        hybrid_flag=False
    else:
        hybrid_flag=True
    return generate_gather(reconstruct(j,i,expt,toggle,hybrid_flag),cs)
 

def latent_scatter(m1,m2, ticktext):
    hoverlabels = ["Observed Instance", "Hybrid Instance"]
    fig=px.scatter(x=[m1,m1], y=[m1,m2], range_x=[-1,3], range_y=[-1,3], hover_name=hoverlabels)
    fig.update_xaxes(title="source effects")
    fig.update_yaxes(title="path effects")
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=[0,1,2],ticktext=["0", "1", "2"]),hovermode='closest')
    fig.update_layout(yaxis=dict(tickmode='array', tickvals=[0,1,2],ticktext=["0","1","2"]))
    fig.update_layout(margin={"l":0,"r":0,"b":0,"t":0})
    fig.update_traces(marker=dict(size=20,symbol=["diamond", "bowtie"], color="blue", line_color="black"))
    # fig.update_traces(hovertemplate='GDP: %{customdata}')
    return fig

@app.callback(
         Output("fig_latent_space1", "figure"),
         [
            Input("slider-medium1", "value"),
            Input("slider-medium2", "value"),
         ]
 )
def plot_latent_space1(m1,m2):
    return latent_scatter(m1,m2, ["Baseline", "Monitor"])
 
@app.callback(
         Output("fig_latent_space2", "figure"),
         [
            Input("slider-medium1", "value"),
            Input("slider-medium2", "value"),
         ]
 )
def plot_latent_space2(m2,m1):
    return latent_scatter(m1,m2, ["Monitor", "Baseline"])
 


if __name__ == "__main__":
    app.run_server(debug=True)