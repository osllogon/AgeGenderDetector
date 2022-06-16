# machine learning imports
import os
import tempfile

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

# other imports
import pathlib
import base64

# visualiation imports
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

# own imports
from models.train import predict_age_gender
from models.models import CNNClassifier, load_model_from_name
from models.utils import load_list, load_dict, LABEL_GENDER
from .visualize import visualize_datasets_age, visualize_datasets_gender, \
    get_datasets, visualize_model, OriginalSaliencyMap, HeatMap, SaliencyMapCombined, IntegratedSaliencyMap

# Variables globales
MODELS_PATH = 'models/savedAgeGender'
DATA_PATH = "./data/UTKFace"
# if true, heat maps will be showed
SHOW_HEAT_MAPS = False
# if true, only the prediction to the uploaded image will be showed (this overwrites SHOW_HEAT_MAPS)
NO_MAPS = False
# if true it will try to use the GPU (faster)
USE_GPU = False
device = torch.device('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu')

curr_model = {}

# Inicializamos la aplicacion de dash
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

TITLE_STYLE = {
    "text-align": "center",
    "text-decoration": "bold",
    "margin": "30px"
}

# Get figures
datasets = get_datasets(DATA_PATH)
figures_dist_age = visualize_datasets_age(datasets)
figures_dist_gender = visualize_datasets_gender(datasets)

best_models = load_list(f'{MODELS_PATH}/bestModels.txt')
options_models = {
    k.name[:-3]: {'label': f'Best: {k.name[:-3]}' if k.name[:-3] in best_models else k.name[:-3], 'value': k.name[:-3]}
    for k in list(pathlib.Path().glob(f'{MODELS_PATH}/*.th'))}

# Tabs content
tab_exploratory = html.Div(
    id="content-exploratory",
    style={"display": 'None'},
    children=[
        # Explanatory info
        html.H2(
            children=["Explanatory information"],
            id="title_explanatory",
            style=TITLE_STYLE
        ),
        html.Div(
            children=[
                dcc.Graph(
                    figure=figures_dist_age[0],
                    id="figures_dist_age_bar",
                    style={
                        "display": "inline-block",
                        "width": '50%'
                    }
                ),
                dcc.Graph(
                    figure=figures_dist_age[1],
                    id="figures_dist_age_box",
                    style={
                        "display": "inline-block",
                        "width": '50%'
                    }
                ),
                dcc.Graph(
                    figure=figures_dist_gender,
                    id="figures_dist_gender",
                    style={
                        "display": "block",
                    }
                )
            ]
        )
    ]
)

tab_result = html.Div(
    id="content-result",
    style={"display": 'None'},
    children=[
        # Metrics per dataset and age
        html.H2(
            children=["Metrics per dataset and age"],
            id="title_metrics_dataset_age",
            style=TITLE_STYLE
        ),
        html.Div(
            children=[
                dcc.Dropdown(
                    options=list(options_models.values()),
                    placeholder='Choose a model',
                    id='dropdown_model',
                    style={
                        "display": "block",
                        "margin-left": "15%",
                        "width": "70%"
                    }
                ),
                dcc.Loading(
                    id="loading_model",
                    type="default",
                    children=[
                        dcc.Graph(
                            id="figure_datasets_metrics",
                            style={
                                "display": "block",
                            }
                        ),
                        dcc.Graph(
                            id="figure_age_error_metrics",
                            style={
                                "display": "block",
                            }
                        ),
                        dcc.Graph(
                            id="figure_age_error_metrics_line",
                            style={
                                "display": "block",
                            }
                        )
                    ]
                )
            ]
        ),

        # Title of the prediction section
        html.H2(
            children=["Prediction"],
            id="prediction_title",
            style=TITLE_STYLE
        ),
        html.Div(
            children=[
                dbc.Input(id="input_age", placeholder="Introduce age", type="text"),
                dbc.Select(
                    id="input_gender",
                    options=[
                        {"label": "Man", "value": "man"},
                        {"label": "Woman", "value": "woman"}
                    ],
                    style={
                        'margin-top': '0.5%'
                    }
                )
            ],
            style={
                'width': '15%',
                'margin-top': '2.5%',
                'margin-left': '42.5%',
                'margin-bottom': '1%'
            }
        ),
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '50%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': 'auto'
            },
            # Only one image can be uploaded
            multiple=False
        ),
        dcc.Loading(
            id="loading_results",
            type="default",
            children=[
                # original image
                html.Div(
                    children=[
                        dbc.Card(
                            children=[
                                dbc.CardHeader('Original Image'),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id="figure_original_image"
                                        )
                                    ]
                                )
                            ],
                            style={
                                'width': '30%',
                                'margin-left': '35%',
                                'margin-top': '2%',
                                'margin-bottom': '2%'
                            },
                        )
                    ]
                ),
                # Prediction results
                html.H2(
                    children=["Results"],
                    id="results_title",
                    style=TITLE_STYLE
                ),
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                dbc.Row(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardHeader("Age", style={'text-align': 'center'}),
                                                dbc.CardBody(
                                                    [
                                                        html.H5("", id="card-age",
                                                                className="card-title",
                                                                style={'text-align': 'center'}),
                                                    ]
                                                ),
                                                dbc.CardFooter('', id='age_footer',
                                                               style={'text-align': 'center'})
                                            ],
                                            id='age_complete_card',
                                            style={
                                                "width": "15%",
                                                'margin-left': '32.5%',
                                                'margin-right': '5%'
                                            },
                                            color="primary",
                                            inverse=True),
                                        dbc.Card(
                                            [
                                                dbc.CardHeader("Gender",
                                                               style={'text-align': 'center'}),
                                                dbc.CardBody(
                                                    [
                                                        html.H5("", id="card-gender",
                                                                className="card-title",
                                                                style={'text-align': 'center'}),
                                                    ]
                                                ),
                                                dbc.CardFooter('', id='gender_footer',
                                                               style={'text-align': 'center'})
                                            ],
                                            id='gender_complete_card',
                                            style={
                                                "width": "15%",
                                            },
                                            color="primary",
                                            inverse=True),
                                    ]
                                ),
                            ],
                            style={
                                'margin': 'auto'
                            }
                        ),
                    ],
                    style={
                        "margin-top": "2%"
                    }
                ),
                # saliency maps images
                html.H2(
                    children=["Maps"],
                    id="maps_title",
                    style=TITLE_STYLE
                ),
                html.Div(
                    children=[
                        dbc.Card(
                            [
                                dbc.CardHeader("Saliency Maps", style={'text-align': 'center'}),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader("Age",
                                                                       style={'text-align': 'center'}),
                                                        dbc.CardBody(
                                                            [
                                                                dcc.Graph(
                                                                    id="saliency_map_age"
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "35%",
                                                        'margin-left': '12.5%',
                                                        'margin-right': '5%'
                                                    },
                                                    color='dark',
                                                    outline=True),
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader("Gender",
                                                                       style={'text-align': 'center'}),
                                                        dbc.CardBody(
                                                            [
                                                                dcc.Graph(
                                                                    id="saliency_map_gender"
                                                                )
                                                            ]
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "35%",
                                                    },
                                                    color="dark",
                                                    outline=True),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            style={
                                'width': '80%',
                                'margin-left': '10%',
                                'margin-top': '2%'
                            },
                            color='dark',
                            outline=True
                        )
                    ]
                ),
                # integrated saliency maps images
                html.Div(
                    children=[
                        dbc.Card(
                            [
                                dbc.CardHeader("Integrated Saliency Maps",
                                               style={'text-align': 'center'}),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader("Age",
                                                                       style={'text-align': 'center'}),
                                                        dbc.CardBody(
                                                            [
                                                                dcc.Graph(
                                                                    id="integrated_saliency_map_age"
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "35%",
                                                        'margin-left': '12.5%',
                                                        'margin-right': '5%'
                                                    },
                                                    color='dark',
                                                    outline=True),
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader("Gender",
                                                                       style={'text-align': 'center'}),
                                                        dbc.CardBody(
                                                            [
                                                                dcc.Graph(
                                                                    id="integrated_saliency_map_gender"
                                                                )
                                                            ]
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "35%",
                                                    },
                                                    color="dark",
                                                    outline=True),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            style={
                                'width': '80%',
                                'margin-left': '10%',
                                'margin-top': '2%'
                            },
                            color='dark',
                            outline=True
                        )
                    ]
                ),
                # saliency combined with original image maps images
                html.Div(
                    children=[
                        dbc.Card(
                            [
                                dbc.CardHeader("Saliency Maps combined with Original Image",
                                               style={'text-align': 'center'}),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader("Age",
                                                                       style={'text-align': 'center'}),
                                                        dbc.CardBody(
                                                            [
                                                                dcc.Graph(
                                                                    id="saliency_map_combined_age"
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "35%",
                                                        'margin-left': '12.5%',
                                                        'margin-right': '5%'
                                                    },
                                                    color='dark',
                                                    outline=True),
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader("Gender",
                                                                       style={'text-align': 'center'}),
                                                        dbc.CardBody(
                                                            [
                                                                dcc.Graph(
                                                                    id="saliency_map_combined_gender"
                                                                )
                                                            ]
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "35%",
                                                    },
                                                    color="dark",
                                                    outline=True),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            style={
                                'width': '80%',
                                'margin-left': '10%',
                                'margin-top': '2%'
                            },
                            color='dark',
                            outline=True
                        )
                    ]
                ),
                # heat maps
                html.Div(
                    children=[
                        dbc.Card(
                            [
                                dbc.CardHeader("Heat Maps", style={'text-align': 'center'}),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader("Age",
                                                                       style={'text-align': 'center'}),
                                                        dbc.CardBody(
                                                            [
                                                                dcc.Graph(
                                                                    id="heat_map_age"
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "35%",
                                                        'margin-left': '12.5%',
                                                        'margin-right': '5%'
                                                    },
                                                    color='dark',
                                                    outline=True),
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader("Gender",
                                                                       style={'text-align': 'center'}),
                                                        dbc.CardBody(
                                                            [
                                                                dcc.Graph(
                                                                    id="heat_map_gender"
                                                                )
                                                            ]
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "35%",
                                                    },
                                                    color="dark",
                                                    outline=True),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            style={
                                'width': '80%',
                                'margin-left': '10%',
                                'margin-top': '2%'
                            },
                            color='dark',
                            outline=True
                        )
                    ]
                ),
            ]
        )
    ]
)

# Main content
app.layout = html.Div(
    style={
        'backgroundColor': 'lightgray',
    },
    children=[
        html.Div(
            style={
                'backgroundColor': 'white',
                "margin-right": "7%",
                "margin-left": "7%",
                "border-style": "groove"
            },
            children=[
                # Title
                html.H1(
                    children=["Age-Gender Detector"],
                    id="title",
                    style=TITLE_STYLE
                ),
                # tabs
                dcc.Tabs(
                    id="tabs",
                    value="tab-exploratory",
                    children=[
                        dcc.Tab(label="Exploratory", value="tab-exploratory"),
                        dcc.Tab(label="Results", value="tab-results")
                    ]
                ),
                tab_exploratory,
                tab_result
            ]
        )
    ]
)


# Tabs control
@app.callback(
    Output("content-exploratory", "style"),
    Output("content-result", "style"),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-exploratory':
        return {"display": 'block'}, {"display": 'none'}
    elif tab == 'tab-results':
        return {"display": 'none'}, {"display": 'block'}
    else:
        return {"display": 'none'}, {"display": 'none'}


# Dropdown model
@app.callback(
    Output("figure_datasets_metrics", "figure"),
    Output("figure_age_error_metrics", "figure"),
    Output("figure_age_error_metrics_line", "figure"),
    Output("figure_datasets_metrics", "style"),
    Output("figure_age_error_metrics", "style"),
    Output("figure_age_error_metrics_line", "style"),
    Input("dropdown_model", "value")
)
def dropdown_model(model_name):
    if model_name is None:
        return go.Figure(), go.Figure(), go.Figure(), {"display": 'block'}, {"display": 'none'}, {"display": 'none'}

    if model_name != curr_model.get("name", ""):
        curr_model["name"] = model_name
        curr_model["model"] = load_model_from_name(f"{MODELS_PATH}/{model_name}").to(device)
    model = curr_model["model"]

    return *visualize_model(
        # f"{MODELS_PATH}/{model}",
        model,
        DATA_PATH,
        num_workers=0,
        batch_size=32,
        use_gpu=USE_GPU
    ), {"display": 'block'}, {"display": 'block'}, {"display": 'block'}


# Predict footer
@app.callback(Output('age_footer', 'children'),
              Output('gender_footer', 'children'),
              Output('age_complete_card', 'color'),
              Output('gender_complete_card', 'color'),
              Input('card-age', 'children'),
              Input("card-gender", "children"),
              Input('input_age', 'value'),
              Input('input_gender', 'value'))
def update_footer(age_prediction, gender_prediction, age, gender) -> tuple[str, str, str, str]:
    if age_prediction != '' and gender_prediction != '' and age is not None and gender is not None:
        age_prediction = int(age_prediction)
        age = int(age)
        age_result = 'correct' if np.abs(age - age_prediction) <= 10 else 'incorrect'
        gender_result = 'correct' if gender_prediction == gender else 'incorrect'
        age_color = 'success' if np.abs(age - age_prediction) <= 10 else 'danger'
        gender_color = 'success' if gender_prediction == gender else 'danger'
        return age_result, gender_result, age_color, gender_color
    else:
        return '', '', 'primary', 'primary'


# Update Maps
@app.callback(Output('figure_original_image', 'figure'),
              Output('saliency_map_age', 'figure'),
              Output('saliency_map_gender', 'figure'),
              Output('integrated_saliency_map_age', 'figure'),
              Output('integrated_saliency_map_gender', 'figure'),
              Output('saliency_map_combined_age', 'figure'),
              Output('saliency_map_combined_gender', 'figure'),
              Output('heat_map_age', 'figure'),
              Output('heat_map_gender', 'figure'),
              Output('card-age', 'children'),
              Output('card-gender', 'children'),
              Input('upload-image', 'contents'),
              Input("dropdown_model", "value"))
def update_prediction(image, model_name):
    zero_array = np.zeros((200, 200, 3))
    zero_array[:, :] = 255
    zero_fig = px.imshow(zero_array)
    zero_fig.update_layout(coloraxis_showscale=False)
    zero_fig.update_xaxes(showticklabels=False)
    zero_fig.update_yaxes(showticklabels=False)

    if image is None:
        return zero_fig, zero_fig, zero_fig, zero_fig, zero_fig, zero_fig, zero_fig, zero_fig, zero_fig, "", ""

    data = image.encode("utf8").split(b";base64,")[1]
    # temp_dir = tempfile.TemporaryDirectory()
    # temp_img = f'{temp_dir.name}/image.jpg'
    temp_dir = "temporary"
    temp_img = f'{temp_dir}/image.jpg'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    with open(temp_img, 'wb') as fp:
        fp.write(base64.decodebytes(data))

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    image = Image.open(temp_img)
    image = transforms(image)
    original_image = torch.swapaxes(torch.swapaxes(image, 0, 2), 0, 1)
    original_fig = px.imshow(original_image.detach().numpy())
    original_fig.update_layout(coloraxis_showscale=False)
    original_fig.update_xaxes(showticklabels=False)
    original_fig.update_yaxes(showticklabels=False)

    if model_name is not None:
        if model_name != curr_model.get("name", ""):
            curr_model["name"] = model_name
            curr_model["model"] = load_model_from_name(f"{MODELS_PATH}/{model_name}").to(device)
        model = curr_model["model"]

        predictions = predict_age_gender(model, [temp_img], use_gpu=USE_GPU)
        gender = int(predictions[0][0].item())
        age = predictions[0][1].item()

        image = image.to(device)

        dict_model = load_dict(f"{MODELS_PATH}/{model_name}.dict")
        model = CNNClassifier(**dict_model).to(device)
        model.load_state_dict(torch.load(f"{MODELS_PATH}/{model_name}.th"))

        original_saliency_map = OriginalSaliencyMap(model)
        saliency_map_gender, saliency_map_age = original_saliency_map.visualize(image.unsqueeze(0),
                                                                                gender, age, use_gpu=USE_GPU)
        integrated_saliency_map = IntegratedSaliencyMap(model)
        integrated_saliency_map_gender, integrated_saliency_map_age = integrated_saliency_map.visualize(
            image.unsqueeze(0), gender, age, use_gpu=USE_GPU)

        saliency_map_combined = SaliencyMapCombined(model)
        saliency_map_combined_gender, saliency_map_combined_age = saliency_map_combined.visualize(image.unsqueeze(0),
                                                                                                  gender, age,
                                                                                                  use_gpu=USE_GPU)

        if NO_MAPS:
            return original_fig, zero_fig, zero_fig, zero_fig, zero_fig, str(int(age)), LABEL_GENDER[gender]

        if SHOW_HEAT_MAPS:
            heat_map = HeatMap(model)
            heat_map_gender, heat_map_age = heat_map.visualize(image.unsqueeze(0), gender, age, use_gpu=USE_GPU)

            return original_fig, saliency_map_age, saliency_map_gender, integrated_saliency_map_age, \
                   integrated_saliency_map_gender, saliency_map_combined_age, saliency_map_combined_gender, \
                   heat_map_age, heat_map_gender, str(int(age)), LABEL_GENDER[gender]
        else:
            return original_fig, saliency_map_age, saliency_map_gender, integrated_saliency_map_age, \
                   integrated_saliency_map_gender, saliency_map_combined_age, saliency_map_combined_gender, \
                   zero_fig, zero_fig, str(int(age)), LABEL_GENDER[gender]
    else:
        return original_fig, zero_fig, zero_fig, zero_fig, zero_fig, zero_fig, zero_fig, zero_fig, zero_fig, "", ""


if __name__ == '__main__':
    app.run_server(debug=True)
