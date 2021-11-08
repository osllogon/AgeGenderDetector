import pathlib

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objects as go
from models.utils import load_list
from .visualize import visualize_datasets_age, visualize_datasets_gender, get_datasets, visualize_model

MODELS_PATH = 'models/savedAgeGender'
DATA_PATH = "./data/UTKFace"

# Inicializamos la aplicacion de dash
app = dash.Dash()

TITLE_STYLE = {
    "text-align": "center",
    "text-decoration": "bold"
}

# Get figures
datasets = get_datasets(DATA_PATH)
figures_dist_age = visualize_datasets_age(datasets)
figures_dist_gender = visualize_datasets_gender(datasets)

best_models = load_list(f'{MODELS_PATH}/bestModels.txt')
options_models = {
    k.name[:-3]: {'label': f'Best: {k.name[:-3]}' if k.name[:-3] in best_models else k.name[:-3], 'value': k.name[:-3]}
    for k in list(pathlib.Path().glob(f'{MODELS_PATH}/*.th'))}

app.layout = html.Div(
    children=[
        # Title
        html.H1(
            children=["Age-Gender Detector"],
            id="title",
            style=TITLE_STYLE
        ),
        # Section with explanatory info
        html.Div(
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
                ),

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
                                "width":"70%"
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
            ]
        )
    ]
)


@app.callback(
    Output("figure_datasets_metrics", "figure"),
    Output("figure_age_error_metrics", "figure"),
    Output("figure_age_error_metrics_line", "figure"),
    Output("figure_datasets_metrics", "style"),
    Output("figure_age_error_metrics", "style"),
    Output("figure_age_error_metrics_line", "style"),
    Input("dropdown_model", "value")
)
def dropdown_model(model):
    if model is None:
        return go.Figure(), go.Figure(), go.Figure(), {"display": 'block'}, {"display": 'none'}, {"display": 'none'}

    return *visualize_model(
        f"{MODELS_PATH}/{model}",
        DATA_PATH,
        num_workers=0,
        batch_size=64,
        use_gpu=True
    ), {"display": 'block'}, {"display": 'block'}, {"display": 'block'}


if __name__ == '__main__':
    app.run_server()
