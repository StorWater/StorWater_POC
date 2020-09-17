import holoviews as hv
import hvplot
import hvplot.pandas
import hvplot.streamz
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamz
import umap
from bokeh.models import HoverTool
from holoviews import streams
from plotly.subplots import make_subplots
from sklearn.manifold.t_sne import TSNE
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.neighbors.classification import KNeighborsClassifier
from streamz.dataframe import DataFrame as StreamzDataFrame

from model_evaluation import plot_classification_report, plot_confussion_matrix


def define_visibles(x):
    """Funcion usada para definir qué traces se muestran en un grafico de plotly

    Examples
    --------
    >>> define_visibles(x=[2,1,3])
    [[True, True, False, False, False, False],
    [False, False, True, False, False, False],
    [False, False, False, True, True, True]]

    Parameters
    -------
    x: list or np.array
            Contiene el numero de clases/traces por cada dropdown menu. [1,1,1] significa 1 trace, 3 dropdown menus

    Returns
    -------
    list
            Lista de listas, contiene solo True or False.
    """
    if isinstance(x, list):
        x = np.array(x)

    visible_trace = []
    for i, a in enumerate(x):
        visible_trace.append(
            [False] * np.sum(x[0:i]) + [True] * x[i] + [False] * np.sum(x[i + 1 :])
        )

    return visible_trace


def time_vs_y(df, time_col, id_col_name, id_list, cols_descr, y_col="y", title=""):
    """PLot a time series dataset with bars when a certain column is equal to 1

    Example
    --------
    time_vs_y(df = df_m2,
          time_col = 'Timestamp',
          id_col_name = 'DMA',
          id_list =['NEWSEVMA','NORFIFMA'],
          cols_descr = ['PressureBar','m3Volume'],
          y_col = 'is_leakage',
          title='Time series of leakage and pressure at DMA level')

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that contains all the information
    time_col : str
        Name of column with time
    id_col_name : str
        column name to aggregate by. Effectively this sets the dropdown menu
    id_list : list of str
        List of ids to display
    cols_descr : [type]
        Selected columns to plot against time
    y_col : str, optional
        Column of dataset that contains 1 when a gray bar is plotted
    title : str, optional
        Title of plot, by default ""

    Returns
    -------
    plotly.Figure
        PLotly figure that contains dropdown menu for each id_list
    """

    if len(cols_descr) == 0:
        print("No selected columns")
        return 0

    fig = go.Figure()
    buttons = []
    visible_start = [True] + [False] * (len(id_list) - 1)

    # Select which columns are visible for each button
    visible_trace = define_visibles([len(cols_descr)] * len(id_list))

    # Loop over the selected columns and create trace
    for i, z in enumerate(id_list):

        df_subset = df.loc[
            df[id_col_name] == z,
        ]
        # Generate figure and keep data and layout
        for c in cols_descr:
            fig.add_trace(
                go.Scattergl(
                    x=df_subset[time_col],
                    y=df_subset[c],
                    name=f"{c}",
                    visible=visible_start[i],
                )
            )

        # Print lines as shapes
        shapes = list()
        min_val = 0
        for j in np.where(df_subset[y_col] == 1)[0]:
            if j == 0:
                continue
            max_val = (
                df_subset[cols_descr]
                .iloc[
                    j - 1,
                ]
                .max()
                .max()
            )

            shapes.append(
                {
                    "type": "line",
                    "xref": "x",
                    "yref": "y",
                    "x0": df_subset[time_col].iloc[j] - pd.Timedelta(1, "h"),
                    "y0": min_val,
                    "x1": df_subset[time_col].iloc[j],
                    "y1": max_val,
                    "fillcolor": "gray",
                    "type": "rect",
                    "opacity": 0.5,
                    "layer": "below",
                    "line_width": 0,
                }
            ),

        if visible_start[i] is True:
            fig.update_layout(shapes=shapes)

        # Crear botones
        buttons.append(
            dict(
                label=id_list[i],
                method="update",
                args=[{"visible": visible_trace[i]}, {"shapes": shapes}],
            )
        )
    # Añadir botones
    fig.update_layout(
        updatemenus=[
            go.layout.Updatemenu(
                direction="up",
                showactive=True,
                xanchor="center",
                yanchor="bottom",
                pad={"l": 150, "b": -390, "t": 0},
                buttons=buttons,
            )
        ]
    )

    fig.update_layout(width=1100, height=500, title=title)

    return fig


def visualize_time_series(
    df, range_dates_zoom=None, range_y_right=[1.5, 5], range_y_left=[-10, 45]
):
    """[summary]

    Parameters
    ----------
    df : [type]
        [description]
    range_dates_zoom : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scattergl(
            x=df["Timestamp"], y=df["m3Volume"], name="m3Volume", marker_color="#636EFA"
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scattergl(
            x=df["Timestamp"],
            y=df["PressureBar"],
            name="PressureBar",
            marker_color="#EF553B",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(range=range_y_right, secondary_y=True)
    fig.update_yaxes(range=range_y_left, secondary_y=False)

    # Print lines as shapes
    shapes = list()
    min_val = np.min([range_y_right, range_y_left])
    max_val = np.max([range_y_right, range_y_left])
    for j in np.where(df.is_leakage == 1)[0]:
        if j == 0:
            continue
        shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": df["Timestamp"].iloc[j] - pd.Timedelta(1, "h"),
                "y0": min_val,
                "x1": df["Timestamp"].iloc[j],
                "y1": max_val,
                "fillcolor": "gray",
                "type": "rect",
                "opacity": 1,
                "layer": "below",
                "line_width": 0,
            }
        ),

    fig.update_layout(shapes=shapes)

    # Add autoscale option
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                pad={"r": 10, "t": -80},
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="Zoom out",
                        method="relayout",
                        args=[
                            "xaxis",
                            [dict(range=[df.Timestamp.min(), df.Timestamp.max()])],
                        ],
                    ),
                ],
            )
        ]
    )

    # Add figure title
    fig.update_layout(title_text="Time series of inflow volume and pressure")
    fig.update_xaxes(title_text="Timestamp")
    if range_dates_zoom:
        fig.update_xaxes(type="date", range=range_dates_zoom)
    fig.update_yaxes(
        title_text="<b>Volume (m3)</b>",
        secondary_y=False,
        title_font=dict(color="#636EFA"),
    )
    fig.update_yaxes(
        title_text="<b>Pressure (bar)</b>",
        secondary_y=True,
        title_font=dict(color="#EF553B"),
    )

    return fig


def plot_roc_curve(y, y_score):
    px, py, thrs = roc_curve(y, y_score)
    rocdf = pd.DataFrame(
        columns=["False positive rate", "True positive rate", "Threshold"],
        data=np.array([px, py, thrs * 100]).T,
    )
    curve = hv.Curve(rocdf).opts(tools=["hover"], xlabel="False positive rate")
    _auc = auc(px, py)
    return curve * hv.Curve(([0, 1], [0, 1])).opts(
        xlim=(-0.01, 1.0),
        ylim=(-0.01, 1.05),
        width=400,
        height=400,
        show_grid=True,
        title="ROC curve with AUC={:.3f}".format(_auc),
    )


def plot_classification_report(y, y_score, threshold=0.5, **kwargs):
    y_pred = np.where(y_score > threshold, 1, 0)
    report = classification_report(y, y_pred, output_dict=True, **kwargs)
    df = pd.DataFrame(report).applymap(lambda x: "{:.2f}".format(x))
    df = df.T.reset_index()
    return hv.Table(df).opts(title="Classification report")


def plot_confussion_matrix(
    y,
    y_score,
    threshold=0.5,
    target_names: list = None,
    cmap: str = "YlGnBu",
    width=500,
    height: int = 400,
    title: str = "Confusion matrix",
    normalize: bool = False,
):
    value_label = "examples"
    target_label = "true_label"
    pred_label = "predicted_label"
    label_color = "color"

    def melt_distances_to_heatmap(distances: pd.DataFrame) -> pd.DataFrame:
        dist_melt = pd.melt(
            distances.reset_index(), value_name=value_label, id_vars="index"
        )
        dist_melt = dist_melt.rename(
            columns={"index": target_label, "variable": pred_label}
        )
        dist_melt[target_label] = pd.Categorical(dist_melt[target_label])
        dist_melt[pred_label] = pd.Categorical(dist_melt[pred_label])
        coords = dist_melt.copy()
        coords[target_label] = dist_melt[target_label].values.codes
        coords[pred_label] = dist_melt[pred_label].values.codes
        return coords[[pred_label, target_label, value_label]]

    y_pred_bin = np.where(y_score > threshold, 1, 0)
    conf_matrix = confusion_matrix(y, y_pred_bin)
    if normalize:
        conf_matrix = np.round(
            conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis], 3
        )
    # Adjust label color to make them readable when displayed on top of any colormap
    df = melt_distances_to_heatmap(pd.DataFrame(conf_matrix))
    mean = df[value_label].mean()
    df[label_color] = -df[value_label].apply(lambda x: int(x > mean))
    if target_names is not None:
        df[target_label] = df[target_label].apply(lambda x: target_names[x])
        df[pred_label] = df[pred_label].apply(lambda x: target_names[x])
    true_label_name = "Actual label"
    pred_label_name = "Predicted label"

    tooltip = [
        (true_label_name, "@{%s}" % target_label),
        (pred_label_name, "@{%s}" % pred_label),
        ("Examples", "@{%s}" % value_label),
    ]
    hover = HoverTool(tooltips=tooltip)
    heatmap = hv.HeatMap(df, kdims=[pred_label, target_label])
    heatmap.opts(
        title=title, colorbar=True, cmap=cmap, width=width, height=height, tools=[hover]
    )
    labeled = heatmap * hv.Labels(heatmap).opts(text_color=label_color, cmap=cmap)
    return labeled.options(
        xlabel=pred_label_name, ylabel=true_label_name, invert_yaxis=True
    )


def plot_model_evaluation(y, y_score):
    cr = plot_classification_report(y, y_score)
    conf_mat = plot_confussion_matrix(y, y_score, target_names=["No leak", "Leak"])
    roc_c = plot_roc_curve(y, y_score)
    return (cr + conf_mat + roc_c).cols(2)
