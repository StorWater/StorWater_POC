import pandas as pd
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go


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
