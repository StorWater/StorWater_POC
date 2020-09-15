import warnings

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
import pandas as pd
from bokeh.models import HoverTool
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.manifold.t_sne import TSNE
import umap
try:
    import holoviews as hv
    import hvplot.streamz
    import hvplot
    import hvplot.pandas
    from holoviews import streams
    import streamz
    from streamz.dataframe import DataFrame as StreamzDataFrame
except:
    pass


def safe_margin(val, low=True, pct: float = 0.05):
    low_pct, high_pct = 1 - pct, 1 + pct
    func = min if low else max
    return func(val * low_pct, val * high_pct)


def safe_bounds(array, pct: float = 0.05):
    low_x, high_x = array.min(), array.max()
    low_x = safe_margin(low_x, pct=pct)
    high_x = safe_margin(high_x, pct=pct, low=False)
    return low_x, high_x
def predict_grid(model, X):
    x_grid, y_grid = example_meshgrid(X)
    grid = np.c_[x_grid.ravel(), y_grid.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(x_grid.shape)
    return probs, x_grid, y_grid
def plot_confussion_matrix(
    y_test,
    y_pred,
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
        dist_melt = pd.melt(distances.reset_index(), value_name=value_label, id_vars="index")
        dist_melt = dist_melt.rename(columns={"index": target_label, "variable": pred_label})
        dist_melt[target_label] = pd.Categorical(dist_melt[target_label])
        dist_melt[pred_label] = pd.Categorical(dist_melt[pred_label])
        coords = dist_melt.copy()
        coords[target_label] = dist_melt[target_label].values.codes
        coords[pred_label] = dist_melt[pred_label].values.codes
        return coords[[pred_label, target_label, value_label]]

    conf_matrix = confusion_matrix(y_test, y_pred)
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
    heatmap.opts(title=title, colorbar=True, cmap=cmap, width=width, height=height, tools=[hover])
    labeled = heatmap * hv.Labels(heatmap).opts(text_color=label_color, cmap=cmap)
    return labeled.options(xlabel=pred_label_name, ylabel=true_label_name, invert_yaxis=True)


def __plot_decision_boundaries(X, y, y_pred, resolution: int = 100, embedding=None):
    if embedding is None:
        embedding = TSNE(n_components=2, random_state=160290).fit_transform(X)

    x_min, x_max = safe_bounds(embedding[:, 0])
    y_min, y_max = safe_bounds(embedding[:, 1])
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )

    # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    background_model = KNeighborsClassifier(n_neighbors=1).fit(embedding, y_pred)
    voronoi_bg = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoi_bg = voronoi_bg.reshape((resolution, resolution))

    mesh = hv.QuadMesh((xx, yy, voronoi_bg)).opts(cmap="viridis")
    points = hv.Scatter(
        {"x": embedding[:, 0], "y": embedding[:, 1], "pred": y_pred, "class": y},
        kdims=["x", "y"],
        vdims=["pred", "class"],
    )
    errors = y_pred != y
    failed_points = hv.Scatter(
        {"x": embedding[errors, 0], "y": embedding[errors, 1]}, kdims=["x", "y"]
    ).opts(color="red", size=5, alpha=0.9)

    points = points.opts(
        color="pred", cmap="viridis", line_color="grey", size=10, alpha=0.8, tools=["hover"]
    )
    plot = mesh * points * failed_points
    plot = plot.opts(
        xaxis=None, yaxis=None, width=500, height=450, title="Decision boundaries on TSNE"
    )
    return plot


def plot_decision_boundaries(
    X_train,
    y_train,
    y_pred_train,
    X_test,
    y_test,
    y_pred_test,
    resolution: int = 100,
    embedding=None,
):
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    y_pred = np.concatenate([y_pred_train, y_pred_test])

    if embedding is None:
        try:
            embedding = umap.UMAP(n_components=2, random_state=160290).fit_transform(X)
        except:
            from sklearn.manifold import TSNE

            embedding = TSNE(n_components=2, random_state=160290).fit_transform(X)
    x_min, x_max = safe_bounds(embedding[:, 0])
    y_min, y_max = safe_bounds(embedding[:, 1])
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )

    # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    background_model = KNeighborsClassifier(n_neighbors=1).fit(embedding, y_pred)
    voronoi_bg = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoi_bg = voronoi_bg.reshape((resolution, resolution))

    mesh = hv.QuadMesh((xx, yy, voronoi_bg)).opts(cmap="viridis", alpha=0.6)
    points_train = hv.Scatter(
        {
            "x": embedding[: len(y_train), 0],
            "y": embedding[: len(y_train), 1],
            "pred": y_pred_train,
            "class": y_train,
        },
        kdims=["x", "y"],
        vdims=["pred", "class"],
    )
    points_test = hv.Scatter(
        {
            "x": embedding[len(y_train) :, 0],
            "y": embedding[len(y_train) :, 1],
            "pred": y_pred_test,
            "class": y_test,
        },
        kdims=["x", "y"],
        vdims=["pred", "class"],
    )
    errors = y_pred != y
    failed_points = hv.Scatter(
        {"x": embedding[errors, 0], "y": embedding[errors, 1]}, kdims=["x", "y"]
    ).opts(color="red", size=2, alpha=0.9)

    points_train = points_train.opts(
        color="class", cmap="viridis", line_color="grey", size=10, alpha=0.8, tools=["hover"]
    )
    points_test = points_test.opts(
        color="class",
        cmap="viridis",
        line_color="grey",
        size=10,
        alpha=0.8,
        tools=["hover"],
        marker="square",
    )
    plot = mesh * points_train * points_test * failed_points
    plot = plot.opts(xaxis=None, yaxis=None, width=500, height=450, title="Fronteras de decisión")
    return plot


def plot_classification_report(y, y_pred, **kwargs):
    report = classification_report(y, y_pred, output_dict=True, **kwargs)
    df = pd.DataFrame(report).applymap(lambda x: "{:.2f}".format(x))
    df = df.T.reset_index()
    return hv.Table(df).opts(title="Classification report")


def plot_feature_importances(model, target_names=None, feature_names=None, stacked: bool = False):
    n_target, n_features = model.coef_.shape
    ix = feature_names if feature_names is not None else list(range(n_features))
    cols = (
        target_names[:n_target]
        if target_names is not None
        else ["class_{}".format(i) for i in range(n_target)]
    )
    df = pd.DataFrame(index=ix, columns=cols, data=model.coef_.T)
    df.index.name = "Features"
    df.columns.name = "output_class"
    bar = df.hvplot.bar(legend=True, stacked=stacked, rot=75)
    bar = bar.opts(ylabel="Aggregated coefficients", title="Feature importances")
    return bar


def plot_model_evaluation(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    target_names=None,
    feature_names=None,
    normalize: bool = False,
    resolution: int = 100,
    stacked: bool = False,
    embedding: np.ndarray = None,
):
    import panel as pn

    y_pred_test = model.predict(X_test)
    metrics = plot_classification_report(y=y_test, y_pred=y_pred_test, target_names=target_names)
    conf_mat = plot_confussion_matrix(
        y_test=y_test, y_pred=y_pred_test, target_names=target_names, normalize=normalize
    )
    bounds = plot_decision_boundaries(
        X_train=X_train,
        y_train=y_train,
        y_pred_train=model.predict(X_train),
        X_test=X_test,
        y_test=y_test,
        y_pred_test=model.predict(X_test),
        resolution=resolution,
        embedding=embedding,
    )
    # features = plot_feature_importances(
    #     model=model, target_names=target_names, feature_names=feature_names, stacked=stacked
    # )
    gspec = pn.GridSpec(
        min_height=700, height=700, max_height=1200, min_width=750, max_width=1980, width=750
    )
    gspec[0, 0] = bounds
    gspec[1, 1] = metrics
    gspec[1, 0] = pn.pane.HTML(str(model), margin=0)
    gspec[0, 1] = conf_mat
    # gspec[3:5, :] = features
    return gspec
