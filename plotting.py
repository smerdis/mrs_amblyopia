import numpy as np
import pandas as pd

from bokeh.layouts import row, column
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models import BoxSelectTool, LassoSelectTool, Spacer, HoverTool
from bokeh.palettes import Category10


def plot_suppressive_weights(os_fitted_df):
    """
    Plot the fitted w_m and w_d parameters of the two-stage model for all participants.
    """
    TOOLS="pan,wheel_zoom,box_select,lasso_select,reset"

    # create the scatter plot
    p = figure(tools=TOOLS, plot_width=800, plot_height=600, min_border=10, min_border_left=50,
               toolbar_location="above",
               x_axis_label="Monocular suppressive weight",
               y_axis_label="Interocular suppressive weight",
               title="Monocular and interocular suppressive weights")
    p.background_fill_color = "#fafafa"
    p.select(BoxSelectTool).select_every_mousemove = False
    p.select(LassoSelectTool).select_every_mousemove = False

    plot_groups = os_fitted_df.groupby(["Eye", "Orientation"])
    for (gv, g), c in zip(plot_groups, Category10[4]):
        source = ColumnDataSource(os_fitted_df)
        r = p.circle('w_m', 'w_d', size=10, color=c, alpha=0.6, source=source)

    hover = HoverTool(tooltips=[("Subject", "@Subject"),
                                ("Eye", "@Eye"),
                                ("Orientation", "@Orientation"),
                                ("Presentation", "@Presentation"),
                                ("Monocular suppressive weight", "@w_m"),
                                ("Interocular suppressive weight", "@w_d")],
                      mode="mouse", point_policy="follow_mouse", renderers=[r])

    p.add_tools(hover)

    # create the horizontal histogram
    hhist, hedges = np.histogram(os_fitted_df.w_m, bins=200)
    hzeros = np.zeros(len(hedges)-1)
    hmax = max(hhist)*1.1

    LINE_ARGS = dict(color="#3A5785", line_color=None)

    ph = figure(toolbar_location=None, plot_width=p.plot_width, plot_height=200, x_range=p.x_range,
                y_range=(-hmax, hmax), min_border=10, min_border_left=50, y_axis_location="right")
    ph.xgrid.grid_line_color = None
    ph.yaxis.major_label_orientation = np.pi/4
    ph.background_fill_color = "#fafafa"

    ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="white", line_color="#3A5785")
    hh1 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.5, **LINE_ARGS)
    hh2 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.1, **LINE_ARGS)

    # create the vertical histogram
    vhist, vedges = np.histogram(os_fitted_df.w_d, bins=200)
    vzeros = np.zeros(len(vedges)-1)
    vmax = max(vhist)*1.1

    pv = figure(toolbar_location=None, plot_width=200, plot_height=p.plot_height, x_range=(-vmax, vmax),
                y_range=p.y_range, min_border=10, y_axis_location="right")
    pv.ygrid.grid_line_color = None
    pv.xaxis.major_label_orientation = np.pi/4
    pv.background_fill_color = "#fafafa"

    pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color="white", line_color="#3A5785")
    vh1 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.5, **LINE_ARGS)
    vh2 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.1, **LINE_ARGS)

    layout = column(row(p, pv), row(ph, Spacer(width=200, height=200)))

    def update(attr, old, new):
        inds = np.array(new['1d']['indices'])
        if len(inds) == 0 or len(inds) == len(os_fitted_df['w_m']):
            hhist1, hhist2 = hzeros, hzeros
            vhist1, vhist2 = vzeros, vzeros
        else:
            neg_inds = np.ones_like(os_fitted_df['w_m'], dtype=np.bool)
            neg_inds[inds] = False
            hhist1, _ = np.histogram(os_fitted_df['w_m'][inds], bins=hedges)
            vhist1, _ = np.histogram(os_fitted_df['w_d'][inds], bins=vedges)
            hhist2, _ = np.histogram(os_fitted_df['w_m'][neg_inds], bins=hedges)
            vhist2, _ = np.histogram(os_fitted_df['w_d'][neg_inds], bins=vedges)

        hh1.data_source.data["top"]   =  hhist1
        hh2.data_source.data["top"]   = -hhist2
        vh1.data_source.data["right"] =  vhist1
        vh2.data_source.data["right"] = -vhist2

    r.data_source.on_change('selected', update)
    return r, layout
