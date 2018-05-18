import numpy as np
import pandas as pd

from bokeh.layouts import row, column, gridplot
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models import BoxSelectTool, LassoSelectTool, Spacer, HoverTool, CDSView, GroupFilter
from bokeh.palettes import Category20
from bokeh.transform import factor_cmap


def plot_fitted_parameters(fitted_df, x, y, color_factor, plot_factor):
    """
    Plot the fitted model parameters specified by x and y
    in a different color for each level of color_factor,
    on a different plot for each level of plot_factor
    """
    
    # Load the data
    source = ColumnDataSource(fitted_df)
    plot_factor_levels = np.unique(fitted_df[plot_factor])
    views = [CDSView(source=source, filters=[GroupFilter(column_name=plot_factor, group=lvl)]) for lvl in plot_factor_levels]

    # define common attributes of the various plots
    plot_size_and_tools = {'plot_height': 400, 'plot_width': 400, 'min_border':10, 'min_border_left':50, 'toolbar_location':'above',
                        'tools':['pan','wheel_zoom','box_select','lasso_select','reset']}

    ps = []
    for v in views:
        f = v.filters
        plot_factor_level = v.filters[0].group
        # create the scatter plot
        p = figure(title=f"{plot_factor}: {plot_factor_level}", **plot_size_and_tools)
        p.background_fill_color = "#fafafa"
        p.select(BoxSelectTool).select_every_mousemove = False
        p.select(LassoSelectTool).select_every_mousemove = False

        distinct_groups = np.unique(fitted_df[color_factor])
        n_distinct_groups = len(distinct_groups)
        r = p.circle(x, y, source=source, view=v, size=10, 
            color=factor_cmap(color_factor, palette=Category20[n_distinct_groups], factors=distinct_groups),
            legend=color_factor,
            alpha=0.6)

        hover = HoverTool(tooltips=[("Subject", "@Subject"),
                                    ("Eye", "@Eye"),
                                    ("Orientation", "@Orientation"),
                                    ("Monocular suppressive weight", f"@{x}"),
                                    ("Interocular suppressive weight", f"@{y}")],
                          mode="mouse", point_policy="follow_mouse", renderers=[r])

        p.add_tools(hover)
        ps.append(p)

    return gridplot([ps])
