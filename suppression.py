# suppression.py
#
# Functions to read in, deal with, and plot the psychophysical data extracted from MATLAB

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as tick
import matplotlib.image as mpimg
from matplotlib.offsetbox import (OffsetImage,
                                  AnnotationBbox)
import seaborn as sns

import itertools as it

## Plotting functions
def scaterror_plot(x, y, **kwargs):
    #'''This function is designed to be called with FacetGrid.map_dataframe() to make faceted plots of various conditions.'''
    #print x, y, kwargs
    data = kwargs.pop("data")
    ses = data[kwargs.pop("yerr")].values
    # control the plotting
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.get_xaxis().set_major_locator(tick.LogLocator(subs=[1,2,3,4,5,6,7,8,9]))
    #ax.get_yaxis().set_major_locator(tick.LogLocator(subs=[1, 3]))
    #ax.get_xaxis().set_major_formatter(tick.ScalarFormatter())
    #ax.get_yaxis().set_major_formatter(tick.ScalarFormatter())
    #ax.set_ylim([0.6, 4])
    #ax.set_xlim([4, 105])
    plt.errorbar(data=data, x=x, y=y, yerr=ses, **kwargs)
    plt.axhline(y=1, ls='dotted', color='gray')

def group_scatter_plot(x, y, **kwargs):
    """Plotting function for group scatter plots, SfN poster 2018"""
    data = kwargs.pop("data")
    yerr = kwargs.pop("yerr")
    # control the plotting
    fig, ax = plt.subplots(1, figsize=(6.2,5.1))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([0.6, 4.2])
    ax.get_xaxis().set_major_locator(tick.LogLocator(subs=[1,2,3,4,5,6,7,8,9]))
    ax.set_xticklabels([])
    ax.get_yaxis().set_major_locator(tick.LogLocator(subs=[1,2,3,4,5,6,7,8,9]))
    ax.set_yticklabels([])
    ax.get_yaxis().set_major_formatter(tick.NullFormatter())
    ax.get_yaxis().set_minor_formatter(tick.NullFormatter())

    # Plot the eyes separately so we can apply different styling
    traces = data.groupby('Trace')
    fmt = {'Amblyope-Nde':':x', 'Amblyope-De':'--s',
           'Control-Nde':':x', 'Control-De':'--s'}
    for trace, trace_df in traces:
        ses = trace_df[yerr].values
        ax.errorbar(data=trace_df, x=x, y=y, yerr=ses, fmt=fmt[trace], **kwargs)
    ax.axhline(y=1, ls='dotted', color='gray')
    return(fig)

def subject_fit_plot(x, y, **kwargs):
    # set up the data frame for plotting, get kwargs etc
    data = kwargs.pop("data")
    fmt_obs = kwargs.pop("fmt_obs")
    fmt_pred = kwargs.pop("fmt_pred")
    ses = data[kwargs.pop("yerr")].values # SE's of actually observed threshold elevations
    predY = data[kwargs.pop("Ycol")].values
    #relMCToPred = data['RelMCToPred'].values
    #assert(np.all(relMCToPred==relMCToPred[0])) # check all the same

    # control the plotting
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.get_xaxis().set_major_locator(tick.LogLocator())
    ax.get_yaxis().set_major_locator(tick.LogLocator())
    #ax.get_xaxis().set_major_formatter(tick.ScalarFormatter())
    #ax.get_yaxis().set_major_formatter(tick.ScalarFormatter())
    #ax.set_ylim([0.5, np.max([np.max(data[y])+1])])
    #ax.set_xlim([0.5, 10])
    ax.errorbar(data=data, x=x, y=y, yerr=ses,fmt=fmt_obs, **kwargs)
    ax.errorbar(data=data, x=x, y=predY,fmt=fmt_pred, **kwargs)
    ax.axhline(y=1,ls='dotted')
    #ax.axvline(x=relMCToPred[0], ls='dotted')

def subject_fit_plot_pct(x, y, **kwargs):
    # set up the data frame for plotting, get kwargs etc
    data = kwargs.pop("data")
    fmt_obs = kwargs.pop("fmt_obs")

    # control the plotting
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.get_xaxis().set_major_locator(tick.LogLocator())
    ax.get_yaxis().set_major_locator(tick.LogLocator())
    ax.errorbar(data=data, x=x, y=y, fmt=fmt_obs, **kwargs)
    ax.axhline(y=1,ls='dotted')

def population_fit_plot(x, y, **kwargs):
    # set up the data frame for plotting, get kwargs etc
    data = kwargs.pop("data")
    fmt_obs = kwargs.pop("fmt_obs")
    fmt_pred = kwargs.pop("fmt_pred")
    yerr = kwargs.pop("yerr")
    ycol = kwargs.pop("Ycol")

    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.get_xaxis().set_major_locator(tick.LogLocator())
    ax.get_yaxis().set_major_locator(tick.LogLocator())

    pop_colors = {'Control':'C0', 'Amblyope':'C1'}

    for gvpop, gpop in data.groupby(["Population"]):
        #relMCToPred = gpop['RelMCToPred'].values
        #ax.axvline(x=relMCToPred[0], ls='dotted', color=pop_colors[gvpop])
        for gv, g in gpop.groupby(["Subject"]):
            ses = g[yerr].values # SE's of actually observed threshold elevations
            predY = g[ycol].values
            #assert(np.all(relMCToPred==relMCToPred[0])) # check all the same
            
            ax.errorbar(data=g, x=x, y=y, yerr=ses,fmt=fmt_obs, **kwargs)
            ax.errorbar(data=g, x=x, y=predY,fmt=fmt_pred, **kwargs)
            ax.axhline(y=1,ls='dotted')

def population_fit_plot_pct(x, y, **kwargs):
    # set up the data frame for plotting, get kwargs etc
    data = kwargs.pop("data")
    fmt_obs = kwargs.pop("fmt_obs")
    
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.get_xaxis().set_major_locator(tick.LogLocator())
    ax.get_yaxis().set_major_locator(tick.LogLocator())

    pop_colors = {'Control':'C0', 'Amblyope':'C1'}

    for gvpop, gpop in data.groupby(["Population"]):
        for gv, g in gpop.groupby(["Subject"]):            
            ax.errorbar(data=g, x=x, y=y,fmt=fmt_obs, **kwargs)
            ax.axhline(y=1,ls='dotted')

def gaba_vs_psychophys_plot(gv, gr, legend_box = [0.87, 0.55, 0.1, 0.1], **kwargs):
    print(gv)
    xvar = "GABA"
    yvar = "value"
    x_lbl = "GABA (relative to creatine)"
    y_lbl = {'BaselineThresh':'Baseline Threshold (C%)',
            'RelMCToPred':'Relative Mask Contrast to predict threshold at',
            'ThreshPredCritical':'Predicted threshold elevation (multiples of baseline)',
            'DepthOfSuppressionPred':'Depth of suppression (multiples of baseline threshold)\nnegative indicates facilitation',
            'ThreshPredCriticalUnnorm':'Predicted threshold elevation (C%)',
            'slope':'Slope of perceptual suppression fit line',
            'y_int':'y-intercept of perceptual suppression fit line'}
    g = sns.lmplot(data=gr, x=xvar, y=yvar, **kwargs)
    g.set(xlim=[.18, .23])
    g.set_axis_labels(x_lbl, y_lbl[gv[-1]])
    g.fig.suptitle(', '.join(gv), fontsize=16, y=0.97)

    #print(g.fig.get_figwidth())

    if g._legend:
        g._legend.set_title(f"Eye which\nviewed target")
        #bbox = g._legend.get_bbox_to_anchor()
        #print(bbox)

    if 'SS' in gv:
        if 'Iso' in gv:
            im = plt.imread(f"/Users/smerdis/Dropbox/Documents/cal/silverlab/mrs-amblyopia/ss-iso-targetandmask.png")
        elif 'Cross' in gv:
            im = plt.imread(f"/Users/smerdis/Dropbox/Documents/cal/silverlab/mrs-amblyopia/ss-cross-targetandmask.png")
        else:
            print('Unknown orientation...')

    #if bbox:
    #    pts = bbox.get_points()
    #    print(pts)
    #    newax = g.fig.add_axes([pts[0][0], pts[0][1], 0.1, 0.1], anchor='NE')
    #else:
    newax = g.fig.add_axes(legend_box, anchor='NE')
    newax.imshow(im)
    newax.axis('off')

    plt.close(g.fig)
    return(g)

def gaba_vs_psychophys_plot_2line(gv, gr, **kwargs):
    xvar = "GABA"
    x_lbl = "GABA (relative to creatine)"
    yvar = "value"
    y_lbl = {'BaselineThresh':'Baseline Threshold (C%)',
            'RelMCToPred':'Relative Mask Contrast to predict threshold at',
            'ThreshPredCritical':'Predicted threshold elevation (multiples of baseline)',
            'DepthOfSuppressionPred':'Depth of suppression (multiples of baseline threshold)\nnegative indicates facilitation',
            'ThreshPredCriticalUnnorm':'Predicted threshold elevation (C%)',
            'slope':'Slope of perceptual suppression fit line',
            'y_int':'y-intercept of perceptual suppression fit line'}
    g = sns.lmplot(data=gr, row='Orientation',
              col='Presentation', # facet rows and columns
              x=xvar, y=yvar, hue='Eye', sharey=True, markers=["o","x"], **kwargs)
    g.set(xlim=[.18, .23])
    g.fig.suptitle(', '.join(gv), fontsize=16, y=0.97)
    g.fig.subplots_adjust(top=.9, right=.8)
    g.set_axis_labels(x_lbl, y_lbl[gv[-1]])
    plt.close(g.fig)
    return(g)

def gaba_vs_psychophys_plot_2line_nofacet(gv, gr, **kwargs):
    xvar = "GABA"
    yvar = "value"
    g = sns.lmplot(data=gr, 
              x=xvar, y=yvar,hue="Eye", sharey=True, markers=["x","o"], **kwargs)
    g.set_axis_labels('', '')
    #g.set_xticklabels([])
    #g.set_yticklabels([])
    #g.ax.set_ylim([-1, 3])
    #g.set_titles(None)
    g.fig.suptitle(', '.join(gv), fontsize=16, y=0.97)
    plt.close(g.fig)
    return(g)

def gaba_vs_oss_plot_2line(gv, gr):
    xvar = "GABA"
    x_lbl = "GABA (relative to creatine)"
    yvar = "value"
    y_lbl = {'BaselineThresh':'Baseline Threshold (C%), Iso/Cross ratio',
            'RelMCToPred':'Relative Mask Contrast to predict threshold at, Iso/Cross ratio',
            'ThreshPredCritical':'Predicted threshold elevation (multiples of baseline), Iso/Cross ratio',
            'DepthOfSuppressionPred':'Depth of suppression (multiples of baseline threshold)\nIso/Cross ratio',
            'ThreshPredCriticalUnnorm':'Predicted threshold elevation (C%) Iso/Cross ratio',
            'slope':'Slope of perceptual suppression fit line, Iso/Cross ratio',
            'y_int':'y-intercept of perceptual suppression fit line, Iso/Cross ratio'}
    g = sns.lmplot(data=gr, 
              row='Presentation',col='Population',# facet rows and columns
              x=xvar, y=yvar,hue="Eye",sharey=True, markers=["o","x"])
    g.fig.suptitle(', '.join(gv), fontsize=16, y=0.97)
    g.fig.subplots_adjust(top=.9, right=.8)
    g.set_axis_labels(x_lbl, y_lbl[gv[-1]])
    plt.close(g.fig)
    return(g)

def gaba_vs_psychophys_plot_2line_2eye(gv, gr, **kwargs):
    xvar = "GABA"
    x_lbl = "GABA (relative to creatine)"
    yvar = "Nde-De"
    y_lbl = {'BaselineThresh':'Interocular Difference in Baseline Threshold (NDE-DE, C%)',
            'RelMCToPred':'Relative Mask Contrast to predict threshold at',
            'ThreshPredCritical':'Interocular difference in predicted threshold elevation (NDE-DE, multiples of baseline)',
            'DepthOfSuppressionPred':'Interocular difference in predicted depth of suppression\n(in multiples of baseline threshold)',
            'ThreshPredCriticalUnnorm':'Interocular difference in predicted threshold elevation (NDE-DE, C%)',
            'slope':'Interocular difference in slope of perceptual suppression fit line',
            'y_int':'Interocular difference in y-intercept of perceptual suppression fit line'}
    g = sns.lmplot(data=gr, 
                  col='Presentation',hue='Population',# facet rows and columns
                  x=xvar, y=yvar,sharey=True, **kwargs)
    g.fig.suptitle(':'.join(gv), fontsize=16, y=0.97)
    g.fig.subplots_adjust(top=.9, right=.8)
    g.set_axis_labels(x_lbl, y_lbl[gv[-1]])
    plt.close(g.fig)
    return(g)

def oss_plot_2eye(gv, gr, **kwargs):
    xvar = "GABA"
    x_lbl = "GABA (relative to creatine)"
    yvar = "value"
    g = sns.lmplot(data=gr, 
                  col='Presentation',hue='Population',# facet rows and columns
                  x=xvar, y=yvar,sharey=True, **kwargs)
    g.fig.suptitle(':'.join(gv), fontsize=16, y=0.97)
    g.fig.subplots_adjust(top=.9, right=.8)
    g.set_axis_labels(x_lbl, "Mean Iso/Cross OSS ratio across both eyes")
    plt.close(g.fig)
    return(g)

def gaba_vs_psychophys_plot_2line_2eye_nofacet(gv, gr, **kwargs):
    xvar = "GABA"
    yvar = "Nde-De"
    sns.set_palette('tab10')
    g = sns.lmplot(data=gr, 
                  x=xvar, y=yvar,sharey=True, **kwargs)
    #g.fig.suptitle(':'.join(gv), fontsize=16, y=0.97)
    #g.fig.subplots_adjust(top=.9, right=.8)
    #g.set_axis_labels(x_lbl, y_lbl[gv[-1]])
    g.set_axis_labels('', '')
    g.set_xticklabels([])
    g.set_yticklabels([])
    g.ax.set_ylim([-2, 3])
    plt.close(g.fig)
    return(g)

def gaba_vs_psychophys_plot_4line(gv, gr):
    xvar = "GABA"
    x_lbl = "GABA (relative to creatine)"
    yvar = "value"
    y_lbl = {'BaselineThresh':'Baseline Threshold (C%)',
            'ThreshPredCritical':'Predicted Threshold Elevation (multiples of baseline)',
            'ThreshPredCriticalUnnorm':'Predicted Threshold Elevation (C%)'}
    g = sns.lmplot(data=gr, 
                  row='Presentation',col='measure',# facet rows and columns
                  x=xvar, y=yvar,hue="Trace",sharey=True, ci=False, markers=["o","x", "o", "x"], line_kws={"linestyle": "dotted"})
    g.fig.suptitle(gv, fontsize=16, y=0.97)
    g.fig.subplots_adjust(top=.9, right=.8)
    g.set_axis_labels(x_lbl, y_lbl[gv[2]])
    g.fig.set_figwidth(10)
    plt.close(g.fig)
    return(g)

def group_facet_plots(df, plot_func, ofn, grouping_vars, row, col, x, y, col_wrap=None, hue=None, legend=True, **kwargs):
    with PdfPages(ofn) as pdf:
        grouped = df.groupby(grouping_vars)
        for gv, gr in grouped: # each page
            g = sns.FacetGrid(gr, row=row, col=col, hue=hue, col_wrap=col_wrap, height=6, aspect=1.5, sharex=False, sharey=True, margin_titles=True)
            g = g.map_dataframe(plot_func,x,y,**kwargs)
            print('Plotting %s'%'.'.join(gv))
            if legend:
                g = g.add_legend()
            g.fig.suptitle(':'.join(gv), fontsize=14, y=0.97)
            g.fig.subplots_adjust(top=.9, right=.8)
            x_lbl = "Relative Mask Contrast" if x=="RelMaskContrast" else x
            y_lbl = "Threshold Elevation (multiples of baseline)" if y=="ThreshElev" else y
            g.set_axis_labels(x_lbl, y_lbl)
            pdf.savefig(g.fig)
            plt.close(g.fig)
    print('Plots saved at',ofn)
    plt.close('all')