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

import scipy.stats as st

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
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_ylim([0.6, 4.2])
    # ax.get_xaxis().set_major_locator(tick.LogLocator(subs=[1,2,3,4,5,6,7,8,9]))
    # ax.set_xticklabels([])
    # ax.get_yaxis().set_major_locator(tick.LogLocator(subs=[1,2,3,4,5,6,7,8,9]))
    # ax.set_yticklabels([])
    # ax.get_yaxis().set_major_formatter(tick.NullFormatter())
    # ax.get_yaxis().set_minor_formatter(tick.NullFormatter())

    # Plot the eyes separately so we can apply different styling
    traces = data.groupby('Trace')
    fmt = {'Amblyope-Nde':':x', 'Amblyope-De':'--s',
           'Control-Nde':':x', 'Control-De':'--s'}
    # for trace, trace_df in traces:
    #     ses = trace_df[yerr].values
    #     ax.errorbar(data=trace_df, x=x, y=y, yerr=ses, fmt=fmt[trace], **kwargs)
    # ax.axhline(y=1, ls='dotted', color='gray')
    return(fig)

def subject_fit_plot(x, y, **kwargs):
    # set up the data frame for plotting, get kwargs etc
    data = kwargs.pop("data")
    fmt_obs = kwargs.pop("fmt_obs")
    fmt_pred = kwargs.pop("fmt_pred")
    ses = data[kwargs.pop("yerr")].values # SE's of actually observed threshold elevations
    predY = kwargs.pop("Ycol")
    print(x, y, ses, predY)

    # control the plotting
    ax = plt.gca()
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.get_xaxis().set_major_locator(tick.LogLocator())
    # ax.get_yaxis().set_major_locator(tick.LogLocator())
    #ax.get_xaxis().set_major_formatter(tick.ScalarFormatter())
    #ax.get_yaxis().set_major_formatter(tick.ScalarFormatter())
    #ax.set_ylim([0.5, np.max([np.max(data[y])+1])])
    #ax.set_xlim([0.5, 10])
    ax.errorbar(data=data, x=x, y=y, yerr=ses,fmt=fmt_obs, **kwargs)
    ax.errorbar(data=data, x=x, y=predY,fmt=fmt_pred, **kwargs)
    ax.axhline(y=1, ls='dotted', color='grey')
    ax.axvline(x=2, ls='dotted', color='red')

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

def group_facet_plots(df, plot_func, ofn, grouping_vars, row, col, x, y, col_wrap=None, hue=None, legend=True, **kwargs):
    with PdfPages(ofn) as pdf:
        grouped = df.groupby(grouping_vars)
        for gv, gr in grouped: # each page
            g = sns.FacetGrid(gr, row=row, col=col, hue=hue, col_wrap=col_wrap, height=6, aspect=1.5,
                sharex=False, sharey=True, margin_titles=False, legend_out=False, palette=['#6600ff', '#009966'])
            g = g.map_dataframe(plot_func,x,y,**kwargs)
            print('Plotting %s'%'.'.join(gv))
            if legend:
                g = g.add_legend(title="")
            g.fig.suptitle('')
            g.fig.subplots_adjust(top=.9, right=.8)
            x_lbl = "Relative Mask Contrast (multiples of baseline)" if x=="RelMaskContrast" else x
            y_lbl = "Threshold Elevation (multiples of baseline)" if y=="ThreshElev" else y
            g.set_axis_labels(x_lbl, y_lbl)
            g.set_titles('')
            pdf.savefig(g.fig)
            plt.close(g.fig)
    print('Plots saved at',ofn)
    plt.close('all')

def annotate_n(xcol, ycol, tracecol, **kwargs):
    """Annotate each level of hue variable on each facet of graph (i.e. multiple times per facet)"""
    ax = plt.gca()
    rho_result = st.spearmanr(xcol, ycol)
    trace = tracecol.unique()[0] # 'Persons with\nAmblyopia, DE' etc
    colors = kwargs['palette']
    n_thistrace = len(tracecol)
    assert(n_thistrace==len(xcol==len(ycol)))
    annotation = f"N={n_thistrace}, rho={rho_result.correlation:.3f}"#, p={rho_result.pvalue:.3f}"
    if trace == "Persons with\nAmblyopia, DE" or trace=="Normally-sighted\npersons, DE":
        pos = (0.5, 0.85)
        if trace == "Persons with\nAmblyopia, DE":
            color = colors[0]
        if trace=="Normally-sighted\npersons, DE":
            color = colors[2]
    if trace == "Persons with\nAmblyopia, NDE" or trace=="Normally-sighted\npersons, NDE":
        pos = (0.5, 0.8)
        if trace == "Persons with\nAmblyopia, NDE":
            color=colors[1]
        if trace=="Normally-sighted\npersons, NDE":
            color=colors[3]
    ax.text(*pos, annotation, transform=ax.transAxes, fontdict={'color': color}, horizontalalignment='center')

def gaba_vs_psychophys_plot(gv, gr, legend_box = [0.89, 0.55, 0.1, 0.1], legend_img = True, log = False, ylim=(.1, 10), **kwargs):
    """Plotting function for GABA vs. psychophysical measures, with annotations etc."""
    print(gv)
    with sns.plotting_context(context="paper", font_scale=1.2):
        xvar = "GABA"
        yvar = "value"
        g = sns.lmplot(data=gr, x=xvar, y=yvar, **kwargs)
        g.map(annotate_n, 'GABA', 'value', 'Trace', palette=kwargs['palette']) #runs on each level of hue in each facet

        for ax in g.axes.flat: #set various things on each facet
            if log:
                ax.yaxis.set_major_locator(tick.LogLocator(subs=range(1, 10)))
                ax.set_yscale('log')
                ax.set_ylim(ylim)
            if gv[-1] == "ThreshPredCritical":
                ax.axhline(1, color='grey', linestyle='dotted') # facilitation-suppression line
            ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))

        if g._legend:
            g._legend.set_title(f"Target presented to")
            g._legend.set_bbox_to_anchor([legend_box[0], legend_box[1]-0.16, legend_box[2], legend_box[3]])

        g.set(xlim=[.18, .23])

        x_lbl = "GABA:Creatine ratio"
        y_lbl = {'BaselineThresh':'Baseline Threshold (C%)',
                'RelMCToPred':'Relative Mask Contrast to predict threshold at',
                #'ThreshPredCritical':'Predicted threshold elevation, multiples of baseline',
                'ThreshPredCritical':'Predicted threshold elevation, multiples of baseline\n(>1 indicates suppression, <1 facilitation)',
                'DepthOfSuppressionPred':'Depth of suppression, multiples of baseline threshold\nnegative indicates facilitation',
                'ThreshPredCriticalUnnorm':'Predicted threshold elevation (C%)',
                'slope':'Slope of perceptual suppression fit line',
                'y_int':'y-intercept of perceptual suppression fit line'}
        g.set_axis_labels(x_lbl, y_lbl[gv[-1]])

        if 'SS' in gv and legend_img: # display legend schematic image
            if 'Iso' in gv:
                im = plt.imread(f"/Users/smerdis/Dropbox/Documents/cal/silverlab/mrs-amblyopia/ss-iso-targetandmask.png")
            elif 'Cross' in gv:
                im = plt.imread(f"/Users/smerdis/Dropbox/Documents/cal/silverlab/mrs-amblyopia/ss-cross-targetandmask.png")
            else:
                print('Unknown orientation...')
            newax = g.fig.add_axes(legend_box, anchor='NE')
            newax.imshow(im)
            newax.axis('off')

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
            'ThreshPredCritical':'Interocular difference in\npredicted threshold elevation\n(NDE-DE, multiples of baseline)',
            'DepthOfSuppressionPred':'Interocular difference in predicted depth of suppression\n(in multiples of baseline threshold)',
            'ThreshPredCriticalUnnorm':'Interocular difference in predicted threshold elevation (NDE-DE, C%)',
            'slope':'Interocular difference in slope of perceptual suppression fit line',
            'y_int':'Interocular difference in y-intercept of perceptual suppression fit line'}
    g = sns.lmplot(data=gr, 
                  col='Presentation',hue='Population',# facet rows and columns
                  x=xvar, y=yvar,sharey=True, ci=None, **kwargs)
    g._legend.set_title('')
    g.set_titles('')
    #g.fig.suptitle(':'.join(gv), fontsize=16, y=0.97)
    g.fig.subplots_adjust(left=.15, top=.9, right=.8)
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