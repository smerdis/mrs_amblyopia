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

import utils

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
            x_lbl = "Relative Surround Contrast (multiples of baseline)" if x=="RelMaskContrast" else x
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
    if kwargs['pvals'] is not None:
        pvals = kwargs.pop("pvals") #pvals from bootstrap
    else:
        pvals = None
    colors = kwargs['palette']
    n_thistrace = len(tracecol)
    assert(n_thistrace==len(xcol)==len(ycol))
    if n_thistrace > 2:
        if trace == "Persons with\nAmblyopia, DE" or trace=="Normally-sighted\npersons, DE":
            pos = (0.5, 0.9)
            if trace == "Persons with\nAmblyopia, DE":
                color = colors[0]
                pval = pvals[0]
            if trace=="Normally-sighted\npersons, DE":
                color = colors[2]
                pval = pvals[2]
        if trace == "Persons with\nAmblyopia, NDE" or trace=="Normally-sighted\npersons, NDE":
            pos = (0.5, 0.85)
            if trace == "Persons with\nAmblyopia, NDE":
                color=colors[1]
                pval=pvals[1]
            if trace=="Normally-sighted\npersons, NDE":
                color=colors[3]
                pval=pvals[3]
        annotation = fr"N={n_thistrace}, $\rho$={rho_result.correlation:.2f}, p={pval:.2f}"
        ax.text(*pos, annotation, fontsize=16, transform=ax.transAxes, fontdict={'color': color}, horizontalalignment='center')

def gaba_vs_psychophys_plot(gv, gr, legend_box = [0.89, 0.55, 0.1, 0.1], legend_img = True, log = False, ylim = None, **kwargs):
    """Plotting function for GABA vs. psychophysical measures, with annotations etc."""
    print(gv)#, gr)
    with sns.plotting_context(context="paper", font_scale=1.0):
        xvar = "GABA"
        yvar = "value"
        try:
            n_boot = kwargs['n_boot']
        except KeyError:
            n_boot = 1000
        g = sns.lmplot(data=gr, x=xvar, y=yvar, **kwargs)
        iterations, pvals_corrs, pvals_diffs = utils.compare_rs(gr, n_boot=n_boot, resample=False)
        g.map(annotate_n, 'GABA', 'value', 'Trace', pvals=pvals_corrs, palette=kwargs['palette']) #runs on each level of hue in each facet

        if 'SS' in gv and legend_img: # display legend schematic image
            if 'Iso' in gv:
                im = plt.imread(f"/Users/smerdis/Dropbox/Documents/cal/silverlab/mrs-amblyopia/ss-iso-targetandmask-12.png")
            elif 'Cross' in gv:
                im = plt.imread(f"/Users/smerdis/Dropbox/Documents/cal/silverlab/mrs-amblyopia/ss-cross-targetandmask.png")
            else:
                print('Unknown orientation...')

        for axi, ax in enumerate(g.axes.flat): #set various things on each facet
            if log:
                ax.yaxis.set_major_locator(tick.LogLocator(subs=range(1, 10)))
                ax.set_yscale('log')
            if gv[-1] == "ThreshPredCritical":
                ax.axhline(1, color='grey', linestyle='dotted') # facilitation-suppression line
            if ylim is not None:
                if type(ylim[0]) is tuple: # tuple of tuples, i.e. different ylims for amb and con
                    ax.set_ylim(*ylim[axi%2])
                else:
                    ax.set_ylim(*ylim)
            ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
            #ax.yaxis.set_minor_formatter(tick.FormatStrFormatter('%.2f'))
            ax.yaxis.set_minor_formatter(tick.NullFormatter())
            #newax = g.fig.add_axes(legend_box[axi], anchor='NE')
            #newax.imshow(im)
            #newax.axis('off')
            if 'nDicho' in gv:
                ax.legend(loc='lower left', title='Target presented to:\n(other eye viewed surround)')
            elif 'nMono' in gv:
                ax.legend(loc='lower left', title='Target and surround\npresented to:')

        if g._legend:
            g._legend.set_title(f"Target presented to")
            #g._legend.set_bbox_to_anchor([legend_box[0], legend_box[1]-0.16, legend_box[2], legend_box[3]])

        x_lbl = "GABA:Creatine ratio"
        y_lbl = {'BaselineThresh':'Baseline Threshold (C%)',
                'RelMCToPred':'Relative Surround Contrast to predict threshold at',
                #'ThreshPredCritical':'Predicted threshold elevation, multiples of baseline',
                'ThreshPredCritical':'Predicted threshold elevation, multiples of baseline\n(>1 indicates suppression, <1 facilitation)',
                'DepthOfSuppressionPred':'Depth of suppression, multiples of baseline threshold\nnegative indicates facilitation',
                'ThreshPredCriticalUnnorm':'Predicted threshold elevation (C%)',
                'slope':'Slope of perceptual suppression fit line',
                'y_int':'y-intercept of perceptual suppression fit line',
                'OSSSRatio':'Orientation-selective surround suppression\n(Iso-surround:cross-surround ratio)'}
        g.set_axis_labels(x_lbl, y_lbl[gv[-1]])

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