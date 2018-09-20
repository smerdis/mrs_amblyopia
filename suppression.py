# suppression.py
#
# Functions to read in, deal with, and plot the psychophysical data extracted from MATLAB

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as tick
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
    plt.errorbar(data=data, x=x, y=y, yerr=ses, fmt='--o', **kwargs)
    plt.axhline(y=1, ls='dotted', color='gray')

def subject_fit_plot(x, y, **kwargs):
    # set up the data frame for plotting, get kwargs etc
    data = kwargs.pop("data")
    fmt_obs = kwargs.pop("fmt_obs")
    fmt_pred = kwargs.pop("fmt_pred")
    ses = data[kwargs.pop("yerr")].values # SE's of actually observed threshold elevations
    predY = data[kwargs.pop("Ycol")].values
    relMCToPred = data['RelMCToPred'].values
    assert(np.all(relMCToPred==relMCToPred[0])) # check all the same

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
    ax.axvline(x=relMCToPred[0], ls='dotted')

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
        relMCToPred = gpop['RelMCToPred'].values
        ax.axvline(x=relMCToPred[0], ls='dotted', color=pop_colors[gvpop])
        for gv, g in gpop.groupby(["Subject"]):
            ses = g[yerr].values # SE's of actually observed threshold elevations
            predY = g[ycol].values
            assert(np.all(relMCToPred==relMCToPred[0])) # check all the same
            
            ax.errorbar(data=g, x=x, y=y, yerr=ses,fmt=fmt_obs, **kwargs)
            ax.errorbar(data=g, x=x, y=predY,fmt=fmt_pred, **kwargs)
            ax.axhline(y=1,ls='dotted')

def gaba_plot(x, y, **kwargs):
    # set up the data frame for plotting, get kwargs etc
    data = kwargs.pop("data")
    fmt_obs = kwargs.pop("fmt_obs")
    print(x, y)

    # control the plotting
    ax = plt.gca()
    #sns.regplot(data=data, x=x, y=y, fit_reg=True, ci=None, dropna=True)
    sns.kdeplot(data[x], data[y])

def gaba_vs_psychophys_plot_2line(gv, gr):
    xvar = "GABA"
    x_lbl = "GABA (relative to creatine)"
    yvar = "value"
    y_lbl = {'BaselineThresh':'Baseline Threshold (C%)',
            'RelMCToPred':'Relative Mask Contrast to predict threshold at',
            'ThreshPredCritical':'Predicted threshold elevation (multiples of baseline)',
            'DepthOfSuppression':'Depth of suppression (multiples of baseline threshold)\nnegative indicates facilitation',
            'ThreshPredCriticalUnnorm':'Predicted threshold elevation (C%)',
            'slope':'Slope of perceptual suppression fit line',
            'y_int':'y-intercept of perceptual suppression fit line'}
    g = sns.lmplot(data=gr, 
              row='Presentation',col='Population',# facet rows and columns
              x=xvar, y=yvar,hue="Eye",sharey=False, markers=["o","x"])
    if gv[2]=="ThreshPredCritical":
        g.set(yscale='log')
        g.set(ylim=[min(gr[yvar])-.1, 1.1*max(gr[yvar])])
    g.fig.suptitle(', '.join(gv), fontsize=16, y=0.97)
    g.fig.subplots_adjust(top=.9, right=.8)
    g.set_axis_labels(x_lbl, y_lbl[gv[2]])
    plt.close(g.fig)
    return(g)

def gaba_vs_psychophys_plot_2line_2eye(gv, gr, **kwargs):
    xvar = "GABA"
    x_lbl = "GABA (relative to creatine)"
    yvar = "Nde-De"
    y_lbl = {'BaselineThresh':'Interocular Difference in Baseline Threshold (NDE-DE, C%)',
            'RelMCToPred':'Relative Mask Contrast to predict threshold at',
            'ThreshPredCritical':'Interocular difference in predicted threshold elevation (NDE-DE, multiples of baseline)',
            'DepthOfSuppression':'Interocular difference in Depth of suppression',
            'ThreshPredCriticalUnnorm':'Interocular difference in predicted threshold elevation (NDE-DE, C%)',
            'slope':'Interocular difference in slope of perceptual suppression fit line',
            'y_int':'Interocular difference in y-intercept of perceptual suppression fit line'}
    g = sns.lmplot(data=gr, 
                  col='Presentation',hue='Population',# facet rows and columns
                  x=xvar, y=yvar,sharey=False, **kwargs)
    #if gv[2]=="ThreshPredCritical":
    #   g.set(yscale='log')
        #g.set(ylim=[min(gr[yvar])-.1, 1.1*max(gr[yvar])])
    g.fig.suptitle(':'.join(gv), fontsize=16, y=0.97)
    g.fig.subplots_adjust(top=.9, right=.8)
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
                  x=xvar, y=yvar,hue="Trace",sharey=False, ci=False, markers=["o","x", "o", "x"], line_kws={"linestyle": "dotted"})
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
            g = sns.FacetGrid(gr, row=row, col=col, hue=hue, col_wrap=col_wrap, size=6, aspect=1.5, sharex=False, sharey=False, margin_titles=True)
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