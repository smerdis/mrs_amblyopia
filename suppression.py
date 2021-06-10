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
            x_lbl = "Relative surround Contrast (multiples of baseline)" if x=="RelMaskContrast" else x
            y_lbl = "Relative threshold, multiples of baseline\n(>1 indicates suppression, <1 facilitation)" if y=="ThreshElev" else y
            g.set_axis_labels(x_lbl, y_lbl)
            g.set_titles('')
            pdf.savefig(g.fig)
            plt.close(g.fig)
    print('Plots saved at',ofn)
    plt.close('all')

def annotate_facet(xcol, ycol, tracecol, what='rho', **kwargs):
    """Annotate each level of hue variable on each facet of graph (i.e. multiple times per facet)"""
    print(f"Annotating\n{tracecol.iloc[0]}\n{kwargs}")
    ax = plt.gca()
    trace = tracecol.unique()[0] # 'Persons with\nAmblyopia, DE' etc
    if kwargs['pvals'] is not None:
        pvals = kwargs.pop("pvals") #pvals from bootstrap
    else:
        pvals = None
    pal = kwargs['palette']
    n_thistrace = len(tracecol)
    assert(n_thistrace==len(xcol)==len(ycol))
    if n_thistrace > 2:
        if trace == "Persons with\nAmblyopia, DE" or trace=="Normally-sighted\npersons, DE":
            pos = (0.5, 0.9)
            if trace == "Persons with\nAmblyopia, DE":
                pval = pvals[0]
            if trace=="Normally-sighted\npersons, DE":
                pval = pvals[2]
        if trace == "Persons with\nAmblyopia, NDE" or trace=="Normally-sighted\npersons, NDE":
            pos = (0.5, 0.85)
            if trace == "Persons with\nAmblyopia, NDE":
                pval=pvals[1]
            if trace=="Normally-sighted\npersons, NDE":
                pval=pvals[3]
        color = pal[trace]
        if what=='rho':
            result = st.spearmanr(xcol, ycol)
            annotation = fr"N={n_thistrace}, $\rho$={result.correlation:.2f}, p={pval:.2f}"
        elif what=='slope':
            result = st.linregress(xcol, ycol)
            annotation = fr"N={n_thistrace}, slope={result.slope:.2f}, p={pval:.2f}"
        ax.text(*pos, annotation, fontsize=16, transform=ax.transAxes, fontdict={'color': color}, horizontalalignment='center')

def annotate_facet_df(xcol, ycol, tracecol, what='rho', **kwargs):
    """Annotate each level of hue variable on each facet of graph (i.e. multiple times per facet)"""
    #print(f"Annotating\n{tracecol}\n{xcol}\n{ycol}\n{kwargs}")
    ax = plt.gca()
    if kwargs['pvals'] is not None:
        pvals = kwargs.pop("pvals") #pvals from bootstrap
    else:
        pvals = None
    data = kwargs.pop("data") #pvals from bootstrap
    trace = data[tracecol].unique()[0] # 'Persons with\nAmblyopia, DE' etc
    pal = kwargs['palette']
    presentation = kwargs.pop("presentation") #pvals from bootstrap
    n_thistrace = len(data[tracecol])
    assert(n_thistrace==len(data[xcol])==len(data[ycol]))
    if n_thistrace > 2:
        # if this graph has 8 lines (2 presentation conditions x 2 populations x 2 eyes), annotate only the correct 4, which are in kwarg 'presentation'
        # if this graph has 4 lines (2 populations x 2 eyes for one presentation condition, which is then not in data.columns), annotate it
        if ('Presentation' in data.columns and data.Presentation.iloc[0]==presentation) or ('Presentation' not in data.columns):
            if trace == "Persons with\nAmblyopia, DE" or trace=="Normally-sighted\npersons, DE":
                pos = (0.5, 0.9)
                if trace == "Persons with\nAmblyopia, DE":
                    pval = pvals[0]
                if trace=="Normally-sighted\npersons, DE":
                    pval = pvals[2]
            if trace == "Persons with\nAmblyopia, NDE" or trace=="Normally-sighted\npersons, NDE":
                pos = (0.5, 0.85)
                if trace == "Persons with\nAmblyopia, NDE":
                    pval=pvals[1]
                if trace=="Normally-sighted\npersons, NDE":
                    pval=pvals[3]
            color = pal[trace]
            if what=='rho':
                result = st.spearmanr(data[xcol], data[ycol])
                annotation = fr"N={n_thistrace}, $\rho$={result.correlation:.2f}, p={pval:.2f}"
            elif what=='slope':
                result = st.linregress(data[xcol], data[ycol])
                annotation = fr"N={n_thistrace}, slope={result.slope:.2f}, p={pval:.2f}"
            ax.text(*pos, annotation, fontsize=12, transform=ax.transAxes, fontdict={'color': color}, horizontalalignment='center')


def gaba_vs_psychophys_plot(gv, gr, legend_box = [0.89, 0.55, 0.1, 0.1], legend_img = True, log = False, ylim = None, annotate=True, boot_func=utils.compare_rs, **kwargs):
    """Plotting function for GABA vs. psychophysical measures, with annotations etc."""
    print(gv)#, gr)
    with sns.plotting_context(context="paper", font_scale=1.0):
        xvar = "GABA"
        yvar = "value"
        try:
            n_boot = kwargs['n_boot']
        except KeyError:
            n_boot = 1000
        if boot_func == utils.compare_rs:
            what='rho'
        elif boot_func == utils.compare_slopes:
            what = 'slope'
        g = sns.lmplot(data=gr, x=xvar, y=yvar, **kwargs)
        if annotate:
            anno_groups = gr.groupby(['Task','Orientation','Presentation'])
            for agv, agr in anno_groups:
                print(agv[-1])
                iterations, pvals_corrs, pvals_diffs = boot_func(agr, n_boot=n_boot, verbose=False, resample=False)
                #g.map(annotate_facet, 'GABA', 'value', 'Trace', what=what, pvals=pvals_corrs, palette=kwargs['palette']) #runs on each level of hue in each facet
                g.map_dataframe(annotate_facet_df, 'GABA', 'value', 'Trace', what=what, pvals=pvals_corrs, palette=kwargs['palette'], presentation=agv[-1]) #runs on each level of hue in each facet

        # if 'SS' in gv and legend_img: # display legend schematic image
        #     if 'Iso' in gv:
        #         im = plt.imread(f"/Users/smerdis/Dropbox/Documents/cal/silverlab/mrs-amblyopia/ss-iso-targetandmask-12.png")
        #     elif 'Cross' in gv:
        #         im = plt.imread(f"/Users/smerdis/Dropbox/Documents/cal/silverlab/mrs-amblyopia/ss-cross-targetandmask.png")
        #     else:
        #         print('Unknown orientation...')

        for axi, ax in enumerate(g.axes.flat): #set various things on each facet
            if log:
                ax.set_yscale('log')
            if gv[-1] in ("ThreshPredCritical", "ThreshElev"): # assumes 'measure' is last grouping variable
                ax.axhline(1, color='grey', linestyle='dotted') # facilitation-suppression line
            if ylim is not None:
                if type(ylim[0]) is tuple: # tuple of tuples, i.e. different ylims for amb and con
                    ax.set_ylim(*ylim[axi%2])
                else:
                    ax.set_ylim(*ylim)
            ax.yaxis.set_major_locator(tick.FixedLocator([0.5, 1, 2, 3, 5, 10]))
            ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
            #ax.yaxis.set_minor_formatter(tick.FormatStrFormatter('%.1f'))
            ax.yaxis.set_minor_formatter(tick.NullFormatter())
            #newax = g.fig.add_axes(legend_box[axi], anchor='NE')
            #newax.imshow(im)
            #newax.axis('off')
            if 'nDicho' in gv:
                ax.legend(loc='lower left', title='Annulus presented to:\n(other eye viewed surround)')
            elif 'nMono' in gv:
                ax.legend(loc='lower left', title='Annulus and surround\npresented to:')
            else: # page grouping variables don't include presentation, i.e. there are 4 subplots on one page
                ax.legend(loc='lower left', title='Annulus presented to:')

        if g._legend:
            g._legend.set_title(f"Target presented to")
            #g._legend.set_bbox_to_anchor([legend_box[0], legend_box[1]-0.16, legend_box[2], legend_box[3]])

        x_lbl = "GABA:Creatine ratio"
        y_lbl = {'BaselineThresh':'Baseline contrast discrimination threshold (C%)',
                'ThreshElev':'Relative threshold\n(multiples of baseline)',
                'RelMCToPred':'Relative Surround Contrast to predict threshold at',
                #'ThreshPredCritical':'Predicted threshold elevation, multiples of baseline',
                'ThreshPredCritical':'Relative threshold, multiples of baseline\n(>1 indicates suppression, <1 facilitation)',
                'DepthOfSuppressionPred':'Depth of suppression, multiples of baseline threshold\nnegative indicates facilitation',
                'ThreshPredCriticalUnnorm':'Predicted threshold elevation (C%)',
                'slope':'Slope of perceptual suppression fit line',
                'y_int':'y-intercept of perceptual suppression fit line',
                'OSSSRatio':'Orientation-selective surround suppression\n(Iso-surround:cross-surround ratio)'}
        g.set_axis_labels(x_lbl, y_lbl[gv[-1]])

    plt.close(g.fig)
    return(g)
