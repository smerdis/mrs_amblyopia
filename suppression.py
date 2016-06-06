# suppression.py
#
# Functions to read in, deal with, and plot the psychophysical data extracted from MATLAB

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import itertools as it

def load_psychophys(pp_fn):
	df = pd.read_table(pp_fn)
	# add a column with the log of the relative mask contrast, since we often want to plot this
	df['logRelContrast'] = np.log(df.RelMaskContrast)
	return df

def load_gaba(gaba_fn):
	return pd.read_table(gaba_fn)

def scaterror_plot(x, y, **kwargs):
	#'''This function is designed to be called with FacetGrid.map_dataframe() to make faceted plots of various conditions.'''
    #print x, y, kwargs
    data = kwargs.pop("data")
    ses = data[kwargs.pop("yerr")].values
    plt.errorbar(data=data, x=x, y=y, yerr=ses,**kwargs)

def group_facet_plots(df, plot_func, ofn, grouping_vars, row, col, x, y, col_wrap=None, hue=None, legend=True, **kwargs):
	with PdfPages(ofn) as pdf:
		grouped = df.groupby(grouping_vars)
		for gv, gr in grouped:
			#print 'Plotting %s'%'.'.join(gv)
			g = sns.FacetGrid(gr, row=row, col=col, hue=hue, col_wrap=col_wrap)
			g = g.map_dataframe(plot_func,x,y,**kwargs)
			if legend:
				g = g.add_legend()
			g.fig.suptitle('/'.join(gv), fontsize=12, y=1.00)
			pdf.savefig(g.fig)
	print 'Plots saved at %s'%ofn
	plt.close('all')

def model_subj(g):
    '''This function defines how to model a individual group of observations and what to return.
    
    In this case a group is a single subject, eye, task condition.
    We model the threshold elevation as a linear function of (log) relative mask contrast,
    and return the slope of the line and the r-value...'''
    import scipy.stats as st
    slope, intercept, r_value, p_value, std_err = st.linregress(g.logRelContrast, g.ThreshElev) # x, y
    return pd.Series([slope, r_value], index=['slope_lm','R_lm'])