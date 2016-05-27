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

def scaterror_plot(x, y, **kwargs):
	#'''This function is designed to be called with FacetGrid.map_dataframe() to make faceted plots of various conditions.'''
    #print x, y, kwargs
    data = kwargs.pop("data")
    ses = data[kwargs.pop("yerr")].values
    plt.errorbar(data=data, x=x, y=y, yerr=ses,fmt='o')

def group_facet_plots(df, plot_func, ofn, grouping_vars, *xy, **kwargs):
	with PdfPages(ofn) as pdf:
		grouped = df.groupby(grouping_vars)
		for gv, gr in grouped:
			#print kwargs
			row = kwargs["row"]
			col = kwargs["col"]
			hue = kwargs["hue"]
			g = sns.FacetGrid(gr, row=row, col=col, hue=hue)
			g.map_dataframe(plot_func,*xy,**kwargs)
			g.fig.suptitle('/'.join(gv), fontsize=12, y=1.00)
			pdf.savefig(g.fig)
	print 'Plots saved at %s'%ofn