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
			g = sns.FacetGrid(gr, row=row, col=col, hue=hue, col_wrap=col_wrap, size=3, aspect=1.5, sharex=False, sharey=False, margin_titles=True)
			#xlim=(3,110), ylim=(.5,30)
			g.set(xscale='log', yscale='log')
			g = g.map_dataframe(plot_func,x,y,**kwargs)
			if legend:
				g = g.add_legend()
			g.fig.suptitle(gv, fontsize=16, y=0.97)
			g.fig.subplots_adjust(top=.9, right=.8)
			pdf.savefig(g.fig)
	print 'Plots saved at %s'%ofn
	plt.close('all')

def pct_to_db(pct):
	return 20 * np.log10(pct)

def db_to_pct(db):
	return 10**(db/20)

def stage1(cmpt_thiseye, cmpt_othereye, mask_thiseye, mask_othereye, w_xm, w_xd):
		m = 1.3 # stage 1 excitatory exponent
		S = 1 # stage 1 saturation constant (does it really =1 though?)
		return (cmpt_thiseye**m)/(S + cmpt_thiseye + cmpt_othereye + w_xm*mask_thiseye + w_xd*mask_othereye)

def two_stage_model_responses(weights, C_thiseye, C_othereye, X_thiseye, X_othereye):
	'''Calculate the resp(target) [output of the two-stage model of suppression] for a number of observations:
	weights: (w_xm, w_xd) mono- and dichoptic-suppression constants [same for all observations]
	C_thiseye, C_othereye: target contrasts in the two eyes, in percent
	X_thiseye, X_othereye: mask/surround contrasts in the two eyes, in percent

	C's and X's must have the same length (#observations)'''

	assert(len(C_thiseye)==len(C_othereye) & len(C_thiseye)==len(X_thiseye) & len(C_thiseye)==len(X_othereye))

	w_xm = weights[0]
	w_xd = weights[1]
	p = 8 # stage 2 excitatory exponent
	q = 6.5 # stage 2 suppressive exponent
	Z = 0.0085 # stage 2 saturation constant

	responses = np.empty(len(C_thiseye))

	for i,(CDe, CNde, XDe, XNde) in enumerate(zip(C_thiseye, C_othereye, X_thiseye, X_othereye)):
		#stage1De_t = CDe**m/(S + CDe + CNde + w_xm*XDe + w_xd*XNde) # within-eye supp. weight * XDe because this is left eye
		stage1De_t = stage1(CDe, CNde, XDe, XNde, w_xm, w_xd)
		#stage1Nde_t = CNde**m/(S + CDe + CNde + w_xd*XDe + w_xm*XNde) # within-eye supp. weight * XNde because this is right eye
		stage1Nde_t = stage1(CNde, CDe, XNde, XDe, w_xm, w_xd)
		#stage1De_m = XDe**m/(S + XDe + XNde + w_xm*CDe + w_xd*CNde) # now swap the C's and X's to get the response to the other component
		stage1De_m = stage1(XDe, XNde, CDe, CNde, w_xm, w_xd)
		#stage1Nde_m = XNde**m/(S + XDe + XNde + w_xd*CDe + w_xm*CNde)
		stage1Nde_m = stage1(XNde, XDe, CNde, CDe, w_xm, w_xd)
		#print stage1De_t, stage1Nde_t, stage1De_m, stage1Nde_m

		binsum_target = stage1De_t + stage1Nde_t
		binsum_mask = stage1De_m + stage1Nde_m
		#print binsum_target, binsum_mask

		# model with facilitation, taken from Meese & Baker 2009
		# resp_t = (1 + a*binsum_mask)*(binsum_target**p)/(Z+binsum_target**q)
		# remove fac param
		resp_t = (binsum_target**p)/(Z+binsum_target**q)
		responses[i] = resp_t

	return responses

def two_stage_model_error(weights, C_thiseye, C_othereye, X_thiseye, X_othereye):
	# The function which is to be minimized to find the optimal model weights.
	# k is taken from Meese & Baker
	k = 0.2 # proportional to stdev of late additive noise / SNR

	responses = two_stage_model_responses(weights, C_thiseye, C_othereye, X_thiseye, X_othereye)

	#print responses
	return abs(k-responses).sum()

def model_subj(g):
    '''This function defines how to model a individual group of observations and what to return.
    
    In this case a group is a single subject, eye, task condition.
    We model the threshold elevation as a linear function of (log) relative mask contrast,
    and return the slope of the line and the r-value...'''
    import scipy.stats as st
    slope, intercept, r_value, p_value, std_err = st.linregress(g.logRelContrast, g.ThreshElev) # x, y
    return pd.Series([slope, r_value], index=['slope_lm','R_lm'])