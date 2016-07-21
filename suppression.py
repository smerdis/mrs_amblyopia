# suppression.py
#
# Functions to read in, deal with, and plot the psychophysical data extracted from MATLAB

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from scipy.optimize import fmin

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

def fit_plot(x, y, **kwargs):
	data = kwargs.pop("data")
	fmt_obs = kwargs.pop("fmt_obs")
	fmt_pred = kwargs.pop("fmt_pred")
	ses = data[kwargs.pop("yerr")].values # SE's of actually observed threshold elevations
	predY = data[kwargs.pop("Ycol")].values
	plt.errorbar(data=data, x=x, y=y, yerr=ses,fmt=fmt_obs,**kwargs)
	plt.errorbar(data=data, x=x, y=predY,fmt=fmt_pred,**kwargs)

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
	print('Plots saved at',ofn)
	plt.close('all')

def pct_to_db(pct):
	return 20 * np.log10(pct)

def db_to_pct(db):
	return 10**(db/20)

def stage1(cmpt_thiseye, cmpt_othereye, mask_thiseye, mask_othereye, w_xm, w_xd):
		m = 1.3 # stage 1 excitatory exponent
		S = 1 # stage 1 saturation constant (does it really =1 though?)
		return (cmpt_thiseye**m)/(S + cmpt_thiseye + cmpt_othereye + w_xm*mask_thiseye + w_xd*mask_othereye)

def two_stage_nofac_resp(weights, C_thiseye, C_othereye, X_thiseye, X_othereye):
	'''Calculate the output of the two-stage model of suppression (without facilitation) for a number of observations:
	weights: (w_xm, w_xd) mono- and dichoptic-suppression constants [same for all observations]
	C_thiseye, C_othereye: target contrasts in the two eyes, in percent
	X_thiseye, X_othereye: mask/surround contrasts in the two eyes, in percent

	C's and X's must have the same length (#observations)'''

	assert(len(C_thiseye)==len(C_othereye) & len(C_thiseye)==len(X_thiseye) & len(C_thiseye)==len(X_othereye))

	w_xm = weights[0]
	w_xd = weights[1]
	p = 8 # stage 2 excitatory exponent # or 2.4
	q = 6.6 # stage 2 suppressive exponent # or 2
	Z = .08 # stage 2 saturation constant # or 5, or 1

	responses = np.empty(len(C_thiseye))
	binsums = np.empty(len(C_thiseye))

	for i,(CDe, CNde, XDe, XNde) in enumerate(zip(C_thiseye, C_othereye, X_thiseye, X_othereye)):
		stage1De_t = stage1(CDe, CNde, XDe, XNde, w_xm, w_xd)
		stage1Nde_t = stage1(CNde, CDe, XNde, XDe, w_xm, w_xd)
		#stage1De_m = stage1(XDe, XNde, CDe, CNde, w_xm, w_xd)
		#stage1Nde_m = stage1(XNde, XDe, CNde, CDe, w_xm, w_xd)
		#print stage1De_t, stage1Nde_t, stage1De_m, stage1Nde_m

		binsum_target = stage1De_t + stage1Nde_t
		#binsum_mask = stage1De_m + stage1Nde_m
		#print binsum_target, binsum_mask

		# remove fac param
		resp_t = (binsum_target**p)/(Z+binsum_target**q)
		responses[i] = resp_t
		binsums[i] = binsum_target

	return responses, binsums

def two_stage_nofac_error(weights, C_thiseye, C_othereye, X_thiseye, X_othereye):
	# The function which is to be minimized to find the optimal model weights.
	# k is taken from Meese & Baker
	k = 0.2 # proportional to stdev of late additive noise / SNR
	responses, binsums = two_stage_nofac_resp(weights, C_thiseye, C_othereye, X_thiseye, X_othereye)
	return abs(k-responses).sum()

#error function to be minimized to obtain threshElev predictions... (speculating)
def two_stage_nofac_thresh(C_thiseye, C_othereye, X_thiseye, X_othereye, w_m, w_d): #threshs will be minimized
	k = 0.2
	responses, binsums = two_stage_nofac_resp((w_m, w_d), C_thiseye, C_othereye, X_thiseye, X_othereye)
	return abs(k-responses).sum()

def two_stage_fac_resp(weights, C_thiseye, C_othereye, X_thiseye, X_othereye):
	'''Two-stage model with facilitation - model outputs.
	weights: (w_xm, w_xd, a) mono- and dichoptic-suppression constants, a = facilitation parameter [same for all observations]
	C_thiseye, C_othereye: target contrasts in the two eyes, in percent
	X_thiseye, X_othereye: mask/surround contrasts in the two eyes, in percent

	C's and X's must have the same length (#observations)'''

	assert(len(C_thiseye)==len(C_othereye) & len(C_thiseye)==len(X_thiseye) & len(C_thiseye)==len(X_othereye))

	w_xm = weights[0]
	w_xd = weights[1]
	a = weights[2]
	p = 8 # stage 2 excitatory exponent # or 2.4
	q = 6.5 # stage 2 suppressive exponent # or 2
	Z = .0085 # stage 2 saturation constant # or 5, or 1

	responses = np.empty(len(C_thiseye))
	binsums = np.empty(len(C_thiseye))

	for i,(CDe, CNde, XDe, XNde) in enumerate(zip(C_thiseye, C_othereye, X_thiseye, X_othereye)):
		stage1De_t = stage1(CDe, CNde, XDe, XNde, w_xm, w_xd)
		stage1Nde_t = stage1(CNde, CDe, XNde, XDe, w_xm, w_xd)
		stage1De_m = stage1(XDe, XNde, CDe, CNde, w_xm, w_xd)
		stage1Nde_m = stage1(XNde, XDe, CNde, CDe, w_xm, w_xd)

		binsum_target = stage1De_t + stage1Nde_t
		binsum_mask = stage1De_m + stage1Nde_m

		# model with facilitation, taken from Meese & Baker 2009
		resp_t = (1 + a*binsum_mask)*(binsum_target**p)/(Z+binsum_target**q)
		responses[i] = resp_t
		binsums[i] = binsum_target

	return responses, binsums

def two_stage_fac_error(weights, C_thiseye, C_othereye, X_thiseye, X_othereye):
	'''error function to be minimized over a group of observations to determine values of free parameters.

	model with facilitation, so w_m, w_d, a.'''
	# k is taken from Meese & Baker
	k = 0.2 # proportional to stdev of late additive noise / SNR
	responses, binsums = two_stage_fac_resp(weights, C_thiseye, C_othereye, X_thiseye, X_othereye)
	return abs(k-responses).sum()

#error function to be minimized to obtain threshElev predictions.
def two_stage_fac_thresh(C_thiseye, C_othereye, X_thiseye, X_othereye, w_m, w_d, a): #threshs will be minimized
	k = 0.2
	responses, binsums = two_stage_fac_resp((w_m, w_d, a), C_thiseye, C_othereye, X_thiseye, X_othereye)
	return abs(k-responses).sum()

def predict_thresh(func, init_guess, C_other, X_this, X_other, *params):
	'''A wrapper function that accepts a threshold-error minimizing function with arguments in a convenient order'''
	return fmin(func, x0=init_guess, args=(C_other, X_this, X_other, *params))

def model_group_means(g, err_func, thresh_func, ret='preds'):
	'''Model the group means. In this case, this function is to be applied to each group, where a group is a different
	condition of the experiment. In this case, it corresponds to a particular:
	- Task (OS/SS)
	- Presentation (nMono/nDicho)
	- Eye (which viewed the target, De/Nde)
	- Population (Con/Amb)

	The values that are then modeled are RelMaskContrast (x) vs ThreshElev (y)

	if ret='weights', returns the weights (and any other fitted parameters) for this group.
	the default is to return the predicted thresholds (ret='preds')'''

	import inspect

	free_params = inspect.getargspec(thresh_func).args[4:] # 0-3 are C_thiseye, C_othereye, X_thiseye, X_othereye
	n_free = len(free_params)

	print(g.name)

	masks = g.RelMaskContrast
	threshs = g.ThreshElev

	assert(np.all(g.Eye==g.Eye.iloc[0])) # Make sure we only are looking at data for one eye
	Eye = g.Eye.iloc[0]
	assert(np.all(g.Presentation==g.Presentation.iloc[0])) # again, one condition only
	Presentation = g.Presentation.iloc[0]

	if Presentation=='nDicho':
		contrasts = (threshs.as_matrix(), np.zeros_like(threshs), np.zeros_like(masks), masks.as_matrix())
	elif Presentation=='nMono':
		contrasts = (threshs.as_matrix(), np.zeros_like(threshs), masks.as_matrix(), np.zeros_like(masks))

	params = fmin(err_func, np.zeros(n_free), args=contrasts) #fitted weights of free parameters of the model, as implemented by err_func
	print(*zip(free_params,params))
	threshpred = [predict_thresh(thresh_func, [1],[b],[c],[d],*params)[0] for a,b,c,d in zip(*contrasts)]

	if ret=='preds':
		g['ThreshPred'] = threshpred
		return g
	elif ret=='weights':
		return pd.Series([params], index=free_params)