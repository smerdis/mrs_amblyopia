# suppression.py
#
# Functions to read in, deal with, and plot the psychophysical data extracted from MATLAB

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as tick
import seaborn as sns

import scipy.optimize as so
import lmfit as lf
import itertools as it

## Functions to read input
def load_psychophys(pp_fn):
	df = pd.read_table(pp_fn)
	df['logThreshElev'] = np.log10(df['ThreshElev'])
	return df

def load_gaba(gaba_fn):
	return pd.read_table(gaba_fn)

def load_fmri(fmri_fn):
	return pd.read_table(fmri_fn)


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

def fit_plot(x, y, **kwargs):
	# set up the data frame for plotting, get kwargs etc
	data = kwargs.pop("data")
	fmt_obs = kwargs.pop("fmt_obs")
	fmt_pred = kwargs.pop("fmt_pred")
	ses = data[kwargs.pop("yerr")].values # SE's of actually observed threshold elevations
	predY = data[kwargs.pop("Ycol")].values

	# control the plotting
	ax = plt.gca()
	ax.set_xscale('log')
	ax.set_yscale('log')
	#ax.get_xaxis().set_major_locator(tick.LogLocator(subs=[1, 3]))
	#ax.get_yaxis().set_major_locator(tick.LogLocator(subs=[1, 3]))
	#ax.get_xaxis().set_major_formatter(tick.ScalarFormatter())
	#ax.get_yaxis().set_major_formatter(tick.ScalarFormatter())
	#ax.set_ylim([0.5, np.max([np.max(data[y])+1])])
	#ax.set_xlim([0.5, 10])
	ax.errorbar(data=data, x=x, y=y, yerr=ses,fmt=fmt_obs, **kwargs)
	ax.errorbar(data=data, x=x, y=predY,fmt=fmt_pred, **kwargs)
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
            'ThreshPredCritical':'Predicted threshold elevation (multiples of baseline)',
            'ThreshPredCriticalUnnorm':'Predicted threshold elevation (C%)'}
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
    		'ThreshPredCritical':'Interocular difference in predicted threshold elevation (NDE-DE, multiples of baseline)',
            'ThreshPredCriticalUnnorm':'Interocular difference in predicted threshold elevation (NDE-DE, C%)'}
    g = sns.lmplot(data=gr, 
                  col='Presentation',hue='Population',# facet rows and columns
                  x=xvar, y=yvar,sharey=False, **kwargs)
    #if gv[2]=="ThreshPredCritical":
    #	g.set(yscale='log')
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
		for gv, gr in grouped:
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


## Utility functions to convert between db and C%
def pct_to_db(pct):
	return 20 * np.log10(pct)

def db_to_pct(db):
	return 10**(db/20)


## functions defining ways to model individual condition data
### Two-stage model
def stage1(cmpt_thiseye, cmpt_othereye, mask_thiseye, mask_othereye, w_xm, w_xd):
		m = 1.3 # stage 1 excitatory exponent
		S = 1 # stage 1 saturation constant (does it really =1 though?)
		return (cmpt_thiseye**m)/(S + cmpt_thiseye + cmpt_othereye + w_xm*mask_thiseye + w_xd*mask_othereye)

def two_stage_fac_resp(params, C_thiseye, C_othereye, X_thiseye, X_othereye):
	'''Two-stage model with facilitation - model outputs.
	weights: (w_xm, w_xd, a) mono- and dichoptic-suppression constants, a = facilitation parameter [same for all observations]
	C_thiseye, C_othereye: target contrasts in the two eyes, in percent
	X_thiseye, X_othereye: mask/surround contrasts in the two eyes, in percent

	C's and X's must have the same length (#observations)'''

	assert(len(C_thiseye)==len(C_othereye) & len(C_thiseye)==len(X_thiseye) & len(C_thiseye)==len(X_othereye))

	w_xm = params['w_m']#.value
	w_xd = params['w_d']#.value
	a = params['a']#.value
	k = params['k']#.value
	p = params['p'] #8 # stage 2 excitatory exponent # or 2.4
	q = params['q'] #6.5 # stage 2 suppressive exponent # or 2
	Z = params['Z'] #.0085 # stage 2 saturation constant # or 5, or 1

	responses = np.empty(len(C_thiseye))
	#binsums = np.empty(len(C_thiseye))

	for i,(CDe, CNde, XDe, XNde) in enumerate(zip(C_thiseye, C_othereye, X_thiseye, X_othereye)):
		stage1De_t = stage1(CDe, CNde, XDe, XNde, w_xm, w_xd)
		stage1Nde_t = stage1(CNde, CDe, XNde, XDe, w_xm, w_xd)
		stage1De_m = stage1(XDe, XNde, CDe, CNde, w_xm, w_xd)
		stage1Nde_m = stage1(XNde, XDe, CNde, CDe, w_xm, w_xd)

		binsum_target = stage1De_t + stage1Nde_t
		binsum_mask = stage1De_m + stage1Nde_m

		# model with facilitation, taken from Meese & Baker 2009
		resp_t = ((1 + a*binsum_mask)*(binsum_target**p))/(Z+binsum_target**q)
		responses[i] = resp_t

	return k-responses

#error function to be minimized to obtain threshElev predictions.
def two_stage_fac_thresh(thresh_param, C_othereye, X_thiseye, X_othereye, fitted_params): #threshs will be minimized
	C_thiseye = thresh_param['C_thiseye'].value
	return two_stage_fac_resp(fitted_params, [C_thiseye], C_othereye, X_thiseye, X_othereye)


### Linear model
def linear_nofac_err(params, C_thiseye, C_othereye, X_thiseye, X_othereye):
	'''Simple linear model of threshold elevation. 
	This function returns the residual between
	[the prediction specified by the params (y_int, slope) and mask contrast (Xthis/Xother)]
	and the observed (Cthis)
	params: y-intercept, slope (of the line)
	C_thiseye, C_othereye: target contrasts in the two eyes, in percent
	X_thiseye, X_othereye: mask/surround contrasts in the two eyes, in percent

	C's and X's must have the same length (#observations)'''

	assert(len(C_thiseye)==len(C_othereye) & len(C_thiseye)==len(X_thiseye) & len(C_thiseye)==len(X_othereye))

	y_int = params['y_int']#.value
	slope = params['slope']#.value

	responses = np.empty(len(C_thiseye))

	for i,(Cthis, Cother, Xthis, Xother) in enumerate(zip(C_thiseye, C_othereye, X_thiseye, X_othereye)):
		# we don't have a binocular condition, so one of X_thiseye and X_othereye will be 0
		assert(Cother==0) #Cthis is always the nonzero one
		assert(Xthis==0 or Xother==0) #depends on monocular/dichoptic
		responses[i] = Cthis-(y_int + slope*Xthis + slope*Xother) # since one of the X's will be 0, a term will disappear

	return responses

#error function to be minimized to obtain threshElev predictions.
def linear_nofac_thresh(thresh_param, C_othereye, X_thiseye, X_othereye, fitted_params): #threshs will be minimized
	C_thiseye = thresh_param['C_thiseye'].value
	return linear_nofac_err(fitted_params, [C_thiseye], C_othereye, X_thiseye, X_othereye)

## functions that run the model and get predictions, model-agnostic
def predict_thresh(func, init_guess, C_other, X_this, X_other, fitted_params):
	'''A wrapper function that accepts a threshold-error minimizing function with arguments in a convenient order'''
	thresh_params = lf.Parameters()
	thresh_params.add(name='C_thiseye', value=init_guess, vary=True)
	thresh_fit = lf.minimize(func, thresh_params, args=(C_other, X_this, X_other, fitted_params))
	return thresh_fit.params['C_thiseye'].value

def model_condition(g, err_func, thresh_func, params, ret='preds', predtype='linear', supp_only=False, log=False):
	'''Model a condition. In this case, this function is to be applied to each group, where a group is a particular:
	- Task (OS/SS)
	- Mask Orientation (Iso/Cross)
	- Presentation (nMono/nDicho)
	- Eye (which viewed the target, De/Nde)
	- Population (Con/Amb)

	The values that are then modeled are RelMaskContrast (x) vs ThreshElev (y)

	if ret='weights', returns the weights (and any other fitted parameters) for this group.
	the default is to return the predicted thresholds (ret='preds')'''

	print(g.name, g.RelMCToPred.iat[0])

	masks = g.RelMaskContrast
	if log:
		threshs = g.logThreshElev
	else:
		threshs = g.ThreshElev
	thresh_preds = np.empty_like(threshs)
	thresh_preds[:] = np.nan
	if supp_only: #remove the facilitation part of the data, and points before it
		fac_idxs = np.where(threshs <= 1)[0]
		if fac_idxs.size==0:
			last_fac_idx = -1
		else:
			last_fac_idx = fac_idxs[-1]
		threshs = threshs[last_fac_idx+1:]
		masks = masks[last_fac_idx+1:]
		if len(masks) <= 1: # there is only one data point after the facilitation regime, so we can't estimate a slope
			if ret=='preds':
				g['ThreshPred'] = thresh_preds # should be nan's
				return g
			else:
				return None

	assert(np.all(g.Eye==g.Eye.iloc[0])) # Make sure we only are looking at data for one eye
	Eye = g.Eye.iloc[0]
	assert(np.all(g.Presentation==g.Presentation.iloc[0])) # again, one condition only
	Presentation = g.Presentation.iloc[0]
	assert(np.all(g.BaselineThresh==g.BaselineThresh.iloc[0])) # has to be true since we will compute un-normalized threshelev (=C%)
	BaselineThresh = g.BaselineThresh.iloc[0]

	if Presentation=='nDicho':
		contrasts = (threshs.as_matrix(), np.zeros_like(threshs), np.zeros_like(masks), masks.as_matrix())
	elif Presentation=='nMono':
		contrasts = (threshs.as_matrix(), np.zeros_like(threshs), masks.as_matrix(), np.zeros_like(masks))

	params_fit = lf.minimize(err_func, params, args=contrasts)
	pfit = params_fit.params
	pfit.pretty_print()
	#params = fmin(err_func, np.zeros(n_free), args=contrasts) #fitted weights of free parameters of the model, as implemented by err_func
	#print(*zip(free_params,params))
	if predtype=='linear':
		pv = pfit.valuesdict()
		threshpred = [pv['y_int'] + relmc*pv['slope'] for relmc in masks]
		ThreshPredBinCenter = pv['y_int'] + g.RelMCToPred.iat[0]*pv['slope']
		ThreshPredBinCenterUnnorm = ThreshPredBinCenter * BaselineThresh
	else:
		threshpred = [predict_thresh(thresh_func, a,[b],[c],[d],pfit) for a,b,c,d in zip(*contrasts)]
	if ret=='preds':
		if supp_only:
			thresh_preds[last_fac_idx+1:] = threshpred
		else:
			thresh_preds = threshpred
		g['ThreshPred'] = thresh_preds
		return g
	elif ret=='weights':
		retvars = pd.Series(pfit.valuesdict(), index=params.keys())
		if ThreshPredBinCenter:
			retvars['ThreshPredCritical'] = ThreshPredBinCenter
			retvars['ThreshPredCriticalUnnorm'] = ThreshPredBinCenterUnnorm
		return retvars