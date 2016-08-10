# suppression.py
#
# Functions to read in, deal with, and plot the psychophysical data extracted from MATLAB

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

import scipy.optimize as so
from lmfit import minimize, Parameters

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
			#g.set(yscale='log')
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
	Z = .009 # stage 2 saturation constant # or 5, or 1

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
	p = 8 # stage 2 excitatory exponent # or 2.4
	q = 6.5 # stage 2 suppressive exponent # or 2
	Z = .0085 # stage 2 saturation constant # or 5, or 1

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
		#binsums[i] = binsum_target

	return k-responses

# def two_stage_fac_error(weights, C_thiseye, C_othereye, X_thiseye, X_othereye):
# 	'''error function to be minimized over a group of observations to determine values of free parameters.

# 	model with facilitation, so w_m, w_d, a.'''
# 	# k is taken from Meese & Baker
# 	#k = 0.2 # proportional to stdev of late additive noise / SNR
# 	#responses, binsums = two_stage_fac_resp(weights, C_thiseye, C_othereye, X_thiseye, X_othereye)
# 	#return np.sum((k-responses)**2)

#error function to be minimized to obtain threshElev predictions.
def two_stage_fac_thresh(thresh_param, C_othereye, X_thiseye, X_othereye, fitted_params): #threshs will be minimized
	C_thiseye = thresh_param['C_thiseye'].value
	return two_stage_fac_resp(fitted_params, [C_thiseye], C_othereye, X_thiseye, X_othereye)

def predict_thresh(func, init_guess, C_other, X_this, X_other, fitted_params):
	'''A wrapper function that accepts a threshold-error minimizing function with arguments in a convenient order'''
	thresh_params = Parameters()
	thresh_params.add(name='C_thiseye', value=init_guess, vary=True)
	thresh_fit = minimize(func, thresh_params, args=(C_other, X_this, X_other, fitted_params))
	return thresh_fit.params['C_thiseye'].value
	#res = so.minimize(func, x0=init_guess, args=(C_other, X_this, X_other, *params))
	#print(res.x)
	#return res.x

def task_2st_fac_resp(weights, C_thiseye, C_othereye, X_thiseye, X_othereye, target_eye, presentation):
	'''Two-stage model with facilitation - model outputs.
	Models a whole subject/task (so both eyes, both presentation conditions)
	As a result, expects 9 parameters: 1 alpha and 8 weights, 2 per eye/pres = 2 x (2x2) = 8
	weights: # assume params is (a, wm_de_dicho, wd_de_dicho, wm_de_mono, wd_de_mono, wm_nde_dicho, wd_nde_dicho, wm_nde_mono, wd_nde_mono)
	mono- and dichoptic-suppression constants, a = facilitation parameter [same for all observations]
	C_thiseye, C_othereye: target contrasts in the two eyes, in percent
	X_thiseye, X_othereye: mask/surround contrasts in the two eyes, in percent
	target_eye: which eye viewed the target
	presentation: nMono or nDicho

	C's and X's must have the same length (#observations), also target_eye and presentation'''

	assert(len(C_thiseye)==len(C_othereye) & len(C_thiseye)==len(X_thiseye) & len(C_thiseye)==len(X_othereye))
	assert(len(C_thiseye)==len(target_eye) & len(C_thiseye)==len(presentation))

	a, wm_de_dicho, wd_de_dicho, wm_de_mono, wd_de_mono, wm_nde_dicho, wd_nde_dicho, wm_nde_mono, wd_nde_mono = weights
	p = 8 # stage 2 excitatory exponent # or 2.4
	q = 6.5 # stage 2 suppressive exponent # or 2
	Z = .0085 # stage 2 saturation constant # or 5, or 1

	responses = np.empty(len(C_thiseye))
	binsums = np.empty(len(C_thiseye))

	for i,(C_this, C_other, X_this, X_other, eye, pres) in enumerate(zip(C_thiseye, C_othereye, X_thiseye, X_othereye, target_eye, presentation)):
		# determine, based on eye and presentation, which weights to use.
		if eye=='De':
			if pres=='nDicho':
				w_xm = wm_de_dicho
				w_xd = wd_de_dicho
				w_xm_other = wm_nde_dicho
				w_xd_other = wd_nde_dicho
			elif pres=='nMono':
				w_xm = wm_de_mono
				w_xd = wd_de_mono
				w_xm_other = wm_nde_mono
				w_xd_other = wd_nde_mono
		elif eye=='Nde':
			if pres=='nDicho':
				w_xm = wm_nde_dicho
				w_xd = wd_nde_dicho
				w_xm_other = wm_de_dicho
				w_xd_other = wd_de_dicho
			elif pres=='nMono':
				w_xm = wm_nde_mono
				w_xd = wd_nde_mono
				w_xm_other = wm_de_mono
				w_xd_other = wd_de_mono
		#print(eye, pres)

		# calculate model responses using the appropriate weights
		stage1_this_t = stage1(C_this, C_other, X_this, X_other, w_xm, w_xd)
		stage1_other_t = stage1(C_other, C_this, X_other, X_this, w_xm_other, w_xd_other)
		stage1_this_m = stage1(X_this, X_other, C_this, C_other, w_xm, w_xd)
		stage1_other_m = stage1(X_other, X_this, C_other, C_this, w_xm_other, w_xd_other)

		binsum_target = stage1_this_t + stage1_other_t
		binsum_mask = stage1_this_m + stage1_other_m

		# model with facilitation, taken from Meese & Baker 2009
		resp_t = ((1 + a*binsum_mask)*(binsum_target**p))/(Z+binsum_target**q)
		responses[i] = resp_t
		binsums[i] = binsum_target

	return responses, binsums

def task_2st_fac_err(params, C_thiseye, C_othereye, X_thiseye, X_othereye, target_eye, presentation):
	'''model an entire task for one 'subject' (for the group data, combo of Task/Orientation/Population). Allows constraining
	a over both eyes/presentations, while letting w_m/w_d vary.'''

	# assume params is a, wm_de_dicho, wd_de_dicho, wm_de_mono, wd_de_mono, wm_nde_dicho, wd_nde_dicho, wm_nde_mono, wd_nde_mono
	k = 0.2 # proportional to stdev of late additive noise / SNR
	responses, binsums = task_2st_fac_resp(params, C_thiseye, C_othereye, X_thiseye, X_othereye, target_eye, presentation)
	#return ((k-responses)**2).sum()
	return ((k-responses)**2).sum()

def task_2st_fac_thresh(C_thiseye, C_othereye, X_thiseye, X_othereye, target_eye, presentation, a, wm_de_dicho, wd_de_dicho, wm_de_mono, wd_de_mono, wm_nde_dicho, wd_nde_dicho, wm_nde_mono, wd_nde_mono): #threshs will be minimized
	k = 0.2
	responses, binsums = task_2st_fac_resp(
		(a, wm_de_dicho, wd_de_dicho, wm_de_mono, wd_de_mono, wm_nde_dicho, wd_nde_dicho, wm_nde_mono, wd_nde_mono),
		C_thiseye, C_othereye, X_thiseye, X_othereye, target_eye, presentation)
	#print(responses)
	return ((k-responses)**2).sum()

def model_condition(g, err_func, thresh_func, params, ret='preds'):
	'''Model a condition. In this case, this function is to be applied to each group, where a group is a particular:
	- Task (OS/SS)
	- Mask Orientation (Iso/Cross)
	- Presentation (nMono/nDicho)
	- Eye (which viewed the target, De/Nde)
	- Population (Con/Amb)

	The values that are then modeled are RelMaskContrast (x) vs ThreshElev (y)

	if ret='weights', returns the weights (and any other fitted parameters) for this group.
	the default is to return the predicted thresholds (ret='preds')'''

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

	params_fit = minimize(err_func, params, args=contrasts)
	pfit = params_fit.params
	pfit.pretty_print()
	#params = fmin(err_func, np.zeros(n_free), args=contrasts) #fitted weights of free parameters of the model, as implemented by err_func
	#print(*zip(free_params,params))
	threshpred = [predict_thresh(thresh_func, a,[b],[c],[d],pfit) for a,b,c,d in zip(*contrasts)]

	if ret=='preds':
		g['ThreshPred'] = threshpred
		return g
	elif ret=='weights':
		return pd.Series(pfit, index=free_params)

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

	n_input_args = 6 # How many input parameters to the model (rest are free)

	free_params = inspect.getargspec(thresh_func).args[n_input_args:] # 0-5 are C_thiseye, C_othereye, X_thiseye, X_othereye, target_eye, presentation
	n_free = len(free_params)

	print(g.name)

	model_inputs = np.zeros((len(g),n_input_args))
	print(model_inputs.shape)

	sg_vars = ['Eye','Presentation']
	n_sgconds = np.prod([len(np.unique(g[col])) for col in ['Eye','Presentation']]) # how many unique conditions are there within this group?
	len_inputs = len(g) 
	len_inputs0 = len_inputs + n_sgconds # add the 0,0 observation in each group
	subgrouped = g.groupby(sg_vars)
	ind_start = 0
	ind_start0 = 0

	C_this = np.zeros(len_inputs)
	C_other = np.zeros(len_inputs)
	X_this = np.zeros(len_inputs)
	X_other = np.zeros(len_inputs)
	target_eye = np.empty(len_inputs,dtype=object)
	presentation_cond = np.empty(len_inputs,dtype=object)


	C_this0 = np.zeros(len_inputs0)
	C_other0 = np.zeros(len_inputs0)
	X_this0 = np.zeros(len_inputs0)
	X_other0 = np.zeros(len_inputs0)
	target_eye0 = np.empty(len_inputs0,dtype=object)
	presentation_cond0 = np.empty(len_inputs0,dtype=object)

	for (eye, pres), sg in subgrouped:
		n_obs = len(sg)
		print(eye, pres, n_obs)
		masks = sg.RelMaskContrast
		threshs = sg.ThreshElev
		assert(np.all(sg.Eye==sg.Eye.iloc[0])) # Make sure we only are looking at data for one eye
		Eye = sg.Eye.iloc[0]
		assert(np.all(sg.Presentation==sg.Presentation.iloc[0])) # again, one condition only
		Presentation = sg.Presentation.iloc[0]

		if Presentation=='nDicho':
			contrasts = (threshs.as_matrix(), np.zeros_like(threshs), np.zeros_like(masks), masks.as_matrix(), sg.Eye.as_matrix(), sg.Presentation.as_matrix())
		elif Presentation=='nMono':
			contrasts = (threshs.as_matrix(), np.zeros_like(threshs), masks.as_matrix(), np.zeros_like(masks), sg.Eye.as_matrix(), sg.Presentation.as_matrix())

		contrasts0 = tuple((np.insert(foo,0,0) for foo in contrasts))
		contrasts0[4][0] = contrasts0[4][1]
		contrasts0[5][0] = contrasts0[5][1]
		print(contrasts0)


		# # attempting to include 0,0 in the points to be fitted
		# C_this0 = [0]
		# C_this0.extend(contrasts[0])
		# #C_other[ind_start] = 0
		# C_other0 = [0]
		# C_other0.extend(contrasts[1])
		# #X_this[ind_start] = 0
		# X_this0 = [0]
		# X_this0.extend(contrasts[2])
		# #X_other[ind_start] = 0
		# X_other0 = [0]
		# X_other0.extend(contrasts[3])
		# #print(C_this0)
		# #target_eye[ind_start] = contrasts[4][0]
		# target_eye0 = [contrasts[4][0]]
		# target_eye0.extend(contrasts[4])
		# #presentation_cond[ind_start] = contrasts[5][0]
		# presentation0 = [contrasts[5][0]]
		# presentation0.extend(contrasts[5])
		# #ind_start = ind_start + 1
		# # end attempt
		idxs = list(range(ind_start, ind_start+n_obs))
		C_this[idxs] = contrasts[0]
		C_other[idxs] = contrasts[1]
		X_this[idxs] = contrasts[2]
		X_other[idxs] = contrasts[3]
		target_eye[idxs] = contrasts[4]
		presentation_cond[idxs] = contrasts[5]
		ind_start = ind_start+n_obs

		# inputs, but including the (0,0) point
		# result - this doesn't work either, since (0,0) is treated like any other data point, and so error is tolerated
		# would need to re-express this as a constraint
		idxs0 = list(range(ind_start0, ind_start0+n_obs+1))
		C_this0[idxs0] = contrasts0[0]
		C_other0[idxs0] = contrasts0[1]
		X_this0[idxs0] = contrasts0[2]
		X_other0[idxs0] = contrasts0[3]
		target_eye0[idxs0] = contrasts0[4]
		presentation_cond0[idxs0] = contrasts0[5]
		ind_start0 = ind_start0 + n_obs + 1

		#print(contrasts, contrasts[0].shape)

	inputs = (C_this, C_other, X_this, X_other, target_eye, presentation_cond)
	inputs_with0 = (C_this0, C_other0, X_this0, X_other0, target_eye0, presentation_cond0)
	print(inputs)
	print(inputs_with0)
	#print(pd.DataFrame([C_this, C_other, X_this, X_other, target_eye, presentation_cond]))
	opt_res = so.minimize(err_func, np.zeros(n_free), args=inputs, bounds=[(0, None) for p in free_params])
	#opt_res = so.minimize(err_func, np.zeros(n_free), args=inputs_with0, bounds=[(0, None) for p in free_params])
	print(opt_res['success'], opt_res['message'])
	if not opt_res['success']:
		error('optimization failed')
	params = opt_res.x
	print(*zip(free_params,params))

	if ret=='preds':
		threshpred = [predict_thresh(thresh_func, [a],[b],[c],[d],[e],[f],*params)[0] for a,b,c,d,e,f in zip(*inputs)]
		#threshpred = [predict_thresh(thresh_func, [1],[b],[c],[d],[e],[f],*params)[0] for a,b,c,d,e,f in zip(*inputs)]
		g['ThreshPred'] = threshpred
		return g
	elif ret=='weights':
	 	return pd.Series(params, index=free_params)
