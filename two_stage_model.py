# This file contains the functions that implement the two-stage model of suppression described in
# https://www.sciencedirect.com/science/article/pii/S0042698908002290 <--- this paper
# which I tried to fit to the threshold data previously
#
# For this project, I'm fitting it to the individual trial data
# So some of the code will be the same and some will be new

import numpy as np
import lmfit as lf

def stage1(cmpt_thiseye, cmpt_othereye, mask_thiseye, mask_othereye, m=1.3, S=1, w_xm=1, w_xd=1):
    """First (pre-binocular-summation) stage of the two-stage model"""
    return (cmpt_thiseye**m)/(S + cmpt_thiseye + cmpt_othereye + w_xm*mask_thiseye + w_xd*mask_othereye)

def two_stage_parameters():
    """
    Return the parameters of the two-stage model as a lmfit Parameters object.
    This allows their values, ranges, and whetehr they are free or fixed to be specified.
    """
    params = lf.Parameters()
    params.add('m', value=1.3, vary=False)
    params.add('S', value=1, vary=False)
    params.add('w_m', value=1, min=0.0, max=10.0, vary=True)
    params.add('w_d', value=1, min=0.0, max=10.0, vary=True)
    params.add('a', value=0, vary=True)
    params.add('k', value=0.2, vary=False)
    params.add('p', value=8, vary=False)
    params.add('q', value=6.5, vary=False)
    params.add('Z', value=.0085, vary=False)
    return params

def two_stage_response(params, C_thiseye, C_othereye, X_thiseye, X_othereye, n, c):
    """
    Two-stage model of contrast response with facilitation
    C_thiseye, C_othereye: target contrasts in the two eyes, in percent
    X_thiseye, X_othereye: mask/surround contrasts in the two eyes, in percent
    n: number of trials in this condition
    c: number of correct responses in this condition

    Inputs should be lists or pd.Series and should all be the same length.

    Output is predicted % correct - provided % correct (c/n), that is, the error - hence this is to be minimized
    """

    from scipy.stats import norm

    m = params['m'] # stage 1 excitatory constant
    S = params['S'] # stage 1 suppressive constant
    w_xm = params['w_m'] # want to fit this - monocular suppression weight
    w_xd = params['w_d'] # want to fit this - interocular suppression weight
    a = params['a'] # facilitation parameter
    k = params['k'] # incremental response needed to achieve specified performance (d')
    p = params['p'] # stage 2 excitatory exponent 
    q = params['q'] # stage 2 suppressive exponent
    Z = params['Z'] # stage 2 saturation constant

    responses = np.empty(len(C_thiseye))
    dprimes = np.empty(len(C_thiseye))
    predicted_pct_correct = np.empty(len(C_thiseye))
    errors = np.empty(len(C_thiseye))
    likelihoods = np.empty(len(C_thiseye))

    for i,(CDe, CNde, XDe, XNde, ntrials, ncorrect) in enumerate(zip(C_thiseye, C_othereye, X_thiseye, X_othereye, n, c)):
        stage1De_t = stage1(CDe, CNde, XDe, XNde, m, S, w_xm, w_xd)
        stage1Nde_t = stage1(CNde, CDe, XNde, XDe, m, S, w_xm, w_xd)
        stage1De_m = stage1(XDe, XNde, CDe, CNde, m, S, w_xm, w_xd)
        stage1Nde_m = stage1(XNde, XDe, CNde, CDe, m, S, w_xm, w_xd)

        binsum_target = stage1De_t + stage1Nde_t
        binsum_mask = stage1De_m + stage1Nde_m

        # model with facilitation, taken from Meese & Baker 2009
        resp_tm = ((1 + a*binsum_mask)*(binsum_target**p))/(Z+binsum_target**q)
        responses[i] = resp_tm

        # calculate some additional stuff that we could use
        dprimes[i] = resp_tm * k # not sure about this...
        predicted_pct_correct[i] = norm.cdf(dprimes[i])
        errors[i] = predicted_pct_correct[i]-(ncorrect/ntrials)
        likelihoods[i] = loglikelihood(ntrials, ncorrect, predicted_pct_correct[i])

    return likelihoods

def loglikelihood(n, c, predicted_pct_correct):
    """
    Calculate log-likelihood of the parameters that predicted predicted_pct_correct for a condition with n trials and c corrects (observed)
    """
    if c==n or c==0 or predicted_pct_correct==0 or predicted_pct_correct==1:
        return 0
    else:
        c_pred = np.round(predicted_pct_correct*n)
        ll_pred = (c_pred * np.log(predicted_pct_correct)) + ((n-c_pred)*(np.log(1-predicted_pct_correct)))
        ll_obs = (c*np.log(c/n)) + ((n-c)*np.log(1-(c/n)))
        return ll_pred-ll_obs # since lf.minimize() minimizes the sum of squares, return a residual