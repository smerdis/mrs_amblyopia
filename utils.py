import numpy as np
import pandas as pd
from scipy.io import loadmat
import lmfit as lf
import glob

# functions to convert between db and Michelson contrast
def pct_to_db(pct):
    return 20 * np.log10(pct)

def db_to_pct(db):
    return 10**(db/20)

## Functions to read input
def load_psychophys(pp_fn):
    df = pd.read_csv(pp_fn, sep='\t')
    df['logThreshElev'] = np.log10(df['ThreshElev'])
    df['logRelMaskContrast'] = np.log10(df['RelMaskContrast'])
    return df

def load_gaba(gaba_fn, pres_cond='occ_binoc'):
    gdf = pd.read_csv(gaba_fn, sep='\t')
    return gdf[gdf.Presentation==pres_cond] # this is the gaba measure we want to use

def load_fmri(fmri_fn):
    return pd.read_csv(fmri_fn, sep='\t')

def predict_thresh(func, init_guess, C_other, X_this, X_other, fitted_params):
    """
    A wrapper function that accepts a threshold-error minimizing function with arguments in a convenient order
    """
    #print(init_guess, np.any(np.isnan(fitted_params)))
    thresh_params = lf.Parameters()
    thresh_params.add(name='C_thiseye', value=init_guess, min=0.0, vary=True)
    thresh_fit = lf.minimize(func, thresh_params, args=(C_other, X_this, X_other, fitted_params))
    return thresh_fit.params['C_thiseye'].value

def test_all_bins(test_group, gvars_pair, bin_col, y_col, test_func, **kwargs):
    """
    Accepts data grouped by (Task, Orientation, Presentation, Population)
    then feeds each bin within this to test_one_bin().
    """
    ttg_pairbin = test_group.groupby(gvars_pair)
    print(f"There are {len(ttg_pairbin)} bins in this condition.")
    g_bin = ttg_pairbin.apply(test_one_bin, y_col, test_func, **kwargs).reset_index()
    minp_bin = g_bin[bin_col].iat[g_bin.pvalue.idxmin()]
    print(f"{bin_col} {minp_bin} has lowest p-value.\n")
    return pd.Series(minp_bin, ['BinNumberToPred'])

def test_one_bin(ttg, y_col, test_func, **kwargs):
    """
    Accepts data grouped by (Task, Orientation, Presentation, Population, BinNumber)
    and does the specified t-test (should be like ttest_ind or ttest_rel from scipy.stats)
    on the ThreshElev values in the bin.
    Specifically, tests where NDE and DE are most different
    """
    nde = ttg[ttg['Eye']=='Nde'][y_col].values
    de = ttg[ttg['Eye']=='De'][y_col].values
    # print(f"{ttg.name}",
    #     f"{nde} <= NDE, n={len(nde)}",
    #     f"{de} <= DE, n={len(de)}", sep='\n')
    if (len(nde) >0 and len(de) >0): # we have observations for both eyes
        if (np.mean(nde) > 1) or (np.mean(de) > 1): # if either eye averages being in suppression
            tt_res = test_func(nde, de, **kwargs)
            if tt_res:
                print(f"{len(de)} DE obs, {len(nde)} NDE obs\np-value: {tt_res[1]:.10f}")
                return pd.Series(tt_res[1], ['pvalue'])
        else:
            print('Not in suppression, mean of both eyes is < 1.')
    else:
        print('A group with no obs, skipping')


def add_pred_col(g, bin_col, y_col):
    '''
    Add a column with the numeric value we want the model to be evaluated/generate predictions at
    '''
    assert(np.all(g.BinNumberToPred==g.BinNumberToPred.iat[0]))
    RelMCToPredGroup = g[g[bin_col]==g.BinNumberToPred.iat[0]]
    print(RelMCToPredGroup.head())
    #print(RelMCToPredGroup.head()[['Subject','Eye',bin_col,'BinNumberToPred',y_col]])
    RelMCToPred = RelMCToPredGroup.BinNumberToPred.iat[0].astype(int)
    #ObservedThreshElevCriticalBin = RelMCToPredGroup.ThreshElev
    #for gv2, g2 in g.groupby(['Subject', 'Eye']):
    #    print(gv2, g2[['BinNumber','BinNumberToPred']])
    #    assert(np.any(g2['BinNumber']==g2['BinNumberToPred']))
    #print(ObservedThreshElevCriticalBin, len(ObservedThreshElevCriticalBin))
    #print(RelMCToPred, len(g))
    g['RelMCToPred'] = RelMCToPred
    return g

def find_pct_to_predict(df, gvars, bin_col, y_col, **kwargs):
    '''
    A function that wraps this operation:
    Take a data frame and a set of grouping variables,
    Group the data frame in this way and perform a test on bins within the data
    The binning is provided by the BinNumber column which must be present (and was in the original data)
    '''
    gvars_pair = gvars + [bin_col]
    test_groups = df.groupby(gvars)
    binpred = test_groups.apply(test_all_bins, gvars_pair, bin_col, y_col, **kwargs).reset_index()
    #print("binpred cols: ", binpred.columns)
    print("Any NaNs in BinNumberToPred?", np.any(np.isnan(binpred.BinNumberToPred)))
    # After this line we have a column called BinNumberToPred at the end of the df
    df = pd.merge(df, binpred, on=gvars)

    #print(df.columns)
    #print(df.head())

    # Now group the data separately by Eye, since NDE/DE have different RelMaskContrasts
    # at the center of their respective bins-at-which-to-predict
    condition_groups = df.groupby(gvars + ['Eye'])
    # make sure all conditions have the same bin number to predict within them
    assert(np.all(condition_groups.apply(
        lambda g: np.all(g.BinNumberToPred==g.BinNumberToPred.iat[0])
    ).reset_index()))
    # Add a column with the actual numerical value at the center of the bin for each Eye
    df_to_model = condition_groups.apply(add_pred_col, bin_col, y_col)
    return df_to_model

# This function has been superseded by a simpler implementation using statsmodels
# However, it would still be useful to fit e.g. the two-stage model to the thresholds
# So I'm preserving it in this commit, though it might be deleted later as cruft.

# def model_threshold(g, err_func, thresh_func, params, ret='preds'):
#     """
#     Model a condition. This function is to be applied to each group, where a group is a particular:
#     - Task (OS/SS)
#     - Eye (which viewed the target, De/Nde)
#     - Population (Con/Amb)
#     - [other grouping variables, possibly]

#     The values that are then modeled are RelMaskContrast (x) vs ThreshElev (y)

#     if ret='weights', returns the weights (and any other fitted parameters) for this group.
#     the default is to return the predicted thresholds (ret='preds')
#     """

#     print(g.name)

#     assert(np.all(g.Eye==g.Eye.iloc[0])) # Make sure we only are looking at data for one eye
#     Eye = g.Eye.iloc[0]

#     assert(np.all(g.BaselineThresh==g.BaselineThresh.iloc[0])) # has to be true since we will compute un-normalized threshelev (=C%)
#     BaselineThresh = g.BaselineThresh.iloc[0]

#     # The following code allows both presentation conditions to be modeled simultaneously
#     # if they are both included in one group (g)
#     # If not, i.e. Presentation is among the grouping variables, it still works as usual
#     conditions = g.groupby(['Presentation'])
#     # set up empty ndarrays to hold the arguments that will be passed to the model
#     thresh_this, thresh_other, mask_this, mask_other = np.empty(0), np.empty(0), np.empty(0), np.empty(0)
#     # build the input arrays
#     for Presentation, pres_g in conditions:
#         assert(np.all(pres_g.Presentation==pres_g.Presentation.iloc[0])) # again, one condition only

#         # collect the masks and thresholds, put them in the right order (which depends on Presentation) to pass to the model
#         masks = pres_g.RelMaskContrast
#         threshs = pres_g.ThreshElev
#         if Presentation=='nDicho':
#             tt, to, mt, mo = (threshs.as_matrix(), np.zeros_like(threshs), np.zeros_like(masks), masks.as_matrix())
#         elif Presentation=='nMono':
#             tt, to, mt, mo = (threshs.as_matrix(), np.zeros_like(threshs), masks.as_matrix(), np.zeros_like(masks))
#         thresh_this = np.hstack((thresh_this, tt))
#         thresh_other = np.hstack((thresh_other, to))
#         mask_this = np.hstack((mask_this, mt))
#         mask_other = np.hstack((mask_other, mo))

#     contrasts = (thresh_this, thresh_other, mask_this, mask_other)
#     #print(contrasts)
#     params_fit = lf.minimize(err_func, params, args=contrasts)
#     pfit = params_fit.params
#     pfit.pretty_print()
#     threshpred = [predict_thresh(thresh_func, a,[b],[c],[d],pfit) for a,b,c,d in zip(*contrasts)]
#     if ret=='preds':
#         g['ThreshPred'] = threshpred
#         return g
#     elif ret=='weights':
#         retvars = pd.Series(pfit.valuesdict(), index=params.keys())
#         try:
#             if 'RelMCToPred' in g.columns: # We are given a RelMaskContrast we want to evaluate our model at
#                 retvars['ThreshPredCritical'] = ThreshPredBinCenter
#                 retvars['ThreshPredCriticalUnnorm'] = ThreshPredBinCenterUnnorm
#         except NameError:
#             pass
#         return retvars
