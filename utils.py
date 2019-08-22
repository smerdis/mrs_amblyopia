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