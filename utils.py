import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy.stats as st
import statsmodels.formula.api as sm
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

def test_baseline_diffs(g):
    print(g.iloc[0][['Task', 'Orientation', 'Presentation', 'Population']])
    ndes = np.unique(g[g.Eye=='Nde']['BaselineThresh'])
    des = np.unique(g[g.Eye=='De']['BaselineThresh'])
    print(ndes, len(ndes), '\n', des, len(des))
    ttres = st.ttest_ind(ndes, des)
    print(ttres)
    return ttres

def describe_baselines(g):
    N = len(np.unique(g['BaselineThresh']))
    baseline_mean = g['BaselineThresh'].mean()
    baseline_std = np.unique(g['BaselineThresh']).std()
    baseline_SEM = baseline_std/np.sqrt(N)
    d = {'N':N, 'mean':baseline_mean, 'std':baseline_std, 'SEM':baseline_SEM}
    return pd.Series(d)

def get_interocular_baseline_diff(g):
    if len(g) < 2:
        return g
    else:
        assert(len(g)==2)
        nde_mean = g[g.Eye=='Nde']['mean'].iloc[0]
        de_mean = g[g.Eye=='De']['mean'].iloc[0]
        g['BaselineDiff'] = nde_mean - de_mean
        return g

def make_baseline_df_to_plot(df):
    return df.groupby('Population').apply(get_interocular_baseline_diff)

def linear_fit(df, x, y):
    result = sm.ols(formula=f"{y} ~ {x}", data=df).fit()
    return result

def linear_fit_params(df, x, y):
    result = linear_fit(df, x, y)
    ret = result.params
    ret.index = ret.index.str.replace(x, 'slope').str.replace('Intercept','y_int')
    fit_df = ret.append(pd.Series({'rsquared':result.rsquared}))
    return fit_df

def linear_fit_predictions(df, x, y):
    result = linear_fit(df, x, y)
    preds = pd.Series(result.predict(), index=df[x], name='ThreshPred')
    return preds

def remove_outliers_iqr(g):
    "Remove values failing the 1.5 * IQR rule"
    q1 = g['rsquared'].quantile(.25)
    q3 = g['rsquared'].quantile(.75)
    iqr = q3 - q1
    mask = g['rsquared'].between(q1-1.5*iqr, q3+1.5*iqr, inclusive=True)
    return g.loc[mask]

def remove_outliers_halfvar(g):
    "Remove values failing the 50% variance explained (R^2 > .5) rule"
    mask = g['rsquared'].between(.5, 1, inclusive=True)
    return g.loc[mask]

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

def calc_rs(df, permute=False):
    if permute:
        gaba = np.random.permutation(df['GABA'])
        measure = np.random.permutation(df['value'])
        corr = st.spearmanr(gaba, measure).correlation
    else:
        corr = st.spearmanr(df['GABA'], df['value']).correlation
    if np.isnan(corr):
        corr = 0
    return pd.Series({"correlation": corr})

def rs_diff(df):
    c = df
    amb_de = c.loc['Amblyope','De']['correlation']
    amb_nde = c.loc['Amblyope', 'Nde']['correlation']
    amb_diff = amb_nde-amb_de
    con_de = c.loc['Control','De']['correlation']
    con_nde = c.loc['Control', 'Nde']['correlation']
    con_diff = con_nde-con_de
    pop_diff = amb_diff - con_diff
    return [amb_diff, con_diff, pop_diff]

def compare_rs(df, n_boot=2, verbose=False, resample=False):
    print(df.name)
    if verbose:
        print(f"given df: (head)",
              df[['Population','Eye','Subject','GABA','value']].head(),
              sep="\n")
    corrs = df.groupby(['Population', 'Eye']).apply(calc_rs)
    print(corrs)
    a_real, c_real, p_real = rs_diff(corrs)
    print(f"Real (observed) r_s differences:\nA\tC\tP\n{a_real:.3}\t{c_real:.3}\t{p_real:.3}")
    rs_permute = np.empty((3, n_boot), dtype=np.float32)
    corrs_permute = np.empty((4, n_boot), dtype=np.float32)
    for i in range(n_boot):
        # sample with replacement
        if resample:
            samples = df.groupby(['Population', 'Eye'], as_index=False).apply(
                lambda x: x.sample(n=len(x), replace=True)).reset_index()
        else:
            samples = df
        permute_corrs = samples.groupby(['Population', 'Eye']).apply(calc_rs, permute=True)
        amb_diff, con_diff, pop_diff = rs_diff(permute_corrs)
        if verbose:
            print(f"resampled to: (iteration {i})",
              samples[['Population','Eye','Subject','GABA','value']].head(), sep="\n")
            print(corrs)
            print(f"Amb Nde-De: {amb_diff}")
            print(f"Con Nde-De: {con_diff}")
            print(f"Amb - Con: {pop_diff}")
        rs_permute[:, i] = [amb_diff, con_diff, pop_diff]
        amb_de = permute_corrs.loc['Amblyope','De']['correlation']
        amb_nde = permute_corrs.loc['Amblyope', 'Nde']['correlation']
        con_de = permute_corrs.loc['Control','De']['correlation']
        con_nde = permute_corrs.loc['Control', 'Nde']['correlation']
        corrs_permute[:, i] = [amb_de, amb_nde, con_de, con_nde]     
    comps = ['A', 'C', 'P']
    corrs_in_order = ['AD', 'AN', 'CD', 'CN']
    print("Percentiles for permuted r_s differences:")
    for i in range(3):
        p = np.percentile(rs_permute[i, :], np.array([0, 0.5, 1, 1.5, 2, 2.5, 5, 25, 50, 75, 95, 97.5, 100]))
        print(comps[i], p)
    for i in range(4):
        p = np.percentile(corrs_permute[i, :], np.array([0, 0.5, 1, 1.5, 2, 2.5, 5, 25, 50, 75, 95, 97.5, 100]))
        print(corrs_in_order[i], p)
    rs_df = pd.DataFrame({'amb_rdiff':rs_permute[0, :],
                          'con_rdiff':rs_permute[1, :],
                          'pop_rdiff':rs_permute[2, :]})
    return rs_df

def calculate_orientation_selective_suppression(df, **kwargs):
    #print(df[['Orientation', 'value']])
    if len(df.Orientation.unique())==2:
        v1 = df[df.Orientation=='Iso']['value'].iloc[0]
        v2 = df[df.Orientation=='Cross']['value'].iloc[0]
        iso_cross_oss_ratio = v1/v2
        #iso_cross_mean = np.mean([v1, v2])
    else:
        iso_cross_oss_ratio = np.nan
        #iso_cross_mean = np.nan
    print(f"Iso/Cross ratio: {iso_cross_oss_ratio}")
    return pd.Series(iso_cross_oss_ratio, ['value'])