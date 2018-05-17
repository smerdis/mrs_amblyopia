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
    df = pd.read_table(pp_fn)
    df['logThreshElev'] = np.log10(df['ThreshElev'])
    return df

def load_gaba(gaba_fn):
    gdf = pd.read_table(gaba_fn)
    return gdf[gdf.Presentation=='occ_binoc'] # this is the gaba measure we want to use

def load_fmri(fmri_fn):
    return pd.read_table(fmri_fn)

def load_individual_data(data_file, columns):
    """Function that loads individual psychophysics data stored in .mat files.
    
    data_file: path to .mat file
    columns: list of column names. must match number of columns in the .mat file"""

    subject_code = data_file.split('_filteredData')[0].split('/')[-1] # get subject name from filename
    mat = loadmat(data_file)
    mdata = mat['data']
    mdtype = mdata.dtype
    subj_df = pd.DataFrame(data=mdata, columns=columns)
    
    # Convert the following columns to ints if they exist in this data frame
    cols_to_intify = ["StaircaseNumber", "Eye", "Orientation", "Presentation", "TrialNumberStaircase",
                "ResponseAccuracy", "ProbeInterval", "ProbeLocation", "FileNumber"]
    # Furthermore, make these ones categorical too
    cols_to_categorize = ["Eye", "Presentation", "ResponseAccuracy"]
    for col in cols_to_intify:
        if col in subj_df.columns:
            subj_df[col] = subj_df[col].astype(int)
            if col in cols_to_categorize:
                subj_df[col] = pd.Categorical(subj_df[col])
                
    # Finally, round these ones to avoid floating point errors
    cols_to_round = ["ProbeContrastRecommended", "ProbeContrastUsed"]
    for col in cols_to_round:
        if col in cols_to_intify or col in cols_to_categorize:
            raise Error(f"Column {col} is listed as both integer and floating point!")
        subj_df[col] = np.round(subj_df[col], 2)

    # Make Subject column and put it first
    subj_df['Subject'] = subject_code
    subj_df = subj_df[['Subject', *columns]]

    # Omit the baseline conditions (orientations -1 and -2) when returning
    return subj_df[subj_df.Orientation>=0]

def load_individual_os_data(data_file):
    """
    Load data for Orientation Suppression task which has 11 columns.

    Staircase number for a given test block (= file number)
    Eye (1=weaker eye, 2= fellow eye)
    Mask Orientation (0=parallel, 90=orthogonal)
    Binocular condition (1= monocular, 2=dichoptic)
    Mask Contrast (michelson)
    Trial number for this staircase
    Probe contrast recommended by staircase algorithm
    Response Accuracy (1=correct, 0=incorrect)
    Probe Contrast used (I don't remember it ever being different from #7 and was really just a sanity check)
    Interval that probe was presented in (1 or 2)
    File number (=test block)
    """
    columns_os = ["StaircaseNumber", "Eye", "Orientation", "Presentation", "MaskContrast", "TrialNumberStaircase",
              "ProbeContrastRecommended", "ResponseAccuracy", "ProbeContrastUsed", "ProbeInterval", "FileNumber"]
    return load_individual_data(data_file, columns_os)

def load_individual_ss_data(data_file):
    """
    Load data for Surround Suppression task which has 12 columns.

    Staircase number for a given test block (= file number)
    Eye (1=weaker eye, 2= fellow eye)
    Mask Orientation (0=parallel, 90=orthogonal)
    Binocular Condition (1= monocular, 2=dichoptic)
    Trial number for this staircase
    Contrast increment recommended by staircase algorithm
    Response Accuracy (1=correct, 0=incorrect)
    Mask Contrast (michelson)
    Probe location (1-4, let me know if you need to know which number represents which quadrant)
    Response (1-4)
    Probe contrast increment used (I don't remember it ever being different from #6 and was really just a sanity check)
    File number (=test block)
    """
    columns_ss = ["StaircaseNumber", "Eye", "Orientation", "Presentation", "TrialNumberStaircase",
              "ProbeContrastRecommended", "ResponseAccuracy", "MaskContrast", "ProbeLocation",
                  "Response", "ProbeContrastUsed",  "FileNumber"]
    return load_individual_data(data_file, columns_ss)

def load_all_data(globstr, loader_func):
    """
    Load all the data from the files specified by globstr using the specified loading function,
    concatenate them and return.
    """
    data_files = glob.glob(globstr)
    df = pd.DataFrame()
    for data_file in data_files:
        df = pd.concat([df, loader_func(data_file)])
    return df

def summarize_conditions(df, gvars):
    """
    Condense the data frame from individual trials to statistics of each condition that will be used for modeling.
    """
    grouped = df.groupby(gvars)

    # For each combination of the above, how many trials (n), how many correct (c), and what's the percentage?
    condensed_df = grouped['ResponseAccuracy'].agg([len, np.count_nonzero]).rename(columns={'len':'n', 'count_nonzero':'c'})
    condensed_df['pct_correct'] = condensed_df['c']/condensed_df['n']

    return grouped, condensed_df

def predict_thresh(func, init_guess, C_other, X_this, X_other, fitted_params):
    '''A wrapper function that accepts a threshold-error minimizing function with arguments in a convenient order'''
    #print(init_guess, np.any(np.isnan(fitted_params)))
    thresh_params = lf.Parameters()
    thresh_params.add(name='C_thiseye', value=init_guess, min=0.0, vary=True)
    thresh_fit = lf.minimize(func, thresh_params, args=(C_other, X_this, X_other, fitted_params))
    return thresh_fit.params['C_thiseye'].value

def model_trials(g, err_func, params):
    """
    Model the trials within a condition. This function is to be applied to each group of observations, where a group is a particular:
    - Subject (individual)
    - Task (OS/SS)
    - Mask Orientation (Iso/Cross)
    - Presentation (nMono/nDicho)
    - Eye (which viewed the target, De/Nde)

    For this group of observations, the params (an lmfit Parameters object) are fitted by minimizing err_func.
    """

    print(g.name, len(g))

    assert(np.all(g.Eye==g.Eye.iloc[0])) # Make sure we only are looking at data for one eye
    Eye = g.Eye.iloc[0]
    assert(np.all(g.Presentation==g.Presentation.iloc[0])) # again, one condition only
    Presentation = g.Presentation.iloc[0]

    t = g.ProbeContrastUsed
    m = g.MaskContrast
    n = g.n
    c = g.c
    zs = np.zeros_like(t)

    #print(t, m, n, c)
    if Presentation==2: # dichoptic - target and mask to different eyes
        contrasts = (t, zs, zs, m, n, c)
    elif Presentation==1: # monocular
        contrasts = (t, zs, m, zs, n, c)

    params_fit = lf.minimize(err_func, params, args=contrasts)
    pfit = params_fit.params
    return pd.Series(pfit.valuesdict(), index=params.keys())

def model_threshold(g, err_func, thresh_func, params, ret='preds', predtype='linear'):
    '''Model a condition. In this case, this function is to be applied to each group, where a group is a particular:
    - Task (OS/SS)
    - Eye (which viewed the target, De/Nde)
    - Population (Con/Amb)

    (both Presentations, nMono and nDicho, will be in the group, as will both Orientations (Iso/Cross))

    The values that are then modeled are RelMaskContrast (x) vs ThreshElev (y)

    if ret='weights', returns the weights (and any other fitted parameters) for this group.
    the default is to return the predicted thresholds (ret='preds')'''

    print(g.name)

    assert(np.all(g.Eye==g.Eye.iloc[0])) # Make sure we only are looking at data for one eye
    Eye = g.Eye.iloc[0]

    assert(np.all(g.BaselineThresh==g.BaselineThresh.iloc[0])) # has to be true since we will compute un-normalized threshelev (=C%)
    BaselineThresh = g.BaselineThresh.iloc[0]

    conditions = g.groupby(['Presentation'])
    # set up empty ndarrays to hold the arguments that will be passed to the model
    thresh_this, thresh_other, mask_this, mask_other = np.empty(0), np.empty(0), np.empty(0), np.empty(0)
    # build the input arrays
    for Presentation, pres_g in conditions:
        assert(np.all(pres_g.Presentation==pres_g.Presentation.iloc[0])) # again, one condition only

        # collect the masks and thresholds, put them in the right order (which depends on Presentation) to pass to the model
        masks = pres_g.RelMaskContrast
        threshs = pres_g.ThreshElev

        if Presentation=='nDicho':
            tt, to, mt, mo = (threshs.as_matrix(), np.zeros_like(threshs), np.zeros_like(masks), masks.as_matrix())
        elif Presentation=='nMono':
            tt, to, mt, mo = (threshs.as_matrix(), np.zeros_like(threshs), masks.as_matrix(), np.zeros_like(masks))
        thresh_this = np.hstack((thresh_this, tt))
        thresh_other = np.hstack((thresh_other, to))
        mask_this = np.hstack((mask_this, mt))
        mask_other = np.hstack((mask_other, mo))

    contrasts = (thresh_this, thresh_other, mask_this, mask_other)
    #print(contrasts)
    params_fit = lf.minimize(err_func, params, args=contrasts)
    pfit = params_fit.params
    pfit.pretty_print()
    threshpred = [predict_thresh(thresh_func, a,[b],[c],[d],pfit) for a,b,c,d in zip(*contrasts)]
    if ret=='preds':
        g['ThreshPred'] = threshpred
        return g
    elif ret=='weights':
        retvars = pd.Series(pfit.valuesdict(), index=params.keys())
        try:
            if ThreshPredBinCenter: # This is set if we use the linear model. TODO clean up
                retvars['ThreshPredCritical'] = ThreshPredBinCenter
                retvars['ThreshPredCriticalUnnorm'] = ThreshPredBinCenterUnnorm
        except NameError:
            pass
        return retvars