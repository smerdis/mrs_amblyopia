import numpy as np
import lmfit as lf

### Linear model
def parameters():
    lm_params = lf.Parameters()
    lm_params.add('y_int', value=1, vary=True)
    lm_params.add('slope', value=1, vary=True)
    return lm_params

def err(params, C_thiseye, C_othereye, X_thiseye, X_othereye):
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
def thresh(thresh_param, C_othereye, X_thiseye, X_othereye, fitted_params): #threshs will be minimized
    C_thiseye = thresh_param['C_thiseye'].value
    return err(fitted_params, [C_thiseye], C_othereye, X_thiseye, X_othereye)