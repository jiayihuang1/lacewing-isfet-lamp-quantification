import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize, least_squares
from scipy.interpolate import interp1d
import warnings

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

# ============================================================================
# SIGMOID FUNCTIONS
# ============================================================================

def sigmoid_5p(x, Fm, Fb, Sc, Cs, As):
    """5-Parameter Logistic (5PL)
    y = Fm / (1 + exp(-Sc * (x - Cs)))^As + Fb
    Parameters: Fm (max), Fb (min), Sc (slope), Cs (center), As (asymmetry)
    """
    return Fm / (1.0 + np.exp(-Sc * (x - Cs)))**As + Fb

# ============================================================================
# DERIVATIONS
# ============================================================================

def sigmoid_5p_first_derivative(x, Fm, Fb, Sc, Cs, As):
    """First derivative of 5PL wrt x
    dy/dx = Fm * As * Sc * exp(-Sc*(x - Cs)) / (1 + exp(-Sc*(x - Cs)))^(As+1)
    """
    exp_term = np.exp(-Sc * (x - Cs))
    denominator = (1.0 + exp_term)**(As + 1)
    return Fm * As * Sc * exp_term / denominator

def sigmoid_5p_second_derivative(x, Fm, Fb, Sc, Cs, As):
    """Second derivative of 5PL wrt x
    d²y/dx² = Fm * As * Sc² * exp(-Sc*(x - Cs)) * (As*exp(-Sc*(x - Cs)) - 1) / (1 + exp(-Sc*(x - Cs)))^(As+2)
    """
    exp_term = np.exp(-Sc * (x - Cs))
    numerator = Fm * As * Sc**2 * exp_term * (As * exp_term - 1)
    denominator = (1.0 + exp_term)**(As + 2)
    return numerator / denominator

def calculate_first_derivative(x, y, method='gradient', window_length=11, polyorder=2):
    """
    Calculate first derivative of original curve
    
    Parameters
    ----------
    x : array
        X values (time)
    y : array
        Y values (original curve)
    method : str
        'gradient' - numpy gradient (simple)
        'savgol' - Savitzky-Golay (smoothed)
        'diff' - finite difference
    window_length : int
        Window length for Savitzky-Golay
    polyorder : int
        Polynomial order for Savitzky-Golay
    
    Returns
    -------
    dy_dx : array
        First derivative
    """
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if method == 'gradient':
        # Simple numpy gradient
        dy_dx = np.gradient(y, x)
    
    elif method == 'savgol':
        # Savitzky-Golay filter (smooth + derivative)
        dy_dx = savgol_filter(y, window_length=window_length, polyorder=polyorder, deriv=1)
        dy_dx = dy_dx / np.gradient(x)  # Normalize by dx
    
    elif method == 'diff':
        # Finite difference
        dy_dx = np.diff(y) / np.diff(x)
        dy_dx = np.append(dy_dx, dy_dx[-1])  # Pad last value
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return dy_dx


def calculate_second_derivative(x, y, method='gradient', window_length=11, polyorder=2):
    """
    Calculate second derivative of original curve
    
    Parameters
    ----------
    x : array
        X values (time)
    y : array
        Y values (original curve)
    method : str
        'gradient' - numpy gradient (simple)
        'savgol' - Savitzky-Golay (smoothed)
        'diff' - finite difference
    
    Returns
    -------
    d2y_dx2 : array
        Second derivative
    """
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if method == 'gradient':
        # First derivative
        dy_dx = np.gradient(y, x)
        # Second derivative
        d2y_dx2 = np.gradient(dy_dx, x)
    
    elif method == 'savgol':
        # Savitzky-Golay filter (second derivative)
        d2y_dx2 = savgol_filter(y, window_length=window_length, polyorder=polyorder, deriv=2)
        d2y_dx2 = d2y_dx2 / (np.gradient(x)**2)  # Normalize by dx²
    
    elif method == 'diff':
        # Finite difference
        dy_dx = np.diff(y) / np.diff(x)
        d2y_dx2 = np.diff(dy_dx) / np.diff(x[:-1])
        d2y_dx2 = np.append(d2y_dx2, d2y_dx2[-1])
        d2y_dx2 = np.append(d2y_dx2, d2y_dx2[-1])
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return d2y_dx2

# ============================================================================
# FEATURE EXTRACTIONS
# ============================================================================

def extract_kinetic_parameters(x, y, params, threshold=0.1):
    """
    Extract kinetic parameters from fitted 5PL curve
    Including all features from Table S2 + additional features
    
    Parameters
    ----------
    x : array
        Time values
    y : array
        Original response values (fitted curve)
    params : tuple
        (Fm, Fb, Sc, Cs, As) - fitted 5PL parameters
    threshold : float
        If < 1: fraction of max slope (0.1 = 10% of peak)
        If >= 1: absolute threshold value
    
    Returns
    -------
    dict with all kinetic parameters from Table S2 + new features
    """
    
    Fm, Fb, Sc, Cs, As = params
    
    # ========================================================================
    # 1. CALCULATE FIRST AND SECOND DERIVATIVES
    # ========================================================================
    
    dy_dx_vals = sigmoid_5p_first_derivative(x, Fm, Fb, Sc, Cs, As)
    d2y_dx2_vals = sigmoid_5p_second_derivative(x, Fm, Fb, Sc, Cs, As)
    
    # ========================================================================
    # 2. FIND xms (maximum slope) and dy_xms (maximum slope value)
    # ========================================================================
    
    max_slope_idx = np.argmax(dy_dx_vals)
    xms = x[max_slope_idx]
    dy_xms = dy_dx_vals[max_slope_idx]
    
    # ========================================================================
    # 3. FIND xs, xe (TH-crossing points with INTERPOLATION)
    # ========================================================================
    
    # Set threshold as fraction of peak if < 1
    if threshold < 1:
        TH = threshold * dy_xms
    else:
        TH = threshold
    
    # Find where 1st derivative crosses threshold
    above_threshold = dy_dx_vals > TH
    rising_indices = np.where(above_threshold)[0]
    
    # xs: first crossing point with interpolation
    if len(rising_indices) > 0:
        xs_idx = rising_indices[0]
        if xs_idx > 0:
            x1, x2 = x[xs_idx-1], x[xs_idx]
            y1, y2 = dy_dx_vals[xs_idx-1], dy_dx_vals[xs_idx]
            xs = x1 + (TH - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x[xs_idx]
        else:
            xs = x[xs_idx]
    else:
        xs = np.nan
    
    # xe: last crossing point with interpolation
    if len(rising_indices) > 0:
        xe_idx = rising_indices[-1]
        if xe_idx < len(x) - 1:
            x1, x2 = x[xe_idx], x[xe_idx+1]
            y1, y2 = dy_dx_vals[xe_idx], dy_dx_vals[xe_idx+1]
            xe = x1 + (TH - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x[xe_idx]
        else:
            xe = x[xe_idx]
    else:
        xe = np.nan
    
    # ========================================================================
    # 4. FIND xp1, xp2 (positive and negative peaks of 2nd derivative)
    # ========================================================================
    
    peaks_pos, _ = find_peaks(d2y_dx2_vals)
    if len(peaks_pos) > 0:
        xp1_idx = peaks_pos[np.argmax(d2y_dx2_vals[peaks_pos])]
        xp1 = x[xp1_idx]
        dy_xp1 = dy_dx_vals[xp1_idx]
        d2y_xp1 = d2y_dx2_vals[xp1_idx]
    else:
        xp1 = np.nan
        dy_xp1 = np.nan
        d2y_xp1 = np.nan
    
    peaks_neg, _ = find_peaks(-d2y_dx2_vals)
    if len(peaks_neg) > 0:
        xp2_idx = peaks_neg[np.argmin(d2y_dx2_vals[peaks_neg])]
        xp2 = x[xp2_idx]
        dy_xp2 = dy_dx_vals[xp2_idx]
        d2y_xp2 = d2y_dx2_vals[xp2_idx]
    else:
        xp2 = np.nan
        dy_xp2 = np.nan
        d2y_xp2 = np.nan
    
    # ========================================================================
    # 5. COLLECT Y-VALUES AT CRITICAL POINTS
    # ========================================================================
    
    y_xms = sigmoid_5p(xms, *params)
    y_xs = sigmoid_5p(xs, *params) if not np.isnan(xs) else np.nan
    y_xe = sigmoid_5p(xe, *params) if not np.isnan(xe) else np.nan
    y_xp1 = sigmoid_5p(xp1, *params) if not np.isnan(xp1) else np.nan
    y_xp2 = sigmoid_5p(xp2, *params) if not np.isnan(xp2) else np.nan
    
    # ========================================================================
    # 6. TABLE S2 FEATURES: THRESHOLD DISTANCE
    # ========================================================================
    
    threshold_distance = xe - xs if (not np.isnan(xs) and not np.isnan(xe)) else np.nan
    
    # ========================================================================
    # 7. TABLE S2 FEATURES: FIRST-HALF DISTANCE
    # ========================================================================
    
    first_half_distance = xms - xs if not np.isnan(xs) else np.nan
    
    # ========================================================================
    # 8. TABLE S2 FEATURES: SECOND-HALF DISTANCE
    # ========================================================================
    
    second_half_distance = xe - xms if not np.isnan(xe) else np.nan
    
    # ========================================================================
    # 9. TABLE S2 FEATURES: DISTANCE ASYMMETRICAL INDEX
    # ========================================================================
    
    if not np.isnan(first_half_distance) and first_half_distance != 0:
        distance_asymmetry_index = second_half_distance / first_half_distance
    else:
        distance_asymmetry_index = np.nan
    
    # ========================================================================
    # 10. TABLE S2 FEATURES: PEAK-SHIFTING DISTANCE
    # ========================================================================
    
    peak_shifting_distance = xp2 - xp1 if (not np.isnan(xp1) and not np.isnan(xp2)) else np.nan
    
    # ========================================================================
    # 11. TABLE S2 FEATURES: AREA UNDER CURVE (A1, A2)
    # ========================================================================
    
    if not np.isnan(xs) and not np.isnan(xms):
        xs_idx = np.argmin(np.abs(x - xs))
        xms_idx = np.argmin(np.abs(x - xms))
        A1 = np.trapz(dy_dx_vals[xs_idx:xms_idx+1], x[xs_idx:xms_idx+1])
    else:
        A1 = np.nan
    
    if not np.isnan(xms) and not np.isnan(xe):
        xms_idx = np.argmin(np.abs(x - xms))
        xe_idx = np.argmin(np.abs(x - xe))
        A2 = np.trapz(dy_dx_vals[xms_idx:xe_idx+1], x[xms_idx:xe_idx+1])
    else:
        A2 = np.nan
    
    # ========================================================================
    # 12. TABLE S2 FEATURES: AREA ASYMMETRICAL INDEX
    # ========================================================================
    
    if not np.isnan(A1) and A1 != 0:
        area_asymmetry_index = A2 / A1
    else:
        area_asymmetry_index = np.nan
    
    # ========================================================================
    # 13. TABLE S2 FEATURES: MAXIMUM SLOPE
    # ========================================================================
    
    maximum_slope = dy_xms
    
    # ========================================================================
    # 14. TABLE S2 FEATURES: POSITIVE SECOND-DERIVATIVE PEAK HEIGHT
    # ========================================================================
    
    positive_second_deriv_peak = d2y_xp1
    
    # ========================================================================
    # 15. TABLE S2 FEATURES: NEGATIVE SECOND-DERIVATIVE PEAK HEIGHT
    # ========================================================================
    
    negative_second_deriv_peak = np.abs(d2y_xp2) if not np.isnan(d2y_xp2) else np.nan
    
    # ========================================================================
    # 16. TABLE S2 FEATURES: PEAK ASYMMETRY INDEX
    # ========================================================================
    
    if not np.isnan(positive_second_deriv_peak) and not np.isnan(negative_second_deriv_peak):
        if negative_second_deriv_peak != 0:
            peak_asymmetry_index = positive_second_deriv_peak / negative_second_deriv_peak
        else:
            peak_asymmetry_index = np.nan
    else:
        peak_asymmetry_index = np.nan
    
    # ========================================================================
    # NEW FEATURE 1: Ct - Time where F(t) exceeds 20% of maximum
    # ========================================================================
    # Ct = time when F(t) first exceeds 0.2 * max(F(t))
    
    F_vals = sigmoid_5p(x, *params)
    F_max = np.max(F_vals)
    threshold_20pct = 0.2 * F_max  
    
    above_20pct = F_vals > threshold_20pct
    crossing_indices = np.where(above_20pct)[0]
    
    if len(crossing_indices) > 0:
        # First crossing with interpolation
        ct_idx = crossing_indices[0]
        if ct_idx > 0:
            x1, x2 = x[ct_idx-1], x[ct_idx]
            y1, y2 = F_vals[ct_idx-1], F_vals[ct_idx]
            Ct = x1 + (threshold_20pct - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x[ct_idx]
        else:
            Ct = x[ct_idx]
    else:
        Ct = np.nan
    
    # ========================================================================
    # NEW FEATURE 2: Cy0 - Y-intercept of tangent at inflection point
    # ========================================================================
    # Cy0 = Y-intercept of tangent line at inflection point
    # Inflection point is where F''(t) = 0, which occurs at t = Cs for 5PL
    
    x_inflection = Cs + (np.log(As) / Sc)
    y_inflection = sigmoid_5p(x_inflection, *params)
    dy_inflection = sigmoid_5p_first_derivative(x_inflection, *params)
    
    # Tangent line: y - y_inf = m(x - x_inf)
    # Intersection with abscissa (y = 0):
    # 0 = y_inf + dy_inflection * (Cy0 - x_inflection)
    # Cy0 = x_inflection - (y_inflection / dy_inflection)
    
    if dy_inflection != 0:
        Cy0 = x_inflection - (y_inflection / dy_inflection)
    else:
        Cy0 = np.nan
    
    # ========================================================================
    # NEW FEATURE 3: -log10(F0) - Log10 of initial fluorescence
    # ========================================================================
    # F0 = F(t=0)
    
    # F0 = sigmoid_5p(0, *params)
    F0 = y[0]
    
    # Avoid log of zero or negative
    if F0 > 0:
        log_F0 = -np.log10(F0)
    else:
        log_F0 = np.nan
    
    # ========================================================================
    # RETURN ALL KINETIC PARAMETERS
    # ========================================================================
    
    return {
        # Basic critical points
        'xms': xms,
        'xs': xs,
        'xe': xe,
        'xp1': xp1,
        'xp2': xp2,
        'TH': TH,
        
        # Y-values at critical points
        'y_xms': y_xms,
        'y_xs': y_xs,
        'y_xe': y_xe,
        'y_xp1': y_xp1,
        'y_xp2': y_xp2,
        
        # First and second derivative values
        'dy_xms': dy_xms,
        'dy_xp1': dy_xp1,
        'dy_xp2': dy_xp2,
        'd2y_xp1': d2y_xp1,
        'd2y_xp2': d2y_xp2,
        
        # ====== TABLE S2 FEATURES ======
        
        # Distance metrics
        'threshold_distance': threshold_distance,
        'first_half_distance': first_half_distance,
        'second_half_distance': second_half_distance,
        'distance_asymmetry_index': distance_asymmetry_index,
        'peak_shifting_distance': peak_shifting_distance,
        
        # Area under curve
        'A1': A1,
        'A2': A2,
        'area_asymmetry_index': area_asymmetry_index,
        
        # Peak heights
        # 'maximum_slope': maximum_slope,   ---> dy_xms
        # 'positive_second_deriv_peak': positive_second_deriv_peak,   ---> d2y_xp1
        # 'negative_second_deriv_peak': negative_second_deriv_peak,   ---> np.abs(d2y_xp2)
        'peak_asymmetry_index': peak_asymmetry_index,
        
        # ====== NEW FEATURES ======
        'Ct': Ct,              # Time when F(t) exceeds 20% of maximum
        'Cy0': Cy0,            # Y-intercept of tangent at inflection point
        'log_F0': log_F0,      # -log10(F0), where F0 = F(0)
        'F0': F0,
    }


def extract_kinetic_parameters_original(x, y, threshold=0.1, deriv_method='gradient'):
    """
    Extract kinetic parameters from ORIGINAL (unfitted) curve
    
    Parameters
    ----------
    x : array
        Time values
    y : array
        Original curve values (NOT fitted)
    threshold : float
        Fraction of max slope (0.1 = 10%)
    deriv_method : str
        Method for derivatives ('gradient', 'savgol', 'diff')
    
    Returns
    -------
    dict with all kinetic parameters
    """
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # ========================================================================
    # 1. CALCULATE DERIVATIVES FROM ORIGINAL CURVE
    # ========================================================================
    
    dy_dx_vals = calculate_first_derivative(x, y, method=deriv_method)
    d2y_dx2_vals = calculate_second_derivative(x, y, method=deriv_method)
    
    # ========================================================================
    # 2. FIND xms (maximum slope)
    # ========================================================================
    
    max_slope_idx = np.argmax(dy_dx_vals)
    xms = x[max_slope_idx]
    dy_xms = dy_dx_vals[max_slope_idx]
    y_xms = y[max_slope_idx]
    
    # ========================================================================
    # 3. FIND xs, xe (threshold crossing points)
    # ========================================================================
    
    if threshold < 1:
        TH = threshold * dy_xms
    else:
        TH = threshold
    
    above_threshold = dy_dx_vals > TH
    rising_indices = np.where(above_threshold)[0]
    
    # xs: first crossing
    if len(rising_indices) > 0:
        xs_idx = rising_indices[0]
        if xs_idx > 0:
            x1, x2 = x[xs_idx-1], x[xs_idx]
            y1, y2 = dy_dx_vals[xs_idx-1], dy_dx_vals[xs_idx]
            xs = x1 + (TH - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x[xs_idx]
        else:
            xs = x[xs_idx]
    else:
        xs = np.nan
    
    # xe: last crossing
    if len(rising_indices) > 0:
        xe_idx = rising_indices[-1]
        if xe_idx < len(x) - 1:
            x1, x2 = x[xe_idx], x[xe_idx+1]
            y1, y2 = dy_dx_vals[xe_idx], dy_dx_vals[xe_idx+1]
            xe = x1 + (TH - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x[xe_idx]
        else:
            xe = x[xe_idx]
    else:
        xe = np.nan
    
    # ========================================================================
    # 4. FIND xp1, xp2 (peaks of 2nd derivative)
    # ========================================================================
    
    from scipy.signal import find_peaks
    
    peaks_pos, _ = find_peaks(d2y_dx2_vals)
    if len(peaks_pos) > 0:
        xp1_idx = peaks_pos[np.argmax(d2y_dx2_vals[peaks_pos])]
        xp1 = x[xp1_idx]
        dy_xp1 = dy_dx_vals[xp1_idx]
        d2y_xp1 = d2y_dx2_vals[xp1_idx]
    else:
        xp1 = np.nan
        dy_xp1 = np.nan
        d2y_xp1 = np.nan
    
    peaks_neg, _ = find_peaks(-d2y_dx2_vals)
    if len(peaks_neg) > 0:
        xp2_idx = peaks_neg[np.argmin(d2y_dx2_vals[peaks_neg])]
        xp2 = x[xp2_idx]
        dy_xp2 = dy_dx_vals[xp2_idx]
        d2y_xp2 = d2y_dx2_vals[xp2_idx]
    else:
        xp2 = np.nan
        dy_xp2 = np.nan
        d2y_xp2 = np.nan
    
    # ========================================================================
    # 5. Y-VALUES AT CRITICAL POINTS
    # ========================================================================
    
    y_xs = y[np.argmin(np.abs(x - xs))] if not np.isnan(xs) else np.nan
    y_xe = y[np.argmin(np.abs(x - xe))] if not np.isnan(xe) else np.nan
    y_xp1 = y[np.argmin(np.abs(x - xp1))] if not np.isnan(xp1) else np.nan
    y_xp2 = y[np.argmin(np.abs(x - xp2))] if not np.isnan(xp2) else np.nan
    
    # ========================================================================
    # 6-16. TABLE S2 FEATURES
    # ========================================================================
    
    threshold_distance = xe - xs if (not np.isnan(xs) and not np.isnan(xe)) else np.nan
    first_half_distance = xms - xs if not np.isnan(xs) else np.nan
    second_half_distance = xe - xms if not np.isnan(xe) else np.nan
    
    if not np.isnan(first_half_distance) and first_half_distance != 0:
        distance_asymmetry_index = second_half_distance / first_half_distance
    else:
        distance_asymmetry_index = np.nan
    
    peak_shifting_distance = xp2 - xp1 if (not np.isnan(xp1) and not np.isnan(xp2)) else np.nan
    
    # Area under curve
    if not np.isnan(xs) and not np.isnan(xms):
        xs_idx = np.argmin(np.abs(x - xs))
        xms_idx = np.argmin(np.abs(x - xms))
        A1 = np.trapz(dy_dx_vals[xs_idx:xms_idx+1], x[xs_idx:xms_idx+1])
    else:
        A1 = np.nan
    
    if not np.isnan(xms) and not np.isnan(xe):
        xms_idx = np.argmin(np.abs(x - xms))
        xe_idx = np.argmin(np.abs(x - xe))
        A2 = np.trapz(dy_dx_vals[xms_idx:xe_idx+1], x[xms_idx:xe_idx+1])
    else:
        A2 = np.nan
    
    if not np.isnan(A1) and A1 != 0:
        area_asymmetry_index = A2 / A1
    else:
        area_asymmetry_index = np.nan
    
    maximum_slope = dy_xms
    positive_second_deriv_peak = d2y_xp1
    negative_second_deriv_peak = np.abs(d2y_xp2) if not np.isnan(d2y_xp2) else np.nan
    
    if not np.isnan(positive_second_deriv_peak) and not np.isnan(negative_second_deriv_peak):
        if negative_second_deriv_peak != 0:
            peak_asymmetry_index = positive_second_deriv_peak / negative_second_deriv_peak
        else:
            peak_asymmetry_index = np.nan
    else:
        peak_asymmetry_index = np.nan
    
    # ========================================================================
    # NEW FEATURE 1: Ct - Time where F(t) exceeds 20% of maximum
    # ========================================================================
    params, _ = fit_5p(x, y, normalize=True)
    Fm_fit, Fb_fit, Sc_fit, Cs_fit, As_fit = params
    
    F_vals = sigmoid_5p(x, *params)
    F_vals_dydx = sigmoid_5p_first_derivative(x, *params)
    
    F_min = np.min(F_vals)
    F_max = np.max(F_vals)
    
    if F_max > F_min:
        F_norm = (F_vals - F_min) / (F_max - F_min)
    else:
        F_norm = np.zeros_like(F_vals)
    
    above_20pct = F_norm > 0.2
    crossing_indices = np.where(above_20pct)[0]
    
    if len(crossing_indices) > 0:
        ct_idx = crossing_indices[0]
        if ct_idx > 0:
            x1, x2 = x[ct_idx-1], x[ct_idx]
            # Interpolate using the [0, 1] NORMALIZED y-values and the 0.2 threshold
            y1, y2 = F_norm[ct_idx-1], F_norm[ct_idx] 
            Ct = x1 + (0.2 - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x[ct_idx]
        else:
            Ct = x[ct_idx]
    else:
        Ct = np.nan
    
    # ========================================================================
    # NEW FEATURE 2: Cy0 - Y-intercept of tangent at inflection point
    # ========================================================================
    # For original curve, inflection point is at xp1 (positive peak of 2nd deriv)
    
    x_inflection = Cs_fit + (np.log(As_fit) / Sc_fit)
    y_inflection = sigmoid_5p(x_inflection, *params)
    dy_inflection = sigmoid_5p_first_derivative(x_inflection, *params)
    
    if dy_inflection != 0:
        Cy0 = x_inflection - (y_inflection / dy_inflection)
    else:
        Cy0 = np.nan
    
    # ========================================================================
    # NEW FEATURE 3: -log10(F0) - Log10 of initial fluorescence
    # ========================================================================
    
    F0 = y[0]  # F at first time point
    
    if F0 > 0:
        log_F0 = -np.log10(F0)
    else:
        log_F0 = np.nan
    
    # ========================================================================
    # RETURN ALL PARAMETERS
    # ========================================================================
    
    return {
        # '5pl_params': params,
        'Fm': Fm_fit,
        'Fb': Fb_fit,
        'Sc': Sc_fit,
        'Cs': Cs_fit,
        'As': As_fit,
        'Send': np.mean(dy_dx_vals[-5:]),
        'Send_abs': np.mean(np.abs(dy_dx_vals[-5:])),
        'Send_fit': np.mean(F_vals_dydx[-5:]),
        'Send_fit_abs': np.mean(np.abs(F_vals_dydx[-5:])),
        'F_max': F_max,
        'ct_idx': ct_idx,
        "5p_sigmoid_fitted": F_vals,
        "5p_sigmoid_fitted_dydx": F_vals_dydx,

        # Basic critical points
        'xms': xms,
        'xs': xs,
        'xe': xe,
        'xp1': xp1,
        'xp2': xp2,
        'TH': TH,
        
        # Y-values at critical points
        'y_xms': y_xms,
        'y_xs': y_xs,
        'y_xe': y_xe,
        'y_xp1': y_xp1,
        'y_xp2': y_xp2,
        
        # First and second derivative values
        'dy_xms': dy_xms,
        'dy_xp1': dy_xp1,
        'dy_xp2': dy_xp2,
        'd2y_xp1': d2y_xp1,
        'd2y_xp2': d2y_xp2,
        
        # ====== TABLE S2 FEATURES ======
        
        # Distance metrics
        'threshold_distance': threshold_distance,
        'first_half_distance': first_half_distance,
        'second_half_distance': second_half_distance,
        'distance_asymmetry_index': distance_asymmetry_index,
        'peak_shifting_distance': peak_shifting_distance,
        
        # Area under curve
        'A1': A1,
        'A2': A2,
        'area_asymmetry_index': area_asymmetry_index,
        
        # Peak heights
        # 'maximum_slope': maximum_slope,
        # 'positive_second_deriv_peak': positive_second_deriv_peak,
        # 'negative_second_deriv_peak': negative_second_deriv_peak,
        'peak_asymmetry_index': peak_asymmetry_index,
        
        # ====== NEW FEATURES ======
        
        'Ct': Ct,              # Time when F(t) exceeds 20% of maximum
        'Cy0': Cy0,            # Y-intercept of tangent at inflection point
        'log_F0': log_F0,      # -log10(F0), where F0 = F(0)
        'F0': F0,
    }
# ============================================================================
# FITTING FUNCTIONS
# ============================================================================

def _fit_sigmoid(sigmoid_func, x, y, p0=None, bounds=None, maxfev=100000, 
                 normalize=True, func_name="sigmoid"):
    """Generic sigmoid fitting function."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Remove NaNs and Infs
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    
    if len(x) < 3:
        raise ValueError(f"Need at least 3 valid data points, got {len(x)}")
    
    # Store original scales for denormalization
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    if normalize:
        x_norm = (x - x_min) / (x_max - x_min + 1e-10)
        y_norm = (y - y_min) / (y_max - y_min + 1e-10)
    else:
        x_norm, y_norm = x, y

    if p0 is None:
        raise ValueError(f"{func_name}: Must provide initial guess p0")
    
    if bounds is None:
        raise ValueError(f"{func_name}: Must provide bounds")

    # Convert bounds from list of tuples to two formats
    bounds_minimize = [(float(b[0]), float(b[1])) for b in bounds]
    lower = np.array([float(b[0]) for b in bounds])
    upper = np.array([float(b[1]) for b in bounds])
    bounds_lsq = (lower, upper)

    def objective(params):
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            try:
                pred = sigmoid_func(x_norm, *params)
                if np.any(~np.isfinite(pred)):
                    return 1e10
                residuals = y_norm - pred
                return np.sum(residuals**2)
            except:
                return 1e10

    # Try L-BFGS-B first
    result = minimize(objective, p0, bounds=bounds_minimize, method='L-BFGS-B', options={'maxfun': maxfev, 'ftol': 1e-9})
    
    # Fallback to least_squares if failed
    if not result.success:
        def residuals(params):
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                try:
                    return y_norm - sigmoid_func(x_norm, *params)
                except:
                    return np.full_like(y_norm, 1e10)
        
        result = least_squares(residuals, p0, bounds=bounds_lsq, max_nfev=maxfev)
    
    # Denormalize parameters if normalization was applied
    params = result.x
    if normalize:
        params = _denormalize_params(params, func_name, x_min, x_max, y_min, y_max)
    
    return params, result


def _denormalize_params(params, func_name, x_min, x_max, y_min, y_max):
    """Denormalize fitted parameters based on sigmoid model type."""
    params = np.array(params, dtype=float)
    
    if func_name == "5PL":
        # sigmoid_5p(x, Fm, Fb, Sc, Cs, As): y = Fm / (1 + exp(-Sc * (x - Cs)))^As + Fb
        # Denormalize: Fm (max), Fb (min), Sc (slope), Cs (center), As (asymmetry - stays same)
        Fm, Fb, Sc, Cs, As = params
        Fm_denorm = Fm * (y_max - y_min)
        Fb_denorm = Fb * (y_max - y_min) + y_min
        Cs_denorm = Cs * (x_max - x_min) + x_min
        Sc_denorm = Sc / (x_max - x_min + 1e-10)
        return np.array([Fm_denorm, Fb_denorm, Sc_denorm, Cs_denorm, As])
    else:
        # Unknown model, return as is
        return params

def fit_5p(x, y, p0=None, bounds=None, maxfev=100000, normalize=True):
    """Fit 5-Parameter Logistic (5PL)"""
    if p0 is None:
        if normalize:
            # p0 = (Fm_amplitude, Fb_bottom, slope, center, asymmetry)
            p0 = (1.0, 0.0, 10.0, 0.5, 1.0) 
        else:
            p0 = (np.max(y) - np.min(y), np.min(y), 1.0, np.median(x), 1.0)
            
    if bounds is None:
        if normalize:
            bounds = [(-0.5, 2.0), (-0.5, 1.5), (0.01, 500), (-0.5, 1.5), (0.1, 10)]
        else:
            bounds = [(-1e6, 1e6), (-1e6, 1e6), (0.01, 100), (np.min(x), np.max(x)), (0.1, 10)]
    
    return _fit_sigmoid(sigmoid_5p, x, y, p0=p0, bounds=bounds, 
                       maxfev=maxfev, normalize=normalize, func_name="5PL")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def predict(sigmoid_func, x, params):
    """Predict y values using fitted sigmoid"""
    return sigmoid_func(x, *params)


def calculate_loss(y_true, y_pred):
    """Calculate various loss metrics"""
    ssr = np.sum((y_true - y_pred)**2)
    mse = ssr / len(y_true)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - (ssr / (ss_tot + 1e-10))
    
    return {
        'ssr': ssr,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared
    }