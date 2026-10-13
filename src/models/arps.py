import numpy as np
from scipy.optimize import curve_fit
import warnings

def exponential_decline(t, qi, di):
    """
    Declinación exponencial (b = 0).
    t: tiempo (ej. meses)
    qi: tasa de producción inicial
    di: tasa de declinación nominal inicial
    """
    return qi * np.exp(-di * t)

def hyperbolic_decline(t, qi, di, b):
    """
    Declinación hiperbólica de Arps.
    b: factor de declinación (0 < b < 1 para convencionales, b > 1 común en shale inicial)
    """
    # Evitar división por cero
    if b == 0:
        return exponential_decline(t, qi, di)
    
    # Manejar posibles valores negativos en la base (físicamente irreal, pero curve_fit puede probarlos)
    base = 1 + b * di * t
    base = np.maximum(base, 1e-10) 
    
    return qi / np.power(base, 1/b)

def modified_arps(t, qi, di, b, d_term):
    if b == 0:
        return exponential_decline(t, qi, di)
        
    t_switch = (di / d_term - 1) / (b * di)
    t = np.asarray(t, dtype=float)
    q = np.zeros_like(t)
    
    hyp_mask = t <= t_switch
    q[hyp_mask] = hyperbolic_decline(t[hyp_mask], qi, di, b)
    
    if np.any(~hyp_mask):
        q_switch = hyperbolic_decline(t_switch, qi, di, b)
        exp_mask = ~hyp_mask
        q[exp_mask] = exponential_decline(t[exp_mask] - t_switch, q_switch, d_term)
        
    return q

def stretched_exponential(t, qi, tau, n):
    """
    Modelo de Stretched Exponential Decay.
    q(t) = qi * exp(-(t / tau)^n)
    tau: parámetro temporal
    n: exponente de forma (0 < n <= 1)
    Físicamente robusto para flujos transientes severos.
    """
    t = np.asarray(t, dtype=float)
    tau = max(tau, 1e-5)
    return qi * np.exp(- (t / tau)**n)

class ArpsModeler:
    """
    Clase orientada a objetos para ajustar modelos de declinación a datos reales.
    """
    def __init__(self, model_type='hyperbolic', d_term_annual=0.08):
        # model_type puede ser 'hyperbolic', 'modified', o 'stretched'
        self.model_type = model_type
        self.params = {}
        # d_term_annual = 0.08 significa 8% de declinación anual terminal.
        # Lo pasamos a mensual:
        self.d_term_monthly = d_term_annual / 12.0
        
    def fit(self, t, q):
        """
        Ajusta la curva a los datos de producción usando mínimos cuadrados no lineales.
        Utiliza Weighted Least Squares (WLS) dando peso para minimizar el error relativo.
        """
        # Filtrar NaNs
        valid = ~np.isnan(q) & ~np.isnan(t)
        t_valid = t[valid]
        q_valid = q[valid]
        
        if len(t_valid) < 3:
            raise ValueError("No hay suficientes datos para ajustar la curva.")
            
        # Estimación inicial
        qi_guess = max(q_valid)
        di_guess = 0.1
        b_guess = 1.0
        
        # Ponderación suave para que no inestabilice la matriz Jacobiana
        sigma_weights = 1.0 / np.sqrt(np.maximum(q_valid, 1.0))
        
        # Parámetros de control base para evitar que el optimizador se cuelgue
        fit_kwargs = {
            'sigma': sigma_weights,
            'absolute_sigma': False,
            'maxfev': 2000,          # Límite de iteraciones
            'loss': 'soft_l1'        # Función de pérdida robusta contra outliers
        }
        
        if self.model_type == 'hyperbolic':
            bounds = ([0, 0, 0], [np.inf, np.inf, 2.5])
            p0 = [qi_guess, di_guess, b_guess]
            fit_kwargs['bounds'] = bounds
            fit_kwargs['p0'] = p0
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    popt, _ = curve_fit(hyperbolic_decline, t_valid, q_valid, **fit_kwargs)
                except RuntimeError:
                    # Fallback si no converge
                    popt = p0
                
            self.params = {'qi': popt[0], 'di': popt[1], 'b': popt[2]}
            
        elif self.model_type == 'modified':
            bounds = ([0, 0, 0], [np.inf, np.inf, 2.5])
            p0 = [qi_guess, di_guess, b_guess]
            fit_kwargs['bounds'] = bounds
            fit_kwargs['p0'] = p0
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    popt, _ = curve_fit(hyperbolic_decline, t_valid, q_valid, **fit_kwargs)
                except RuntimeError:
                    popt = p0
                
            self.params = {'qi': popt[0], 'di': popt[1], 'b': popt[2], 'd_term': self.d_term_monthly}
            
        elif self.model_type == 'stretched':
            # Bounds: qi > 0, tau > 0, n entre 0 y 1.5
            bounds = ([0, 0, 0], [np.inf, np.inf, 1.5])
            p0 = [qi_guess, 10.0, 0.5] # tau guess=10, n guess=0.5
            fit_kwargs['p0'] = p0
            fit_kwargs['bounds'] = bounds
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    popt, _ = curve_fit(stretched_exponential, t_valid, q_valid, **fit_kwargs)
                except RuntimeError:
                    popt = p0
            self.params = {'qi': popt[0], 'tau': popt[1], 'n': popt[2]}
            
        return self.params

    def predict(self, t):
        """Genera la curva de producción predicha para los tiempos t."""
        if not self.params:
            raise ValueError("El modelo aún no fue ajustado (llamar a fit() primero).")
            
        if self.model_type == 'hyperbolic':
            return hyperbolic_decline(t, **self.params)
        elif self.model_type == 'modified':
            return modified_arps(t, **self.params)
        elif self.model_type == 'stretched':
            return stretched_exponential(t, **self.params)
            
    def calculate_eur(self, t_max):
        """
        Calcula el Estimated Ultimate Recovery (EUR) integrando numéricamente 
        la curva hasta un mes límite (ej. t_max = 360 meses = 30 años).
        """
        t_array = np.arange(0, t_max + 1)
        q_pred = self.predict(t_array)
        # Integración trapezoidal simple
        return np.trapz(q_pred, t_array)
