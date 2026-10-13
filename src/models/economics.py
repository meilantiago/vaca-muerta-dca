import numpy as np

class EconomicEvaluator:
    """
    Evaluador económico para pozos de Vaca Muerta.
    Los parámetros por defecto representan estimaciones realistas del mercado
    argentino (Crudo Medanito) y costos operativos de la Cuenca Neuquina.
    """
    def __init__(self, 
                 oil_price_usd_bbl=70.0, 
                 opex_usd_bbl=6.0, 
                 royalty_rate=0.15, 
                 discount_rate_annual=0.10):
        """
        Parámetros por defecto:
        - oil_price_usd_bbl: 70 USD/bbl (Precio promedio Medanito / exportación).
        - opex_usd_bbl: 6 USD/bbl (Lifting cost promedio en Vaca Muerta).
        - royalty_rate: 0.15 (12% Regalías Provincia del Neuquén + 3% IIBB).
        - discount_rate_annual: 10% (Tasa de descuento estándar en la industria O&G).
        """
        self.oil_price = oil_price_usd_bbl
        self.opex = opex_usd_bbl
        self.royalties = royalty_rate
        
        # Convertir tasa anual a tasa efectiva mensual
        self.discount_rate_monthly = (1 + discount_rate_annual)**(1/12) - 1
        
        # Factor de conversión: 1 metro cúbico = 6.2898 barriles
        self.m3_to_bbl = 6.2898

    def calculate_cash_flow(self, q_m3_months):
        """
        Calcula el Flujo de Caja Libre (Free Cash Flow) mes a mes para una curva de producción.
        q_m3_months: array con la producción proyectada de petróleo en metros cúbicos por mes.
        
        Retorna:
        - array con el Cash Flow mensual en USD.
        """
        # Convertir producción de m3 a barriles
        q_bbl = np.asarray(q_m3_months) * self.m3_to_bbl
        
        # Ingresos Brutos = Producción * Precio
        gross_revenue = q_bbl * self.oil_price
        
        # Deducciones: Regalías provinciales
        net_revenue = gross_revenue * (1 - self.royalties)
        
        # Costos Operativos (Lifting Cost)
        total_opex = q_bbl * self.opex
        
        # Flujo de Caja Mensual (Simplificado pre-tax corporativo)
        cash_flow_monthly = net_revenue - total_opex
        
        return cash_flow_monthly

    def calculate_npv(self, q_m3_months):
        """
        Calcula el Net Present Value (NPV / Valor Presente Neto) en USD de la curva de producción.
        """
        cf_monthly = self.calculate_cash_flow(q_m3_months)
        
        # Vector de meses [0, 1, 2, ...]
        months = np.arange(len(cf_monthly))
        
        # Aplicar descuento: CF / (1 + r)^t
        discount_factors = (1 + self.discount_rate_monthly) ** months
        discounted_cf = cf_monthly / discount_factors
        
        return np.sum(discounted_cf)
        
    def calculate_payout_time(self, q_m3_months, capex_usd):
        """
        Calcula el tiempo de repago (Payout o ROI en meses) dado un CAPEX inicial (Costo de perforación).
        En Vaca Muerta, perforar y fracturar un pozo horizontal ronda los 12-15 Millones USD.
        """
        cf_monthly = self.calculate_cash_flow(q_m3_months)
        cumulative_cf = np.cumsum(cf_monthly)
        
        # Buscar el mes donde el flujo acumulado supera al CAPEX
        payout_mask = cumulative_cf >= capex_usd
        if not np.any(payout_mask):
            return np.inf # El pozo no se repaga nunca
            
        return np.argmax(payout_mask)
