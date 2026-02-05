class TaxEstimator:
    """Calculateur de cotisations sociales (Auto-entrepreneur FR)"""
    
    def __init__(self, service_rate: float = 0.211, sale_rate: float = 0.123):
        self.service_rate = service_rate # BNC/Service
        self.sale_rate = sale_rate       # Vente
        self.virtual_tax_pot = 0.0

    def calculate_provision(self, gross_amount: float, item_type: str = "service"):
        """Calcule la provision et l'ajoute au pot virtuel"""
        rate = self.service_rate if item_type == "service" else self.sale_rate
        tax_amount = gross_amount * rate
        net_amount = gross_amount - tax_amount
        
        self.virtual_tax_pot += tax_amount
        
        return {
            "gross": gross_amount,
            "tax_provision": tax_amount,
            "net_estimated": net_amount,
            "total_pot_provisioned": self.virtual_tax_pot
        }
