class TokenBridge:
    """Pont vers les actifs réels tokenisés (RealT, Centrifuge)"""
    
    def __init__(self):
        self.portfolio = [
            {"id": "REALT-FLORIDA-102", "type": "Real Estate", "valuation": 1500.0, "yield": 0.091},
            {"id": "CENTRIFUGE-CFG-1", "type": "DeFi Credit", "valuation": 500.0, "yield": 0.12}
        ]

    def get_portfolio(self):
        return {
            "assets": self.portfolio,
            "total_valuation": sum(a["valuation"] for a in self.portfolio),
            "currency": "USD"
        }
