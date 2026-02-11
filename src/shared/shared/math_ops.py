try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

def symlog(x):
    """
    Transformation Symlog (Symmetric Logarithm) pour compresser les magnitudes.
    Formule : symlog(x) = sign(x) * ln(|x| + 1)
    """
    if np is not None and isinstance(x, (int, float)):
        return np.sign(x) * np.log1p(abs(x))
    elif np is not None and isinstance(x, np.ndarray):
        return np.sign(x) * np.log1p(np.abs(x))
    elif torch is not None and torch.is_tensor(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    else:
        # Fallback pour d'autres types scalaires ou si numpy est absent
        import math
        val = abs(x)
        res = math.log1p(val)
        return res if x > 0 else -res if x < 0 else 0

def inv_symlog(x):
    """
    Inverse de la transformation Symlog.
    Formule : inv_symlog(x) = sign(x) * (exp(|x|) - 1)
    """
    if np is not None and isinstance(x, (int, float)):
        return np.sign(x) * (np.expm1(abs(x)))
    elif np is not None and isinstance(x, np.ndarray):
        return np.sign(x) * (np.expm1(np.abs(x)))
    elif torch is not None and torch.is_tensor(x):
        return torch.sign(x) * (torch.expm1(torch.abs(x)))
    else:
        # Fallback
        import math
        val = abs(x)
        res = math.expm1(val)
        return res if x > 0 else -res if x < 0 else 0

def calculate_var(returns, confidence_level=0.95):
    """
    Value at Risk (VaR) historique.
    Mesure la perte maximale potentielle sur un horizon donné avec un niveau de confiance.
    """
    if len(returns) == 0:
        return 0.0
    if np is None:
        raise ImportError("Numpy is required for VaR calculation")
    return np.percentile(returns, (1 - confidence_level) * 100)

def calculate_cvar(returns, confidence_level=0.95):
    """
    Conditional Value at Risk (CVaR) ou Expected Shortfall.
    Moyenne des pertes au-delà de la VaR.
    """
    if np is None:
        raise ImportError("Numpy is required for CVaR calculation")
    var = calculate_var(returns, confidence_level)
    tail_losses = [r for r in returns if r <= var]
    if not tail_losses:
        return var
    return np.mean(tail_losses)
