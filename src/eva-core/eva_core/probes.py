import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class LinearProbe(nn.Module):
    """
    Sonde Linéaire (Linear Probe) pour la sécurité cognitive.
    Vérifie si la représentation interne du LLM est cohérente avec sa sortie textuelle.
    Inspiré de l'hypothèse Othello / Interpretability Research.
    """
    def __init__(self, activation_dim=4096, hidden_dim=256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(activation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def verify_intent(self, activations, stated_intent_favorable):
        """
        Analyse les activations neuronales.
        Retourne True si l'activation interne corrobore l'intention déclarée.
        """
        internal_truth_score = self.classifier(activations)
        
        # Détection de mensonge/hallucination cognitive :
        # Si le LLM veut 'Acheter' mais que la sonde détecte une tendance 'Baisse' (score < 0.5)
        is_consistent = (internal_truth_score > 0.5) == stated_intent_favorable
        
        return is_consistent, internal_truth_score.item()

def check_cognitive_sincerity(activations, text_response, target_action):
    """
    Vérification de la "Sincérité Cognitive" (Othello-GPT).
    Sonde les activations internes du LLM pour vérifier si sa 'pensée' 
    graphique/spatiale du marché correspond à son 'discours' textuel.
    """
    probe = LinearProbe()
    # On simule la sonde analysant si le modèle 'croit' vraiment à la hausse
    is_ok, internal_prob = probe.verify_intent(activations, target_action == "BUY")
    
    logger.info(f"Sincerity Probe: internal_prob={internal_prob:.4f}, action={target_action}")
    
    # Si le LLM justifie par texte un achat mais que sa probabilité interne de hausse est faible (< 0.65)
    if target_action == "BUY" and internal_prob < 0.65:
        return False, "SINCERITY_FAILURE: LLM text justifies BUY but internal activations show BEARISH bias."
        
    return True, "SINCERITY_PASSED"
