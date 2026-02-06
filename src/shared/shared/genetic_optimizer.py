import random
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class StrategyDNA:
    """
    Moteur Génétique pour l'évolution des stratégies d'EVA.
    Fait muter les hyperparamètres pour trouver le "Génome" optimal.
    """
    def __init__(self, population_size: int = 10):
        self.population = [self._generate_random_genome() for _ in range(population_size)]
        self.generation = 0

    def _generate_random_genome(self) -> Dict:
        return {
            "gnn_learning_rate": random.uniform(0.0001, 0.01),
            "risk_per_trade": random.uniform(0.1, 1.0),
            "sentiment_weight": random.uniform(0.1, 0.5),
            "score": 0.0
        }

    def evolve(self):
        """Phase de Mutation et Croisement."""
        self.generation += 1
        logger.info(f"StrategyDNA: Generation {self.generation} evolution started.")
        
        # Sélection des meilleurs (Mock: tri par score)
        self.population.sort(key=lambda x: x["score"], reverse=True)
        top_performers = self.population[:2]
        
        # Mutation
        new_population = top_performers.copy()
        while len(new_population) < 10:
            parent = random.choice(top_performers)
            child = parent.copy()
            # Mutation de 10% sur un paramètre
            gene = random.choice(["gnn_learning_rate", "risk_per_trade", "sentiment_weight"])
            child[gene] *= random.uniform(0.9, 1.1)
            child["score"] = 0.0
            new_population.append(child)
            
        self.population = new_population
        logger.info(f"StrategyDNA: New generation {self.generation} ready for Arena combat.")

    def get_best_genome(self):
        return max(self.population, key=lambda x: x["score"])
