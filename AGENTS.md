# CONVENTIONS ET DIRECTIVES DES AGENTS

Ce document définit les règles strictes que tous les Agents (Développeurs humains et IA) doivent respecter lors de la modification du code source de THE HIVE.

## 1. LANGUE (FRANÇAIS)

**Règle absolue :** La langue officielle du projet est le **Français**.

*   **Docstrings :** Tous les docstrings (modules, classes, fonctions) doivent être rédigés en français.
*   **Commentaires :** Tous les commentaires inline ou blocs doivent être rédigés en français.
*   **Logs :** Les messages de log (`logger.info`, `logger.error`) doivent être en français.
*   **Exceptions :** Les messages d'erreur (`raise ValueError("Message")`) doivent être en français.

*Exception :* Les noms de variables, fonctions et classes restent en **Anglais** (ex: `user_id`, `get_account_balance`, `TradeOrder`) pour respecter les conventions Python (PEP 8) et maintenir la cohérence avec les bibliothèques externes.

## 2. DOCSTRINGS (GOOGLE STYLE)

Tous les modules, classes et fonctions publiques doivent avoir un docstring détaillé suivant le format **Google Style**.

### Structure Requise
```python
def ma_fonction(param1: int, param2: str) -> bool:
    """
    Description courte et impérative de la fonction.

    Description plus détaillée si nécessaire, expliquant le "pourquoi" ou
    les effets de bord importants.

    Args:
        param1 (int): Description du premier paramètre.
        param2 (str): Description du second paramètre.

    Returns:
        bool: Description de la valeur de retour.
              True si succès, False sinon.

    Raises:
        ValueError: Si param1 est négatif.
        ConnectionError: Si le service est injoignable.
    """
    pass
```

### Règles de Contenu
1.  **Args :** Chaque paramètre doit être listé avec son type et une description claire.
2.  **Returns :** Le type de retour et sa signification doivent être explicites.
3.  **Raises :** Toutes les exceptions levées explicitement doivent être documentées.

## 3. COMMENTAIRES

*   Les commentaires doivent expliquer le **POURQUOI**, pas le COMMENT (le code le dit déjà).
*   Utilisez des commentaires pour :
    *   Justifier une décision complexe ou non-intuitive.
    *   Expliquer une règle métier ("Loi 2 - Constitution").
    *   Marquer des zones de code temporaires (`TODO`, `FIXME`).

## 4. EXEMPLE COMPLET

```python
class RiskValidator:
    """
    Valide les ordres de trading selon les règles de gestion des risques.
    """

    def validate_order(self, order: TradeOrder) -> dict[str, Any]:
        """
        Vérifie la conformité d'un ordre avant son exécution.

        Cette méthode applique la Loi 2 de la Constitution (Protection du Capital).
        Elle vérifie notamment le drawdown journalier et l'exposition totale.

        Args:
            order (TradeOrder): L'objet ordre à valider.

        Returns:
            dict[str, Any]: Un dictionnaire contenant :
                - allowed (bool): True si l'ordre est autorisé.
                - reason (str | None): La raison du refus si applicable.

        Raises:
            ValueError: Si l'ordre est mal formé.
        """
        # Vérification du drawdown (Règle critique)
        if self.current_drawdown > 0.04:
            return {"allowed": False, "reason": "MAX_DAILY_LOSS_REACHED"}

        return {"allowed": True, "reason": None}
```
