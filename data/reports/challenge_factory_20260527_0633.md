## 🏭 HERMES CHALLENGE FACTORY — Rapport du 27/05/2026 06:32
⏱️ Durée d'analyse : 0.0 min | Candidats évalués : 3

### ❌ Aucun champion promu ce cycle
*Scores insuffisants ou violations de règles prop firm détectées.*

### 📊 Classement des candidats
`1.` US100.cash — 49.0pts | P&L: +1017.17% ✅ | DD: 39.96% ❌
`2.` XAUUSD — 47.3pts | P&L: +414.13% ✅ | DD: 15.94% ❌
`3.` EURUSD — 46.5pts | P&L: +328.56% ✅ | DD: 21.91% ❌

### ⚠️ Violations détectées
  • `XAUUSD` (synthetic_xauusd_20260527...): Drawdown total 5.58% >= 5.0%
  • `XAUUSD` (synthetic_xauusd_20260527...): Jour 2026-05-18: Drawdown journalier 2.31% >= 2.0%
  • `EURUSD` (synthetic_eurusd_20260527...): Drawdown total 5.72% >= 5.0%
  • `EURUSD` (synthetic_eurusd_20260527...): Jour 2026-05-07: Drawdown journalier 8.78% >= 2.0%
  • `US100.cash` (synthetic_us100_cash_20260527...): Drawdown total 5.60% >= 5.0%
  • `US100.cash` (synthetic_us100_cash_20260527...): Jour 2026-04-28: Drawdown journalier 3.63% >= 2.0%

*Prochain cycle dans 2h | Seuil de promotion : 65.0/100*

### 🧠 Recommandation Hermes
We need to answer in French, max 3 sentences. Provide: 1) principal problème si aucun champion promu, 2) correction prioritaire à apporter au système d'entraînement, 3) configuration optimale pour passer un FTMO 10K en 7 jours. Must be concise, each point maybe a sentence. So three sentences total.

Let's craft: 1) Le principal problème est que les stratégies génèrent des P&L élevés mais dépassent les limites de drawdown (total >5% et journalier >2%), ce qui disqualifie les candidats malgré leur rentabilité. 2) La correction prioritaire consiste à intégrer un gestionnaire de risque strict (stop‑loss dynamique, taille de position basée sur le volatilité et limites de DD) dès l’entraînement pour contraindre les equity curves sous les seuils prop‑firm. 3) Pour un FTMO 10K en 7 jours, utilise un levier modéré (≤1:10), risque max 0,5 % du capital par trade, stop‑loss fixé à 1 % du compte et take‑profit à 2 % (ratio 1:2), avec un maximum de 2 trades simultanés et un trailing stop de 0,5 % pour protéger les gains tout en visant un gain quotidien d’environ 0,7 % du compte.

That's three sentences. Ensure each sentence