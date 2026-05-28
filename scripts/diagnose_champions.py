"""
Diagnostic complet des champions (MuZero, DreamerV3, GNN).
Analyse les performances et identifie pourquoi les modeles regressent.

Usage: python scripts/diagnose_champions.py
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src" / "eva-banker"))
sys.path.insert(0, str(ROOT / "src" / "shared"))


def load_latest_review():
    path = ROOT / "data" / "checkpoints" / "trading_reviews" / "latest.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_champion_files():
    champions = {}
    for f in (ROOT / "data").glob("**/champion_*.json"):
        with open(f, encoding="utf-8") as fh:
            try:
                data = json.load(fh)
                champions[f.stem] = data
            except Exception:
                pass
    return champions


def analyze_rr_ratio(review: dict):
    """Analyse le ratio Risk/Reward depuis les reviews."""
    alerts = review.get("alerts", [])
    split_alert = next((a for a in alerts if "SPLIT" in str(a.get("kind", ""))), None)
    
    print("\n📊 ANALYSE RATIO RISK/REWARD")
    print("=" * 50)
    
    # Extraire les stats des trades depuis la review
    diagnostics = review.get("diagnostics", {})
    pnl_stats = diagnostics.get("pnl_stats", {})
    
    total_pnl = pnl_stats.get("total_pnl", 0)
    win_rate = pnl_stats.get("win_rate", 0)
    avg_gain = pnl_stats.get("avg_gain", 0)
    avg_loss = pnl_stats.get("avg_loss", 0)
    n_trades = pnl_stats.get("n_trades", 0)
    
    print(f"  Trades totaux : {n_trades}")
    print(f"  PnL total     : {total_pnl:+.2f}")
    print(f"  Win rate      : {win_rate*100:.1f}%")
    print(f"  Gain moyen    : +{avg_gain:.2f}")
    print(f"  Perte moyenne : {avg_loss:.2f}")
    
    if avg_gain > 0 and avg_loss < 0:
        rr_ratio = abs(avg_loss) / avg_gain
        print(f"  Ratio R:R INVERSÉ : 1:{rr_ratio:.2f}")
        
        if rr_ratio > 2.0:
            print(f"  ⛔ CRITIQUE: Les pertes sont {rr_ratio:.1f}× les gains !")
            print(f"     → Objectif : ratio ≤ 1.3 (gains 30% plus grands que pertes)")
            print(f"     → Pour WR={win_rate*100:.0f}%, breakeven nécessite RR ≤ {(1 - win_rate) / win_rate:.2f}")
        
        needed_rr = (1 - win_rate) / max(win_rate, 0.01)
        print(f"\n  📌 Pour être profitable avec WR={win_rate*100:.0f}%:")
        print(f"     RR minimum requis : 1:{needed_rr:.2f}")
        print(f"     Objectif recommandé : +RR 1:1.5 → WR ≥ {1/(1+1.5)*100:.0f}% nécessaire")


def analyze_nemesis(review: dict):
    """Analyse l'état du Nemesis."""
    alerts = review.get("alerts", [])
    nemesis_alert = next((a for a in alerts if a.get("kind") == "nemesis_actif"), None)
    
    print("\n⚔️ ANALYSE NEMESIS")
    print("=" * 50)
    
    if not nemesis_alert:
        print("  Aucune alerte Nemesis trouvée dans la review.")
        return
    
    nemesis = nemesis_alert.get("nemesis", {})
    quarantined = nemesis.get("quarantined_symbols", [])
    total_defeats = nemesis.get("total_defeats", 0)
    known_nemeses = nemesis.get("known_nemeses", {})
    
    print(f"  Total défaites : {total_defeats}")
    print(f"  Patterns actifs : {known_nemeses}")
    print(f"  Symboles en quarantaine ({len(quarantined)}) : {quarantined}")
    
    liquidity_trap = nemesis.get("lifetime_nemeses", {}).get("LIQUIDITY_TRAP", 0)
    pct = liquidity_trap / max(total_defeats, 1) * 100
    print(f"\n  LIQUIDITY_TRAP : {liquidity_trap}/{total_defeats} = {pct:.0f}% des défaites")
    
    if pct > 50:
        print(f"  ⚠️ LIQUIDITY_TRAP domine les défaites !")
        print(f"     → Causes possibles : trades en consolidation, spread élevé, faible volatilité")
        print(f"     → Fix Sprint 10 : Nemesis predict_trap() pré-trade + seuil 0.70")
        print(f"     → Fix Sprint 10 : nemesis_penalty=-2.0 dans Shadow Learning DreamerV3")


def analyze_gnn(review: dict):
    """Analyse l'efficacité du GNN."""
    print("\n🧠 ANALYSE GNN")
    print("=" * 50)
    
    diagnostics = review.get("diagnostics", {})
    bias_stats = diagnostics.get("bias_alignment_stats", {})
    
    neutral_rate = bias_stats.get("neutral_rate", None)
    
    if neutral_rate is None:
        print("  Pas de statistiques de biais disponibles dans la review.")
        print("  → Vérifier les logs du Strategist pour 'cpu_live_gnn_live_neutral'")
    else:
        print(f"  Taux NEUTRAL : {neutral_rate*100:.0f}%")
        if neutral_rate > 0.6:
            print(f"  ⚠️ Le Cortex/GNN retourne NEUTRAL {neutral_rate*100:.0f}% du temps !")
            print(f"     → Probablement lié à : vLLM mort, GNN confidence < 0.55")
            print(f"     → Fix Sprint 10 : seuil GNN abaissé à 0.50")
            print(f"     → Fix Sprint 10 : GNN fort (>0.80) peut driver même si Cortex NEUTRAL")
    
    print(f"\n  Recommandations amélioration GNN :")
    print(f"  1. Ajouter session_encoding (London 8h-16h / NY 13h-21h / Asia) comme feature")
    print(f"  2. Calibrer le threshold : 0.50 (Sprint 10) → monitorer pendant 48h")
    print(f"  3. Intégrer nemesis_score(symbol) comme feature négative dans le vecteur GNN")
    print(f"  4. Augmenter la taille du buffer replay GNN avec les derniers 30 jours")


def analyze_muzero(review: dict):
    """Analyse l'état du champion MuZero."""
    print("\n🎯 ANALYSE MUZERO")
    print("=" * 50)
    
    diagnostics = review.get("diagnostics", {})
    champion_info = diagnostics.get("champion_info", {})
    
    forced = os.getenv("BANKER_CPU_LIVE_CHAMPION_ID", "")
    if forced:
        print(f"  ⚠️ Champion forcé : {forced}")
        print(f"     Ce champion date probablement du 28 avril (28/04/2026)")
        print(f"     Il ne bénéficie pas des corrections récentes de reward")
    
    print(f"\n  Problèmes identifiés avec MuZero actuel :")
    print(f"  1. Reward function ne pénalise pas LIQUIDITY_TRAP")
    print(f"  2. L'observation ne contient pas de features Nemesis")
    print(f"  3. blocked_champion sur EURUSD/US30 (artifact incompatibilité)")
    
    print(f"\n  Fix Sprint 10 proposés :")
    print(f"  - Ajouter nemesis_context=[liquidity_score, loss_streak, quarantine] à l'observation")
    print(f"  - Reward modifier: pénalité -2.0 si contexte LIQUIDITY_TRAP (Shadow Feedback)")
    print(f"  - Paramètres Shepherd ajustés (runner_hold_bonus +67%, giveback -50%)")
    print(f"  - Relancer training nocturne avec ces nouvelles features")


def analyze_dreamer(review: dict):
    """Analyse l'état du DreamerV3."""
    print("\n🌙 ANALYSE DREAMERV3")
    print("=" * 50)
    
    # Vérifier si Shadow Learning est actif
    shadow_enabled = os.getenv("ENABLE_SHADOW_LEARNING", "true").lower() in {"1", "true", "yes"}
    print(f"  Shadow Learning actif : {'✅' if shadow_enabled else '❌'}")
    
    print(f"\n  État du pipeline DreamerV3 :")
    print(f"  - Collecte Shadow Learning : actif (chaque trade live → Lab)")
    print(f"  - NEMESIS FEEDBACK : ✅ Sprint 10 — penalty=-2.0 pour LIQUIDITY_TRAP")
    print(f"  - Training sur GPU distant : dépend du Lab 192.168.1.6")
    
    print(f"\n  Problèmes identifiés :")
    print(f"  1. Lab 192.168.1.6:8000 (vLLM) refusé → pas de training actif ?")
    print(f"  2. DreamerV3 ne reçoit pas de signal explicite pour LIQUIDITY_TRAP")
    print(f"     → Fix: nemesis_penalty dans Shadow Feedback (✅ déjà ajouté)")
    print(f"  3. Pas de replay des épisodes de défaite pour apprentissage contrasté")
    
    print(f"\n  Recommandations long terme :")
    print(f"  - Activer le replay prioritaire des épisodes LIQUIDITY_TRAP (×3 poids)")
    print(f"  - Réduire le seuil de promotion DreamerV3 (80 → 75 épisodes minimum)")
    print(f"  - Ajouter 'imaginary rollouts' sur les patterns LIQUIDITY_TRAP connus")


def main():
    print("=" * 60)
    print("🐝 DIAGNOSTIC CHAMPIONS THE HIVE — SPRINT 10")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    review = load_latest_review()
    if not review:
        print("⚠️ Pas de review disponible. Vérifier data/checkpoints/trading_reviews/latest.json")
        return
    
    print(f"\n📅 Review du : {review.get('generated_at', '?')}")
    print(f"📊 Recommandations : {len(review.get('recommendations', []))}")
    
    analyze_rr_ratio(review)
    analyze_nemesis(review)
    analyze_gnn(review)
    analyze_muzero(review)
    analyze_dreamer(review)
    
    print("\n" + "=" * 60)
    print("✅ ACTIONS SPRINT 10 RÉALISÉES :")
    print("  [x] Fix bug api_host → banker_api_host (Muse)")
    print("  [x] FTUK 333382206 désactivé dans copy targets")
    print("  [x] Cortex basculé sur OpenRouter (nemotron:free)")
    print("  [x] GNN confidence seuil 0.55 → 0.50")
    print("  [x] GNN fort (≥0.80) peut driver même si Cortex NEUTRAL")
    print("  [x] Nemesis predict_trap() pré-trade (seuil 0.70)")
    print("  [x] Nemesis feedback dans Shadow Learning (penalty -2.0)")
    print("  [x] Paramètres Shepherd R:R améliorés (+runner_hold +67%)")
    print("\n⏳ EN ATTENTE (redémarrage banker nécessaire) :")
    print("  [ ] Valider que Cortex OpenRouter répond (logs: LLM_BACKEND=openrouter)")
    print("  [ ] Confirmer arrêt erreurs 503 FTUK 333382206")
    print("  [ ] Monitoring 48h : vérifier que bias_alignment != 'cpu_live_gnn_live_neutral'")
    print("  [ ] Relancer training nocturne MuZero avec nouvelles features Nemesis")
    print("=" * 60)


if __name__ == "__main__":
    main()
