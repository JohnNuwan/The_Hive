"""Lance une campagne AlphaEvolve offline sans impact sur le live."""

from __future__ import annotations

import argparse
from pathlib import Path

from eva_lab.alphaevolve import (
    charger_profil_de_base,
    construire_campagne,
    sauvegarder_campagne,
    scorer_variantes_depuis_ga,
)


def _build_parser() -> argparse.ArgumentParser:
    """Construit l'interface CLI de la campagne offline."""

    parser = argparse.ArgumentParser(
        description="Genere des variantes AlphaEvolve offline et les classe sans toucher a la prod.",
    )
    parser.add_argument(
        "--base-profile",
        type=Path,
        default=None,
        help="Fichier JSON de profil de base optionnel.",
    )
    parser.add_argument(
        "--parent-label",
        type=str,
        default="gen_scalp_20260428_045305_ckpt10500_manual",
        help="Label de la reference parente.",
    )
    parser.add_argument(
        "--variant-count",
        type=int,
        default=8,
        help="Nombre de variantes a generer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine pseudo-aleatoire stable.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/alphaevolve/campaigns"),
        help="Dossier cible des campagnes generees.",
    )
    parser.add_argument(
        "--ga-campaign-id",
        type=str,
        default=None,
        help="Campagne GA optionnelle pour enrichir les variantes avec un score proxy.",
    )
    return parser


def main() -> int:
    """Execute la generation offline de variantes AlphaEvolve.

    Returns:
        int: Code retour process standard.
    """

    parser = _build_parser()
    args = parser.parse_args()

    base_profile = charger_profil_de_base(args.base_profile)
    campagne = construire_campagne(
        parent_label=args.parent_label,
        base_profile=base_profile,
        variant_count=args.variant_count,
        seed=args.seed,
    )
    campagne.variants = scorer_variantes_depuis_ga(
        campagne.variants,
        campaign_id=args.ga_campaign_id,
    )
    output_path = sauvegarder_campagne(campagne, args.out_dir)
    print(f"Campagne AlphaEvolve ecrite: {output_path}")
    print(f"Variantes generees: {len(campagne.variants)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
