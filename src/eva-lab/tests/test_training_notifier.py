"""Tests du digest Telegram des entrainements."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from eva_lab.training_notifier import build_training_digest_message, send_training_digest


class TrainingNotifierDigestTests(unittest.TestCase):
    """Verifie le format du digest Telegram training."""

    def test_build_training_digest_message_exposes_run_metrics_and_champion_status(self) -> None:
        """Assemble un digest utile avec run actif, metriques et champion."""
        run_status = {
            "run_id": "nightly_20260424_042458",
            "status": "running",
            "strategy": "research",
            "reason": "no_deployable_champion",
            "resume_source": "explicit_resume",
            "replay_cache_entries": 128,
            "current_step": {
                "name": "muzero_scalp",
                "phase": "optimisation",
                "horizon": "scalp",
                "symbol": "US30.cash",
                "symbol_index": 2,
                "symbol_total": 7,
                "part_index": 11,
                "part_total": 28,
                "episode_step_current": 250,
                "episode_step_total": 300,
                "updated_at": "2026-04-24T06:45:00",
            },
            "latest_metrics": {
                "loss_pol": 5.32,
                "loss_pol_per_head": 0.89,
                "loss_total": 16.44,
                "policy_entropy": 0.44,
                "policy_top1_share": 0.86,
                "root_mask_rate": 0.11,
                "root_mask_rate_trend": 0.03,
                "loss_pol_trend": -0.12,
                "seed_viability_status": "monitoring",
                "seed_viability_reason": "within_seed_window",
                "seed_stage": "offensive_bootstrap",
                "recommended_seed_for_v66": "C:/tmp/muzero_scalp_ckpt_23000.pkl",
                "post_veto_to_hold_rate": 0.01,
                "soft_entry_penalty_rate": 0.18,
                "soft_entry_bonus_rate": 0.32,
                "soft_penalty_to_bonus_ratio": 0.56,
                "soft_penalty_net": -0.14,
                "hold_drag_score": 0.22,
                "close_quality_score": 0.58,
                "split_efficiency": 0.44,
                "split_runner_capture_rate": 0.27,
                "split_runner_profitable_count": 3,
                "split_runner_failed_count": 2,
                "split_zone_capture_rate": 0.18,
                "split_tp_zone_opportunity_count": 7,
                "split_monetization_capture_rate": 0.16,
                "split_monetization_window_count": 9,
                "split_trade_value_delta": 0.84,
                "split_improved_total_trade_count": 2,
                "pyramid_efficiency": 0.51,
                "pyramid_entry_quality_score": 0.49,
                "pyramid_exit_capture_rate": 0.28,
                "pyramid_add_capture_rate": 0.12,
                "pyramid_monetization_capture_rate": 0.14,
                "pyramid_monetization_window_count": 6,
                "pyramid_total_trade_improvement_pct": 0.66,
                "pyramid_failed_to_improve_count": 1,
                "pyramid_missed_add_count": 2,
                "slbe_capture_rate": 0.37,
                "tp_like_missed_count": 3,
                "hard_stop_exit_count": 2,
                "soft_tp_hit_count": 6,
                "full_tp_hit_count": 3,
                "time_stop_trigger_count": 4,
                "runner_extension_count": 2,
                "runner_managed_exit_count": 2,
                "runner_exit_profitable_count": 1,
                "runner_forced_stop_count": 1,
                "runner_extension_capture_rate": 0.09,
                "runner_profit_hold_capture_rate": 0.21,
                "runner_profit_hold_window_count": 5,
                "runner_missed_extension_count": 3,
                "runner_retained_profit_pct": 0.52,
                "runner_giveback_pct": 0.18,
                "runner_giveback_ratio": 0.26,
                "profit_peak_giveback_ratio": 0.31,
                "liquidity_trap_share": 0.10,
                "bad_runner_share": 0.06,
                "bad_pyramid_share": 0.04,
                "root_mask_ema200_share": 0.15,
                "root_mask_vwap_share": 0.45,
                "root_mask_adx_share": 0.20,
                "root_mask_directional_share": 0.20,
                "balanced_episode_rate": 0.91,
                "long_entry_share": 0.48,
                "short_entry_share": 0.52,
            },
            "policy_precheck": {
                "status": "screen_only",
                "reason": "eligible_screen",
                "trends": {
                    "loss_pol_trend": -0.12,
                    "root_mask_rate_trend": 0.03,
                    "split_runner_capture_trend": 0.01,
                    "pyramid_exit_capture_trend": 0.00,
                },
            },
            "family_probe_status": {
                "reason": "family_probe_passed",
                "ready_families": 3,
                "required_ready_families": 3,
                "positive_families": 2,
                "required_positive_families": 2,
            },
        }
        horizon_statuses = {
            "scalp": {
                "selection": "blocked_champion",
                "live_champion_id": None,
                "candidate_id": "gen_scalp_ckpt10000",
                "gate_reason": "profit_factor_insuffisant",
                "directional_metrics": {"directional_bias": "balanced"},
                "candidate_metrics": {
                    "profit_factor": 1.42,
                    "return_pct": 0.15,
                    "win_rate": 33.2,
                    "metrics_by_position_mechanics": {
                        "hold_drag_score": 0.41,
                        "close_quality_score": 0.49,
                        "split_efficiency": 0.45,
                        "split_runner_capture_rate": 0.22,
                        "split_early_count": 1,
                        "split_decorative_count": 2,
                        "pyramid_efficiency": 0.52,
                        "pyramid_entry_quality_score": 0.40,
                        "pyramid_exit_capture_rate": 0.28,
                        "slbe_capture_rate": 0.38,
                    },
                },
            }
        }

        message = build_training_digest_message(
            run_status=run_status,
            horizon_statuses=horizon_statuses,
            horizons=["scalp"],
        )

        self.assertIn("POINT ENTRAINEMENT", message)
        self.assertIn("- id: nightly-20260424-042458", message)
        self.assertIn("- etape: muzero scalp | phase: optimisation | horizon: scalp", message)
        self.assertIn("- policy: total=5.32 | par_tete=0.89 | top1=86.00% | entropy=0.44", message)
        self.assertIn("- defensif: root_mask=11.00% | close_q=0.58 | slbe=0.37 | hold_drag=0.22", message)
        self.assertIn("- offensif: split_cap=0.27 | runner_win=5.00 | pyramid_cap=0.28 | peak_giveback=0.31", message)
        self.assertIn("- fenetres: split=9.00 | runner=5.00 | pyramid=6.00", message)
        self.assertIn("- seed: etage=offensive bootstrap | statut=monitoring | raison=within seed window", message)
        self.assertIn("- seed_reco: muzero-scalp-ckpt-23000", message)
        self.assertIn("PRECHECK POLICY", message)
        self.assertIn("- statut: screen only | raison: eligible screen", message)
        self.assertIn("FAMILY PROBES", message)
        self.assertIn("- statut: family probe passed | pretes: 3.00/3.00 | positives: 2.00/2.00", message)
        self.assertIn("MUZERO SCALP", message)
        self.assertIn("- selection: blocked champion | live: aucun", message)
        self.assertIn("- candidat: gen-scalp-ckpt10000 | gate: profit factor insuffisant", message)
        self.assertIn("- perf: PF=1.42 | Ret=0.15% | WR=33.20%", message)
        self.assertIn("- meca: close_q=0.49 | split_runner=0.22 | pyramid_exit=0.28 | slbe=0.38", message)

    def test_send_training_digest_ignores_unchanged_snapshot(self) -> None:
        """N'envoie pas un second digest si rien d'important n'a change."""
        run_status = {
            "run_id": "nightly_20260502_153431",
            "status": "running",
            "strategy": "research",
            "reason": "champion_stale",
            "resume_source": "explicit_resume",
            "current_step": {
                "name": "muzero_scalp",
                "phase": "collecte",
                "horizon": "scalp",
                "symbol": "XAUUSD",
                "symbol_index": 1,
                "symbol_total": 6,
                "part_index": 5,
                "part_total": 10,
                "episode_step_current": 175,
                "episode_step_total": 300,
            },
            "latest_metrics": {
                "seed_viability_status": "monitoring",
                "seed_viability_reason": "before_seed_window",
            },
        }
        horizon_statuses = {"scalp": {"selection": "champion", "candidate_id": "seed-test"}}

        with TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "training_digest_state.json"
            env = {
                "TELEGRAM_NOTIFY_TRAINING": "1",
                "TELEGRAM_NOTIFY_TRAINING_DIGEST": "1",
                "TELEGRAM_TRAINING_DIGEST_ONLY_ON_CHANGE": "1",
                "TELEGRAM_TRAINING_DIGEST_FORCE_AFTER_MINUTES": "180",
                "TRAINING_DIGEST_STATE_PATH": str(state_path),
            }
            with patch.dict("os.environ", env, clear=False):
                with patch("eva_lab.training_notifier.TelegramClient") as client_cls:
                    self.assertTrue(
                        send_training_digest(
                            run_status=run_status,
                            horizon_statuses=horizon_statuses,
                            horizons=["scalp"],
                        )
                    )
                    self.assertFalse(
                        send_training_digest(
                            run_status=run_status,
                            horizon_statuses=horizon_statuses,
                            horizons=["scalp"],
                        )
                    )
            self.assertEqual(client_cls.return_value.send_sync.call_count, 1)

    def test_send_training_digest_forces_refresh_after_timeout(self) -> None:
        """Renvoie un digest identique apres le delai de securite configure."""
        run_status = {
            "run_id": "nightly_20260502_153431",
            "status": "running",
            "strategy": "research",
            "reason": "champion_stale",
            "resume_source": "explicit_resume",
            "current_step": {
                "name": "muzero_scalp",
                "phase": "collecte",
                "horizon": "scalp",
                "symbol": "XAUUSD",
                "symbol_index": 1,
                "symbol_total": 6,
                "part_index": 5,
                "part_total": 10,
            },
            "latest_metrics": {
                "seed_viability_status": "monitoring",
                "seed_viability_reason": "before_seed_window",
            },
        }

        with TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "training_digest_state.json"
            env = {
                "TELEGRAM_NOTIFY_TRAINING": "1",
                "TELEGRAM_NOTIFY_TRAINING_DIGEST": "1",
                "TELEGRAM_TRAINING_DIGEST_ONLY_ON_CHANGE": "1",
                "TELEGRAM_TRAINING_DIGEST_FORCE_AFTER_MINUTES": "1",
                "TRAINING_DIGEST_STATE_PATH": str(state_path),
            }
            with patch.dict("os.environ", env, clear=False):
                with patch("eva_lab.training_notifier.TelegramClient") as client_cls:
                    self.assertTrue(
                        send_training_digest(
                            run_status=run_status,
                            horizon_statuses={"scalp": {}},
                            horizons=["scalp"],
                        )
                    )
                    payload = json.loads(state_path.read_text(encoding="utf-8"))
                    payload["last_sent_at"] = "2026-05-02T10:00:00"
                    state_path.write_text(json.dumps(payload), encoding="utf-8")
                    self.assertTrue(
                        send_training_digest(
                            run_status=run_status,
                            horizon_statuses={"scalp": {}},
                            horizons=["scalp"],
                        )
                    )
            self.assertEqual(client_cls.return_value.send_sync.call_count, 2)


if __name__ == "__main__":
    unittest.main()
