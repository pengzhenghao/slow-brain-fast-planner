from __future__ import annotations

import numpy as np

from slow_brain_fast_planner.control.delayed_vlm_score_fusion import (
    FusionConfig,
    Limits,
    PlannerScoreModel,
    SimLoopConfig,
    VlmDelayModel,
    load_static_candidate_set,
    make_task,
    run_toy_simulation,
    run_toy_simulation_with_trace,
)


def test_chunk_stitch_trace_matches_non_trace_result() -> None:
    candidates = load_static_candidate_set(
        "assets/trajectory_selection_static_candidates/takeover_kmeans_medoids/"
        "static_candidates_k12.json",
        prepend_origin=True,
    )
    task = make_task("left_turn", v_ref=1.0, radius_m=2.5)
    sim_cfg = SimLoopConfig(
        dt_control=0.1,
        dt_plan=0.2,
        horizon_s=4.0,
        limits=Limits(),
    )
    score_model = PlannerScoreModel(noise_std=0.15, score_scale=1.0, epsilon=0.0)
    vlm_cfg = VlmDelayModel(delay_s=1.0)
    fusion_cfg = FusionConfig()

    result = run_toy_simulation(
        candidates_body=candidates,
        task=task,
        controller="pure_pursuit",
        policy="chunk_stitch",
        sim_cfg=sim_cfg,
        score_model=score_model,
        vlm_cfg=vlm_cfg,
        fusion_cfg=fusion_cfg,
        seed=0,
    )
    traced_result, trace = run_toy_simulation_with_trace(
        candidates_body=candidates,
        task=task,
        controller="pure_pursuit",
        policy="chunk_stitch",
        sim_cfg=sim_cfg,
        score_model=score_model,
        vlm_cfg=vlm_cfg,
        fusion_cfg=fusion_cfg,
        seed=0,
    )

    assert trace.actual_world.shape[0] > 1
    assert np.isclose(traced_result.mean_cte_to_ideal_m, result.mean_cte_to_ideal_m)
    assert traced_result.chosen_idx_hist == result.chosen_idx_hist
