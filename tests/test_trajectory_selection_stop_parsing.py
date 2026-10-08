from slow_brain_fast_planner.benchmarks.vqa_trajectory import parse_vqa_trajectory_action


def test_trajectory_selection_parsing_stop_selected_index_null() -> None:
    obj, note = parse_vqa_trajectory_action(
        '{"action":"stop","selected_index":null}', num_candidates=6
    )
    assert obj is not None
    assert obj["action"] == "stop"
    # stop should not require selected_index
    assert note is None or isinstance(note, str)


def test_trajectory_selection_parsing_legacy_ask_for_help_normalizes_to_stop() -> None:
    obj, note = parse_vqa_trajectory_action(
        '{"action":"ask_for_help","selected_index":null}', num_candidates=6
    )
    assert obj is not None
    assert obj["action"] == "stop"
    assert isinstance(note, str)
    assert "stop" in note
