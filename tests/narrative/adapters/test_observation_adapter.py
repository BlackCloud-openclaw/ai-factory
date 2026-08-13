import pytest

from src.narrative.adapters.observation_adapter import ObservationAdapter


class TestObservationAdapter:
    def test_adapts_dimensions(self):
        data = {
            "dimensions": {
                "dialogue_ratio": 0.15,
                "transition_score": 0.40,
                "emotion_score": 0.30,
            }
        }
        adapter = ObservationAdapter(data)

        assert adapter.get_dimension("dialogue_ratio") == 0.15
        assert adapter.get_dimension("transition_score") == 0.40
        assert adapter.get_all_dimensions() == ("dialogue_ratio", "transition_score", "emotion_score")

    def test_handles_missing_dimension(self):
        data = {"dimensions": {}}
        adapter = ObservationAdapter(data)

        assert adapter.get_dimension("dialogue_ratio") is None
        assert adapter.get_all_dimensions() == ()

    def test_handles_flat_data(self):
        data = {"dialogue_ratio": 0.15, "transition_score": 0.40}
        adapter = ObservationAdapter(data)

        assert adapter.get_dimension("dialogue_ratio") == 0.15
        assert adapter.get_all_dimensions() == ("dialogue_ratio", "transition_score")