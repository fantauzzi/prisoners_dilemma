# File: test/test_ipd.py

import asyncio
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import ipd  # adjust if your module name is different


@pytest.fixture(autouse=True)
def fixed_timestamp(monkeypatch):
    # Force a constant log directory name
    monkeypatch.setattr(ipd, 'get_time_stamp', lambda: 'test_timestamp')


@pytest.fixture
def config():
    # Load the provided YAML from the test/ directory
    cfg_path = Path(__file__).parent / "test_config_2.yaml"
    return OmegaConf.load(cfg_path)


def test_tit_for_tat_tournament_scores_and_history(config):
    # Instantiate and run the round-robin tournament
    tournament = ipd.Tournament(config)
    tournament.play_one_round_robin_tournament()

    # We expect exactly n_rematches matches for our single pair
    n_rematches = int(config.common.n_rematches)
    assert len(tournament.matches) == n_rematches

    reward = float(config.common.payoff.reward)
    max_rounds = int(config.common.max_rounds)

    # Check each match independently
    for match in tournament.matches:
        # Both histories should be full-length and all cooperative
        assert len(match.history[0]) == max_rounds
        assert len(match.history[1]) == max_rounds
        assert all(m == ipd.Move.COOPERATE for m in match.history[0])
        assert all(m == ipd.Move.COOPERATE for m in match.history[1])

        # Raw scores: reward * max_rounds for each side
        expected_score = reward * max_rounds
        assert match.scores == [expected_score, expected_score]

        # Normalised scores: score/rounds = reward
        assert pytest.approx(match.normalised_scores) == [reward, reward]

    # Now overall tournament aggregates
    # Each of our two TitForTat players appears in every match
    total_expected = reward * max_rounds * n_rematches
    assert tournament.scores['Tit4Tat_1'] == pytest.approx(total_expected)
    assert tournament.scores['Tit4Tat_2'] == pytest.approx(total_expected)

    # Normalised tournament scores sum the per-match normals
    norm_expected = reward * n_rematches
    assert tournament.normalised_scores['Tit4Tat_1'] == pytest.approx(norm_expected)
    assert tournament.normalised_scores['Tit4Tat_2'] == pytest.approx(norm_expected)
