import pytest
import asyncio
import random
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

# Import all classes from your main module
# Assuming your main file is named 'prisoners_dilemma.py'
from main_async import Move, Payoff, Prisoner, Random as RandomPrisoner, TitForTat, \
    WinStayLoseShift, OpenAI, Match, Tournament, classify_payoff, get_time_stamp


class TestMatch:
    """Test the Match class."""

    @pytest.fixture
    def prisoners(self):
        """Create two test prisoners."""
        log_dir = f'logs/{get_time_stamp()}'
        Path(log_dir).mkdir(exist_ok=True)
        return TitForTat("Player1"), RandomPrisoner("Player2"), log_dir

    def test_match_initialization(self, prisoners):
        """Test Match initialization."""

        p1, p2, log_dir = prisoners
        match = Match(p1, p2, log_dir=log_dir)

        # Should sort prisoners alphabetically by name
        assert match.prisoner_1.name == "Player1"
        assert match.prisoner_2.name == "Player2"
        assert match.match_counter > 0
        assert len(match.log_files) == 2
        assert match.history == ([], [])
        assert match.scores == [0, 0]

    def test_match_counter_increments(self, prisoners):
        """Test that match counter increments."""
        p1, p2, log_dir = prisoners
        initial_counter = Match.counter

        match1 = Match(p1, p2, log_dir=log_dir)
        match2 = Match(p2, p1, log_dir=log_dir)

        assert match2.match_counter == match1.match_counter + 1
        assert Match.counter == initial_counter + 2


class TestTournament:
    """Test the Tournament class."""

    @pytest.fixture
    def simple_tournament(self):
        """Create a simple tournament for testing."""

        test_config = OmegaConf.load('test_config.yaml')

        payoff = Payoff(reward=3, punishment=1, temptation=5, sucker=0)
        return Tournament(payoff, test_config)

    def test_tournament_initialization(self, simple_tournament):
        """Test Tournament initialization."""
        assert len(simple_tournament.prisoners) == 2
        assert simple_tournament.termination_prob == 0.1
        assert simple_tournament.max_rounds == 10
        assert len(simple_tournament.matches) == 1  # C(2,2) = 1
        assert isinstance(simple_tournament.scores, defaultdict)

    def test_moves_to_rewards_mapping(self, simple_tournament):
        """Test that the moves to rewards mapping is correct."""
        expected = {
            (Move.COOPERATE, Move.COOPERATE): (3, 3),
            (Move.DEFECT, Move.DEFECT): (1, 1),
            (Move.COOPERATE, Move.DEFECT): (0, 5),
            (Move.DEFECT, Move.COOPERATE): (5, 0)
        }
        assert simple_tournament.moves_to_rewards == expected


class TestPrisonerErrorHandling:
    """Test error handling in the Prisoner base class."""

    @pytest.fixture
    def failing_prisoner(self):
        """Create a prisoner that always fails."""

        class FailingPrisoner(Prisoner):
            async def _choose_one_move(self, payoff, termination_prob, max_turns,
                                       history, opponent_history, log_file):
                raise ValueError("Test error")

        return FailingPrisoner("Failer")

    @pytest.mark.asyncio
    async def test_prisoner_error_handling(self, failing_prisoner):
        """Test that prisoner error handling works correctly."""
        payoff = Payoff(reward=3, punishment=1, temptation=5, sucker=0)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            result = await failing_prisoner.choose_one_move(
                payoff=payoff,
                termination_prob=0.0,
                max_turns=100,
                history=[],
                opponent_history=[],
                log_file=log_file
            )
            assert result is None  # Should return None after max attempts
        finally:
            os.unlink(log_file)
