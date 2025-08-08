import pytest
import asyncio
import random
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os
from collections import defaultdict

# Import all classes from your main module
# Assuming your main file is named 'prisoners_dilemma.py'
from ipd import Move, Payoff, Prisoner, Random as RandomPrisoner, TitForTat, \
    WinStayLoseShift, LLM, Match, Tournament, classify_payoff


class TestRandomPrisoner:
    """Test the Random prisoner strategy."""

    @pytest.fixture
    def random_prisoner(self):
        """Create a Random prisoner for testing."""
        return RandomPrisoner("TestRandom")

    @pytest.fixture
    def test_payoff(self):
        """Create a standard payoff matrix for testing."""
        return Payoff(reward=3, punishment=1, temptation=5, sucker=0)

    def test_random_initialization(self, random_prisoner):
        """Test Random prisoner initialization."""
        assert random_prisoner.name == "TestRandom"
        assert isinstance(random_prisoner, Prisoner)

    @pytest.mark.asyncio
    async def test_random_moves_are_valid(self, random_prisoner, test_payoff):
        """Test that Random prisoner only makes valid moves."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            for _ in range(50):
                move = await random_prisoner._choose_one_move(
                    payoff=test_payoff,
                    termination_prob=0.0,
                    max_turns=100,
                    history=[],
                    opponent_history=[],
                    log_file=log_file
                )
                assert move in [Move.COOPERATE, Move.DEFECT]
        finally:
            os.unlink(log_file)

    @pytest.mark.asyncio
    async def test_random_distribution(self, random_prisoner, test_payoff):
        """Test that Random prisoner produces roughly equal distribution of moves."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            random.seed(123)  # For reproducible results
            cooperate_count = 0
            defect_count = 0

            for _ in range(1000):
                move = await random_prisoner._choose_one_move(
                    payoff=test_payoff,
                    termination_prob=0.0,
                    max_turns=100,
                    history=[],
                    opponent_history=[],
                    log_file=log_file
                )
                if move == Move.COOPERATE:
                    cooperate_count += 1
                else:
                    defect_count += 1

            # Should be roughly 50/50 (allow some variance)
            total = cooperate_count + defect_count
            cooperate_ratio = cooperate_count / total
            assert 0.4 <= cooperate_ratio <= 0.6, f"Cooperate ratio {cooperate_ratio} not near 0.5"

        finally:
            os.unlink(log_file)
