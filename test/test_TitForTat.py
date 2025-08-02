import pytest
import asyncio
import random
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os
from collections import defaultdict

# Import all classes from your main module
# Assuming your main file is named 'prisoners_dilemma.py'
from main_async import Move, Payoff, Prisoner, Random as RandomPrisoner, TitForTat, \
    WinStayLoseShift, LLM, Match, Tournament, classify_payoff


class TestTitForTat:
    """Test the TitForTat prisoner strategy."""

    @pytest.fixture
    def tit_for_tat(self):
        """Create a TitForTat prisoner for testing."""
        return TitForTat("TestTitForTat")

    @pytest.fixture
    def test_payoff(self):
        """Create a standard payoff matrix for testing."""
        return Payoff(reward=3, punishment=1, temptation=5, sucker=0)

    def test_tit_for_tat_initialization(self, tit_for_tat):
        """Test TitForTat initialization."""
        assert tit_for_tat.name == "TestTitForTat"
        assert isinstance(tit_for_tat, Prisoner)

    @pytest.mark.asyncio
    async def test_tit_for_tat_first_move_cooperate(self, tit_for_tat, test_payoff):
        """Test that TitForTat's first move is always Cooperate."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            move = await tit_for_tat._choose_one_move(
                payoff=test_payoff,
                termination_prob=0.0,
                max_turns=100,
                history=[],
                opponent_history=[],
                log_file=log_file
            )
            assert move == Move.COOPERATE
        finally:
            os.unlink(log_file)

    @pytest.mark.asyncio
    async def test_tit_for_tat_copies_opponent_last_move(self, tit_for_tat, test_payoff):
        """Test that TitForTat copies the opponent's last move."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            # Test copying COOPERATE
            move = await tit_for_tat._choose_one_move(
                payoff=test_payoff,
                termination_prob=0.0,
                max_turns=100,
                history=[Move.COOPERATE],
                opponent_history=[Move.COOPERATE],
                log_file=log_file
            )
            assert move == Move.COOPERATE

            # Test copying DEFECT
            move = await tit_for_tat._choose_one_move(
                payoff=test_payoff,
                termination_prob=0.0,
                max_turns=100,
                history=[Move.COOPERATE, Move.DEFECT],
                opponent_history=[Move.COOPERATE, Move.DEFECT],
                log_file=log_file
            )
            assert move == Move.DEFECT
        finally:
            os.unlink(log_file)

    @pytest.mark.asyncio
    async def test_tit_for_tat_100_random_turns(self, tit_for_tat, test_payoff):
        """
        Test TitForTat against 100 random moves.
        Verifies first move is Cooperate and subsequent moves copy opponent's previous move.
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            # Set seed for reproducible random moves
            random.seed(42)

            history = []
            opponent_history = []

            for turn in range(100):
                # Get TitForTat's move
                tft_move = await tit_for_tat._choose_one_move(
                    payoff=test_payoff,
                    termination_prob=0.0,
                    max_turns=100,
                    history=history.copy(),
                    opponent_history=opponent_history.copy(),
                    log_file=log_file
                )

                # Check TitForTat's behavior
                if turn == 0:
                    # First move should be Cooperate
                    assert tft_move == Move.COOPERATE, f"First move should be COOPERATE, got {tft_move}"
                else:
                    # Should copy opponent's previous move
                    expected_move = opponent_history[-1]
                    assert tft_move == expected_move, f"Turn {turn}: Expected {expected_move}, got {tft_move}"

                # Generate random opponent move
                opponent_move = random.choice([Move.COOPERATE, Move.DEFECT])

                # Update histories
                history.append(tft_move)
                opponent_history.append(opponent_move)

        finally:
            os.unlink(log_file)
