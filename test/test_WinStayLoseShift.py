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


class TestWinStayLoseShift:
    """Test the WinStayLoseShift prisoner strategy."""

    @pytest.fixture
    def wsls_prisoner(self):
        """Create a WinStayLoseShift prisoner for testing."""
        return WinStayLoseShift("TestWSLS")

    @pytest.fixture
    def test_payoff(self):
        """Create a standard payoff matrix for testing."""
        return Payoff(reward=3, punishment=1, temptation=5, sucker=0)

    def test_wsls_initialization(self, wsls_prisoner):
        """Test WinStayLoseShift initialization."""
        assert wsls_prisoner.name == "TestWSLS"
        assert isinstance(wsls_prisoner, Prisoner)

    @pytest.mark.asyncio
    async def test_wsls_first_move_cooperate(self, wsls_prisoner, test_payoff):
        """Test that WSLS first move is Cooperate."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            move = await wsls_prisoner._choose_one_move(
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
    async def test_wsls_win_stay_behavior(self, wsls_prisoner, test_payoff):
        """Test that WSLS stays with winning moves (reward and temptation)."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            # Test staying after reward (C,C)
            move = await wsls_prisoner._choose_one_move(
                payoff=test_payoff,
                termination_prob=0.0,
                max_turns=100,
                history=[Move.COOPERATE],
                opponent_history=[Move.COOPERATE],
                log_file=log_file
            )
            assert move == Move.COOPERATE

            # Test staying after temptation (D,C)
            move = await wsls_prisoner._choose_one_move(
                payoff=test_payoff,
                termination_prob=0.0,
                max_turns=100,
                history=[Move.DEFECT],
                opponent_history=[Move.COOPERATE],
                log_file=log_file
            )
            assert move == Move.DEFECT
        finally:
            os.unlink(log_file)

    @pytest.mark.asyncio
    async def test_wsls_lose_shift_behavior(self, wsls_prisoner, test_payoff):
        """Test that WSLS shifts after losing moves (sucker and punishment)."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            # Test shifting after sucker (C,D)
            move = await wsls_prisoner._choose_one_move(
                payoff=test_payoff,
                termination_prob=0.0,
                max_turns=100,
                history=[Move.COOPERATE],
                opponent_history=[Move.DEFECT],
                log_file=log_file
            )
            assert move == Move.DEFECT

            # Test shifting after punishment (D,D)
            move = await wsls_prisoner._choose_one_move(
                payoff=test_payoff,
                termination_prob=0.0,
                max_turns=100,
                history=[Move.DEFECT],
                opponent_history=[Move.DEFECT],
                log_file=log_file
            )
            assert move == Move.COOPERATE
        finally:
            os.unlink(log_file)
