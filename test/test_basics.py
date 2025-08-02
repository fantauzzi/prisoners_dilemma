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


class TestMove:
    """Test the Move enum."""

    def test_move_enum_values(self):
        """Test that Move enum has correct values."""
        assert Move.COOPERATE is not None
        assert Move.DEFECT is not None
        assert Move.COOPERATE != Move.DEFECT


class TestPayoff:
    """Test the Payoff NamedTuple."""

    def test_payoff_creation(self):
        """Test creating a Payoff instance."""
        payoff = Payoff(reward=3, punishment=1, temptation=5, sucker=0)
        assert payoff.reward == 3
        assert payoff.punishment == 1
        assert payoff.temptation == 5
        assert payoff.sucker == 0


class TestClassifyPayoff:
    """Test the classify_payoff function."""

    def test_classify_payoff_all_combinations(self):
        """Test all possible move combinations."""
        assert classify_payoff(Move.COOPERATE, Move.COOPERATE) == 'reward'
        assert classify_payoff(Move.COOPERATE, Move.DEFECT) == 'sucker'
        assert classify_payoff(Move.DEFECT, Move.COOPERATE) == 'temptation'
        assert classify_payoff(Move.DEFECT, Move.DEFECT) == 'punishment'
