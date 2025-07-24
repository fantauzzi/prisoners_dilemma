import pytest
import random

from main import Move, Payoff, Prisoner, Tournament, instantiate_prisoners_CB, instantiate_prisoners_2_CB


def test_sanity():
    random.seed(31415)
    payoff: Payoff = Payoff(Reward=3, Punishment=0, Temptation=5, Sucker=1)
    tournament: Tournament = Tournament(instantiate_prisoners_CB, payoff=payoff, termination_prob=0.02, max_rounds=10)
    tournament.play_one_game()
    # assert tournament.terminated
    assert tournament.games_score == [10.0, 14.0]
    assert tournament.completed_round_number == 10


def test_tit_for_tat():
    random.seed(31415)
    payoff: Payoff = Payoff(Reward=3, Punishment=1, Temptation=5, Sucker=0)
    tournament: Tournament = Tournament(instantiate_prisoners_2_CB, payoff=payoff, termination_prob=0.0001,
                                        max_rounds=20)
    tournament.play_one_game()
    # assert tournament.terminated
    assert tournament.games_score == [60.0, 60.0]
    assert tournament.completed_round_number == 20
