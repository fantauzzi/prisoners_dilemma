import pytest
import random

from main import Move, Payoff, Prisoner, Tournament, instantiate_prisoners_CB, instantiate_prisoners_2_CB

def test_sanity():
    random.seed(31415)
    payoff: Payoff = Payoff(Reward=3, Punishment=0, Temptation=5, Sucker=1)
    tournament: Tournament = Tournament(instantiate_prisoners_CB, payoff=payoff, termination_prob=0.02, max_rounds=10)
    tournament.play_one_game()
    assert tournament.terminated
    assert tournament.scores == [10.0, 14.0]
    assert tournament.completed_round_number == 10

def test_tit_for_tat():
    random.seed(31415)
    payoff: Payoff = Payoff(Reward=3, Punishment=1, Temptation=5, Sucker=0)
    tournament: Tournament = Tournament(instantiate_prisoners_2_CB, payoff=payoff, termination_prob=0.0001,
                                        max_rounds=20)
    tournament.play_one_game()
    assert tournament.terminated
    assert tournament.scores == [60.0, 60.0]
    assert tournament.completed_round_number == 20

def test_move_enum_values_and_uniqueness():
    assert Move.COOPERATE != Move.DEFECT
    assert Move.COOPERATE.name == "COOPERATE"
    assert Move.DEFECT.name == "DEFECT"
    # auto() should assign integer values
    assert isinstance(Move.COOPERATE.value, int)
    assert isinstance(Move.DEFECT.value, int)
    assert Move.COOPERATE.value < Move.DEFECT.value


def test_payoff_namedtuple_fields():
    payoff = Payoff(Reward=3.0, Punishment=0.0, Temptation=5.0, Sucker=1.0)
    assert payoff.Reward == 3.0
    assert payoff.Punishment == 0.0
    assert payoff.Temptation == 5.0
    assert payoff.Sucker == 1.0
    # NamedTuple should be iterable or indexable
    assert payoff[0] == 3.0
    assert payoff[3] == 1.0


def test_instantiate_prisoners_cb_returns_two_prisoners():
    p1, p2 = instantiate_prisoners_CB()
    assert isinstance(p1, Prisoner)
    assert isinstance(p2, Prisoner)
    assert p1.name == "Gino"
    assert p2.name == "Pilotino"


def test_prisoner_logging_and_choose_one_move(monkeypatch):
    prisoner = Prisoner("Alice")
    # Stub random.choice to a known move
    monkeypatch.setattr(random, 'choice', lambda seq: Move.DEFECT)
    # Test when prev_opponent_move is None
    move = prisoner.choose_one_move(
        payoff=Payoff(3, 0, 5, 1), termination_prob=0.0, prev_opponent_move=None
    )
    assert move == Move.DEFECT
    assert prisoner.my_moves == [Move.DEFECT]
    assert prisoner.opponent_moves == []

    # Test when prev_opponent_move is provided
    move2 = prisoner.choose_one_move(
        payoff=Payoff(3, 0, 5, 1), termination_prob=0.0, prev_opponent_move=Move.COOPERATE
    )
    assert move2 == Move.DEFECT
    # Now both lists should have grown
    assert prisoner.opponent_moves == [Move.COOPERATE]
    assert prisoner.my_moves[-1] == Move.DEFECT
    # The prisoner knows its moves up to its latest one, and the opponent moves up to the previous one
    assert len(prisoner.my_moves) == len(prisoner.opponent_moves) + 1


def test_tournament_init_invalid_termination_prob():
    with pytest.raises(AssertionError):
        Tournament(instantiate_prisoners_CB, payoff=Payoff(1, 1, 1, 1), termination_prob=-0.1, max_rounds=5)
    with pytest.raises(AssertionError):
        Tournament(instantiate_prisoners_CB, payoff=Payoff(1, 1, 1, 1), termination_prob=1.1, max_rounds=5)


def test_tournament_match_updates_scores_and_termination(monkeypatch):
    # Prepare a tournament with deterministic behavior
    payoff = Payoff(Reward=3, Punishment=0, Temptation=5, Sucker=1)
    # termination_prob = 0: should never terminate
    t = Tournament(instantiate_prisoners_CB, payoff, termination_prob=0.0, max_rounds=10)

    # Stub random.choice to pick specific sequence of moves
    choices = [Move.COOPERATE, Move.DEFECT]
    monkeypatch.setattr(random, 'choice', lambda seq: choices.pop(0))
    # Stub random.random to always be > termination_prob
    monkeypatch.setattr(random, 'random', lambda: 0.5)

    p1, p2 = t.prisoners
    m1, m2 = t.match(p1, p2, prev_move_1=None, prev_move_2=None)
    assert m1 == Move.COOPERATE
    assert m2 == Move.DEFECT
    # Payoff for (COOPERATE, DEFECT) is (Sucker, Temptation) = (1, 5)
    assert t.scores == [1.0, 5.0]
    assert not t.terminated


def test_play_one_game_runs_until_termination(monkeypatch):
    payoff = Payoff(3, 0, 5, 1)
    # termination_prob = 1 -> should terminate after first round
    t = Tournament(instantiate_prisoners_CB, payoff, termination_prob=1.0, max_rounds=10)
    # always cooperate
    monkeypatch.setattr(random, 'choice', lambda seq: Move.COOPERATE)
    monkeypatch.setattr(random, 'random', lambda: 0.0)

    t.play_one_game()
    assert t.completed_round_number == 1
    assert t.terminated
    # scores: one round of mutual cooperation
    assert t.scores == [payoff.Reward, payoff.Reward]


def test_play_one_game_runs_full_max_rounds_no_termination(monkeypatch):
    payoff = Payoff(3, 0, 5, 1)
    # termination_prob = 0 -> should run until max_rounds
    max_rounds = 5
    t = Tournament(instantiate_prisoners_CB, payoff, termination_prob=0.0, max_rounds=max_rounds)
    # alternate moves to vary scores
    seq = [Move.COOPERATE, Move.DEFECT] * max_rounds
    monkeypatch.setattr(random, 'choice', lambda seq_arg: seq.pop(0))
    monkeypatch.setattr(random, 'random', lambda: 1.0)

    t.play_one_game()
    assert t.completed_round_number == max_rounds
    assert t.terminated
    # manual score calculation: rounds alternate (C, D) then (C, D)... for max_rounds
    # each (C, D) yields (Sucker, Temptation) = (1,5)
    expected_rounds = max_rounds
    # Since we always pick first two entries for each round, all rounds are (C, D)
    assert t.scores == [1.0 * expected_rounds, 5.0 * expected_rounds]
