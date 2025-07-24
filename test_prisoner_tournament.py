import pytest
import random

from main import Move, Payoff, Drunk, Tournament, instantiate_4_prisoners_CB


def test_sanity():
    random.seed(31415)
    payoff: Payoff = Payoff(reward=3, punishment=1, temptation=5, sucker=0)
    tournament: Tournament = Tournament(instantiate_4_prisoners_CB,
                                        payoff=payoff,
                                        termination_prob=0.004047,
                                        max_rounds=200)
    tournament.play_one_round_robin_game()

    assert tournament.games_score == {('Drunk_1', 'Drunk_2'): (418, 478), ('Tit4Tat_1', 'Drunk_1'): (434, 434),
                                      ('Tit4Tat_1', 'Drunk_2'): (456, 461), ('Tit4Tat_1', 'Tit4Tat_2'): (153, 153),
                                      ('Tit4Tat_2', 'Drunk_1'): (438, 443), ('Tit4Tat_2', 'Drunk_2'): (451, 456)}

    assert len(tournament.history[('Tit4Tat_1', 'Tit4Tat_2')]) == 51
    assert len(tournament.history[('Tit4Tat_1', 'Drunk_1')]) == 200
    assert tournament.prisoners_score == {'Tit4Tat_1': 1.2416666666666665, 'Tit4Tat_2': 1.2408333333333332,
                                          'Drunk_1': 1.0791666666666666, 'Drunk_2': 1.1624999999999999}
