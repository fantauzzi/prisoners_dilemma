import pytest
import random

from main import Move, Payoff, Drunk, Tournament, instantiate_4_prisoners_CB, instantiate_6_prisoners_CB


def test_sanity():
    random.seed(31415)
    payoff: Payoff = Payoff(reward=3, punishment=1, temptation=5, sucker=0)
    tournament: Tournament = Tournament(instantiate_4_prisoners_CB,
                                        payoff=payoff,
                                        termination_prob=0.004047,
                                        max_rounds=200)
    tournament.play_one_round_robin_game(seed=31415)

    assert tournament.games_score == {('Drunk_1', 'Drunk_2'): (418, 478), ('Tit4Tat_1', 'Drunk_1'): (434, 434),
                                      ('Tit4Tat_1', 'Drunk_2'): (456, 461), ('Tit4Tat_1', 'Tit4Tat_2'): (153, 153),
                                      ('Tit4Tat_2', 'Drunk_1'): (438, 443), ('Tit4Tat_2', 'Drunk_2'): (451, 456)}

    assert len(tournament.history[('Tit4Tat_1', 'Tit4Tat_2')]) == 51
    assert len(tournament.history[('Tit4Tat_1', 'Drunk_1')]) == 200
    assert tournament.prisoners_score == {'Tit4Tat_1': 1.2416666666666665, 'Tit4Tat_2': 1.2408333333333332,
                                          'Drunk_1': 1.0791666666666666, 'Drunk_2': 1.1624999999999999}


def test_sanity2():
    random.seed(31415)
    payoff: Payoff = Payoff(reward=3, punishment=1, temptation=5, sucker=0)
    tournament: Tournament = Tournament(instantiate_6_prisoners_CB,
                                        payoff=payoff,
                                        termination_prob=0.004047,
                                        max_rounds=200)
    tournament.play_one_round_robin_game(seed=31415)

    assert tournament.games_score == {('Drunk_1', 'Drunk_2'): (449, 474), ('Drunk_1', 'WSLS_1'): (458, 343),
                                      ('Drunk_1', 'WSLS_2'): (426, 291), ('Drunk_2', 'WSLS_1'): (12, 7),
                                      ('Drunk_2', 'WSLS_2'): (568, 373), ('Tit4Tat_1', 'Drunk_1'): (434, 434),
                                      ('Tit4Tat_1', 'Drunk_2'): (456, 461), ('Tit4Tat_1', 'Tit4Tat_2'): (153, 153),
                                      ('Tit4Tat_1', 'WSLS_1'): (600, 600), ('Tit4Tat_1', 'WSLS_2'): (492, 492),
                                      ('Tit4Tat_2', 'Drunk_1'): (444, 449), ('Tit4Tat_2', 'Drunk_2'): (408, 413),
                                      ('Tit4Tat_2', 'WSLS_1'): (600, 600), ('Tit4Tat_2', 'WSLS_2'): (600, 600),
                                      ('WSLS_1', 'WSLS_2'): (600, 600)}

    rounds_count = [len(tournament.history[k]) for k in tournament.games_score.keys()]
    assert rounds_count == [51, 200, 200, 200, 164, 200, 178, 200, 200, 200, 164, 150, 4, 200, 200]

    assert tournament.prisoners_score == {'Tit4Tat_1': 0.8966666666666666, 'Tit4Tat_2': 0.9008089887640449,
                                          'Drunk_1': 0.8195121951219513, 'Drunk_2': 0.855681647940075,
                                          'WSLS_1': 0.8560975609756097, 'WSLS_2': 0.8536666666666667}
