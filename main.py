from __future__ import annotations
from collections.abc import Callable
from collections import defaultdict
from itertools import combinations
from enum import Enum, auto
from typing import NamedTuple
import random


class Move(Enum):
    COOPERATE = auto()
    DEFECT = auto()


class Payoff(NamedTuple):
    reward: float
    punishment: float
    temptation: float
    sucker: float


class Prisoner:
    def __init__(self, name: str) -> None:
        self.name: str = name

    def choose_one_move(self,
                        payoff: Payoff,
                        termination_prob: float,
                        history: list[Move],
                        opponent_history: list[Move]) -> Move:
        my_move = random.choice((Move.COOPERATE, Move.DEFECT))

        return my_move


class TitForTat(Prisoner):
    def choose_one_move(self,
                        payoff: Payoff,
                        termination_prob: float,
                        history: list[Move],
                        opponent_history: list[Move]) -> Move:
        my_move = Move.COOPERATE if len(opponent_history) == 0 else opponent_history[-1]

        return my_move


class Tournament:
    def __init__(self,
                 instantiate_prisoners_CB: Callable[[], tuple[Prisoner, ...]],
                 payoff: Payoff,
                 termination_prob: float,
                 max_rounds: int) -> None:
        assert 0 <= termination_prob <= 1
        self.prisoners = instantiate_prisoners_CB()
        self.payoff = payoff
        self.termination_prob = termination_prob
        self.max_rounds = max_rounds
        self.games_score: dict[tuple[str, str], tuple[float, float]] = {}
        self.prisoners_score = defaultdict(float)
        self.history = defaultdict(list)
        self.moves_to_rewards: dict[tuple[Move, Move], tuple[float, float]] = {
            (Move.COOPERATE, Move.COOPERATE): (self.payoff.reward, self.payoff.reward),
            (Move.DEFECT, Move.DEFECT): (self.payoff.punishment, self.payoff.punishment),
            (Move.COOPERATE, Move.DEFECT): (self.payoff.sucker, self.payoff.temptation),
            (Move.DEFECT, Move.COOPERATE): (self.payoff.temptation, self.payoff.sucker)}

    def play_one_move(self, prisoner1: Prisoner, prisoner2: Prisoner) -> tuple[tuple[Move, Move], tuple[float, float]]:

        prisoner1_history = [move for (move, _) in self.history[(prisoner1.name, prisoner2.name)]]
        prisoner2_history = [move for (_, move) in self.history[(prisoner1.name, prisoner2.name)]]
        move_1 = prisoner1.choose_one_move(self.payoff,
                                           self.termination_prob,
                                           prisoner1_history,
                                           prisoner2_history)
        move_2 = prisoner2.choose_one_move(self.payoff,
                                           self.termination_prob,
                                           prisoner2_history,
                                           prisoner1_history)
        rewards: tuple[float, float] = self.moves_to_rewards[(move_1, move_2)]
        return (move_1, move_2), rewards

    def play_one_vs_one_game(self, prisoner_1: Prisoner, prisoner_2: Prisoner) -> tuple[float, float]:
        assert self.games_score.get((prisoner_1.name, prisoner_2.name)) is None
        game_score_1, game_score_2 = 0, 0  # Init. the score for this game
        for _ in range(self.max_rounds):
            (move_1, move_2), (move_score_1, move_score_2) = self.play_one_move(prisoner_1, prisoner_2)
            self.history[(prisoner_1.name, prisoner_2.name)].append((move_1, move_2))
            game_score_1 += move_score_1
            game_score_2 += move_score_2
            if random.random() < self.termination_prob:
                break
        self.games_score[(prisoner_1.name, prisoner_2.name)] = (game_score_1, game_score_2)
        return game_score_1, game_score_2

    def play_one_round_robin_game(self) -> None:
        n_prisoners = len(self.prisoners)
        # Make all pairs of integers from 0 to n_prisoners-1 included, where the first integer is < second integer
        matches = list(combinations(range(0, n_prisoners), 2))
        for match in matches:  # Play the matches
            self.play_one_vs_one_game(self.prisoners[match[0]], self.prisoners[match[1]])
        # Calculate the overall score of each prisoner based on the scores after every game
        for prisoners, scores in self.games_score.items():
            self.prisoners_score[prisoners[0]] += scores[0]
            self.prisoners_score[prisoners[1]] += scores[1]


def instantiate_prisoners_CB() -> tuple[Prisoner, ...]:
    res: tuple[Prisoner, Prisoner] = (Prisoner("Gino"), Prisoner("Pilotino"))
    return res


def instantiate_prisoners_2_CB() -> tuple[Prisoner, ...]:
    res: tuple[Prisoner, Prisoner] = (TitForTat("Gino"), TitForTat("Pilotino"))
    return res


def instantiate_4_prisoners_CB() -> tuple[Prisoner, ...]:
    res = (TitForTat('Tit4Tat_1'), TitForTat('Tit4Tat_2'), Prisoner('Random_1'), Prisoner('Random_2'))
    return res


def main() -> None:
    random.seed(31415)
    payoff: Payoff = Payoff(reward=3, punishment=1, temptation=5, sucker=0)
    tournament: Tournament = Tournament(instantiate_4_prisoners_CB,
                                        payoff=payoff,
                                        termination_prob=1e-9,
                                        max_rounds=200)
    tournament.play_one_round_robin_game()
    for k, v in tournament.games_score.items():
        print(f'{k} : {v}')
    print()
    for k, v in tournament.prisoners_score.items():
        print(f'{k} : {v}')



if __name__ == '__main__':
    main()
