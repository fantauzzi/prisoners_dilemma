# Mainly to resolve the chicken-and-egg issue with the definition of Prisoner and Tournament
from __future__ import annotations

from typing import NamedTuple
from enum import Enum, auto
import random


class Move(Enum):
    COOPERATE = auto()
    DEFECT = auto()


class Payoff(NamedTuple):
    Reward: float
    Punishment: float
    Temptation: float
    Sucker: float


class Prisoner:
    def __init__(self, name: str) -> None:
        self.name = name
        self.opponent_moves: list[Move] = []
        self.my_moves: list[Move] = []
        self._tournament: Tournament | None = None

    def set_tournament(self, tournament: Tournament) -> None:
        self._tournament = tournament

    def log_opponent_move(self, opponent_move: Move) -> None:
        self.opponent_moves.append(opponent_move)

    def log_my_move(self, my_move: Move) -> None:
        self.my_moves.append(my_move)

    def choose_one_move(self) -> Move:
        assert len(self.my_moves) == len(self.opponent_moves)
        my_move = random.choice((Move.COOPERATE, Move.DEFECT))
        self.log_my_move(my_move)
        return my_move


class Tournament:
    def __init__(self, prisoners: list[Prisoner], payoff: Payoff, termination_prob: float,) -> None:
        assert 0 <= termination_prob <= 1
        self.prisoners: tuple[Prisoner, ...] = tuple(prisoners)
        self.payoff = payoff
        self.termination_prob = termination_prob

        for prisoner in self.prisoners:
            prisoner.set_tournament(self)


def main():
    print('Howdy partner!')
    random.seed(31415)


if __name__ == '__main__':
    main()
