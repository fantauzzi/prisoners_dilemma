from __future__ import annotations
from collections.abc import Callable
from enum import Enum, auto
from typing import NamedTuple
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
        self.name: str = name
        self.opponent_moves: list[Move] = []
        self.my_moves: list[Move] = []

    def log_opponent_move(self, opponent_move: Move) -> None:
        self.opponent_moves.append(opponent_move)

    def log_my_move(self, my_move: Move) -> None:
        self.my_moves.append(my_move)

    def choose_one_move(self,
                        payoff: Payoff,
                        termination_prob: float,
                        prev_opponent_move: Move | None, ) -> Move:
        if prev_opponent_move is not None:
            self.log_opponent_move(prev_opponent_move)
        assert len(self.my_moves) == len(self.opponent_moves)
        my_move: Move = random.choice((Move.COOPERATE, Move.DEFECT))
        self.log_my_move(my_move)

        return my_move


class Tournament:
    def __init__(self,
                 instantiate_prisoners_CB: Callable[[], tuple[Prisoner, ...]],
                 payoff: Payoff,
                 termination_prob: float,
                 max_rounds: int) -> None:
        assert 0 <= termination_prob <= 1
        self.prisoners: tuple[Prisoner, ...] = instantiate_prisoners_CB()
        self.payoff: Payoff = payoff
        self.termination_prob: float = termination_prob
        self.scores: list[float] = [0.0] * len(self.prisoners)
        self.terminated: bool = False
        self.completed_round_number: int = 0
        self.max_rounds: int = max_rounds

    def match(self,
              prisoner1: Prisoner,
              prisoner2: Prisoner,
              prev_move_1: Move | None,
              prev_move_2: Move | None, ) -> tuple[Move, Move]:
        assert not self.terminated
        moves_to_rewards: dict[tuple[Move, Move], tuple[float, float]] = {
            (Move.COOPERATE, Move.COOPERATE): (self.payoff.Reward, self.payoff.Reward),
            (Move.DEFECT, Move.DEFECT): (self.payoff.Punishment, self.payoff.Punishment),
            (Move.COOPERATE, Move.DEFECT): (self.payoff.Sucker, self.payoff.Temptation),
            (Move.DEFECT, Move.COOPERATE): (self.payoff.Temptation, self.payoff.Sucker)}

        move_1: Move = prisoner1.choose_one_move(self.payoff, self.termination_prob, prev_move_2)
        move_2: Move = prisoner2.choose_one_move(self.payoff, self.termination_prob, prev_move_1)
        rewards: tuple[float, float] = moves_to_rewards[(move_1, move_2)]
        self.scores[0] += rewards[0]
        self.scores[1] += rewards[1]

        self.terminated = random.random() < self.termination_prob
        return move_1, move_2

    def play_one_game(self) -> None:
        assert not self.terminated
        assert len(self.prisoners) == 2
        prev_move_1: Move | None = None
        prev_move_2: Move | None = None
        while not self.terminated and self.completed_round_number < self.max_rounds:
            prev_move_1, prev_move_2 = self.match(self.prisoners[0],
                                                  self.prisoners[1],
                                                  prev_move_1,
                                                  prev_move_2)
            self.completed_round_number += 1


def instantiate_prisoners_CB() -> tuple[Prisoner, Prisoner]:
    res: tuple[Prisoner, Prisoner] = (Prisoner("Gino"), Prisoner("Pilotino"))
    return res


def main() -> None:
    random.seed(31415)
    payoff: Payoff = Payoff(Reward=3, Punishment=0, Temptation=5, Sucker=1)
    tournament: Tournament = Tournament(instantiate_prisoners_CB, payoff=payoff, termination_prob=0.02, max_rounds=10)
    tournament.play_one_game()
    print(tournament.scores)
    print(tournament.completed_round_number)


if __name__ == '__main__':
    main()
