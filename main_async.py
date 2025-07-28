from __future__ import annotations
from collections.abc import Callable
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations
from enum import Enum, auto
from typing import NamedTuple
import random
import os
from datetime import datetime
import asyncio
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


class Move(Enum):
    COOPERATE = auto()
    DEFECT = auto()


class Payoff(NamedTuple):
    reward: float
    punishment: float
    temptation: float
    sucker: float


class Prisoner(ABC):
    def __init__(self, name: str) -> None:
        self.name: str = name

    @abstractmethod
    async def choose_one_move(self,
                              payoff: Payoff,
                              termination_prob: float,
                              max_turns: int,
                              history: list[Move],
                              opponent_history: list[Move]) -> Move:
        ...


class Random(Prisoner):
    async def choose_one_move(self,
                              payoff: Payoff,
                              termination_prob: float,
                              max_turns: int,
                              history: list[Move],
                              opponent_history: list[Move]) -> Move:
        my_move = random.choice((Move.COOPERATE, Move.DEFECT))
        return my_move


class TitForTat(Prisoner):
    async def choose_one_move(self,
                              payoff: Payoff,
                              termination_prob: float,
                              max_turns: int,
                              history: list[Move],
                              opponent_history: list[Move]) -> Move:
        my_move = Move.COOPERATE if len(opponent_history) == 0 else opponent_history[-1]
        return my_move


def classify_payoff(my_move: Move, opponent_move: Move) -> str:
    match my_move:
        case Move.COOPERATE:
            return 'reward' if opponent_move == Move.COOPERATE else 'sucker'
        case Move.DEFECT:
            return 'temptation' if opponent_move == Move.COOPERATE else 'punishment'
    assert False  # Shouldn't ever get here


class WinStayLoseShift(Prisoner):
    async def choose_one_move(self,
                              payoff: Payoff,
                              termination_prob: float,
                              max_turns: int,
                              history: list[Move],
                              opponent_history: list[Move]) -> Move:

        # First move always cooperate
        if len(opponent_history) == 0:
            return Move.COOPERATE

        # The payoff of the previous turn was one of the best two for me (either reward or temptation), then repeat
        # that same move
        payoff_type = classify_payoff(history[-1], opponent_history[-1])
        if payoff_type in ('reward', 'temptation'):
            return history[-1]
        # Otherwise, switch to the other move
        return Move.COOPERATE if history[-1] == Move.DEFECT else Move.DEFECT





def get_time_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class OpenAI(Prisoner):
    def __init__(self, name: str, model_name: str) -> None:
        super().__init__(name)
        time_stamp = get_time_stamp()
        self.log_file_name = f'logs/log_{name}_{time_stamp}.txt'
        self.moves_count = 0
        self._llm_client = ChatOpenAI(openai_api_key=api_key,
                                      model_name=model_name,
                                      temperature=0.7,  # adjust for creativity (0.0–1.0)
                                      max_tokens=512)

    async def choose_one_move(self,
                              payoff: Payoff,
                              termination_prob: float,
                              max_turns: int,
                              history: list[Move],
                              opponent_history: list[Move]) -> Move:
        # Format the payoff matrix
        payoff_text = (f"Reward (C,C): {payoff.reward}, "
                       f"Punishment (D,D): {payoff.punishment}, "
                       f"Temptation (D,C): {payoff.temptation}, "
                       f"Sucker (C,D): {payoff.sucker}.")
        # Prepare history strings
        me = ' '.join('C' if m == Move.COOPERATE else 'D' for m in history) or 'none'
        opp = ' '.join('C' if m == Move.COOPERATE else 'D' for m in opponent_history) or 'none'
        # Construct prompt
        prompt = ("You are playing the Iterated Prisoner's Dilemma. "
                  "The payoff matrix is: " + payoff_text + "\n"
                                                           f"After each turn, the game terminates with probability {termination_prob}. "
                                                           "The payoff matrix and termination probability remain constant throughout the game. "
                                                           f"The game will not last more than {max_turns} turns.\n"
                                                           f"Previous moves — You: {me}. Opponent: {opp}.\n"
                                                           "Your goal is to maximize your total score across all turns.\n"
                                                           "Provide a brief reasoning, then on a new line output a single character, without any emphasis (like bold or italic): 'C' to cooperate or 'D' to defect.")
        # Invoke the model asynchronously
        response = await asyncio.to_thread(lambda: self._llm_client.invoke([HumanMessage(content=prompt)]))

        # The raw API response is tucked into `result.llm_output`.
        prompt_tokens = response.usage_metadata.get('input_tokens', 'N/A')
        completion_tokens = response.usage_metadata.get('output_tokens', 'N/A')
        total_tokens = response.usage_metadata.get('total_tokens', 'N/A')

        self.moves_count += 1
        # Make file writing async too
        await asyncio.to_thread(self._write_log, prompt, response, prompt_tokens, completion_tokens, total_tokens)

        # Parse the decision from the last line
        decision = response.content.strip().splitlines()[-1].strip().upper()
        assert decision in ('C', 'D')
        return Move.COOPERATE if decision == 'C' else Move.DEFECT

    def _write_log(self, prompt: str, response, prompt_tokens, completion_tokens, total_tokens) -> None:
        """Helper method for synchronous file writing"""
        with open(self.log_file_name, 'a') as the_log:
            the_log.write(f'MOVES COUNT: {self.moves_count}\n')
            the_log.write(f'TIME STAMP: {get_time_stamp()}\n')
            the_log.write(f'PROMPT/COMPLETION/TOTAL TOKENS: {prompt_tokens}/{completion_tokens}/{total_tokens}\n')
            the_log.write('PROMPT:\n')
            the_log.write(prompt + '\n')
            the_log.write('RESPONSE:\n')
            the_log.write(response.content + '\n\n')


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

    async def play_one_move(self, prisoner1: Prisoner, prisoner2: Prisoner) -> tuple[
        tuple[Move, Move], tuple[float, float]]:
        prisoner1_history = [move for (move, _) in self.history[(prisoner1.name, prisoner2.name)]]
        prisoner2_history = [move for (_, move) in self.history[(prisoner1.name, prisoner2.name)]]

        # Run both players' decision-making concurrently
        move_1_task = prisoner1.choose_one_move(self.payoff,
                                                self.termination_prob,
                                                self.max_rounds,
                                                prisoner1_history,
                                                prisoner2_history)
        move_2_task = prisoner2.choose_one_move(self.payoff,
                                                self.termination_prob,
                                                self.max_rounds,
                                                prisoner2_history,
                                                prisoner1_history)

        move_1, move_2 = await asyncio.gather(move_1_task, move_2_task)
        rewards: tuple[float, float] = self.moves_to_rewards[(move_1, move_2)]
        return (move_1, move_2), rewards

    async def play_one_vs_one_game(self, prisoner_1: Prisoner, prisoner_2: Prisoner) -> tuple[float, float]:
        assert self.games_score.get((prisoner_1.name, prisoner_2.name)) is None
        game_score_1, game_score_2 = 0, 0  # Init. the score for this game
        for _ in range(self.max_rounds):
            (move_1, move_2), (move_score_1, move_score_2) = await self.play_one_move(prisoner_1, prisoner_2)
            self.history[(prisoner_1.name, prisoner_2.name)].append((move_1, move_2))
            game_score_1 += move_score_1
            game_score_2 += move_score_2
            if random.random() < self.termination_prob:
                break
        self.games_score[(prisoner_1.name, prisoner_2.name)] = (game_score_1, game_score_2)
        return game_score_1, game_score_2

    def play_one_round_robin_tournament(self, seed=None, max_concurrent_games=32) -> None:
        async def _async_tournament():
            if seed is not None:
                random.seed(seed)
            n_prisoners = len(self.prisoners)
            # Make all pairs of integers from 0 to n_prisoners-1 included, where the first integer is < second integer
            matches = list(combinations(range(0, n_prisoners), 2))

            # Create a semaphore to limit concurrent games
            semaphore = asyncio.Semaphore(max_concurrent_games)

            async def play_game_with_semaphore(match):
                async with semaphore:
                    return await self.play_one_vs_one_game(self.prisoners[match[0]], self.prisoners[match[1]])

            # Create tasks for all matches
            tasks = [play_game_with_semaphore(match) for match in matches]

            # Run all games concurrently with progress bar
            await tqdm.gather(*tasks, desc="Playing matches")

            # Calculate the overall score of each prisoner based on the scores after every game
            # The overall score is per move and per game
            for prisoners, scores in self.games_score.items():
                self.prisoners_score[prisoners[0]] += scores[0] / len(self.history[prisoners])
                self.prisoners_score[prisoners[1]] += scores[1] / len(self.history[prisoners])
            for prisoner, score in self.prisoners_score.items():
                self.prisoners_score[prisoner] /= len(self.history)

        # Run the async tournament in a new event loop
        asyncio.run(_async_tournament())


def instantiate_4_prisoners_CB() -> tuple[Prisoner, ...]:
    res = (TitForTat('Tit4Tat_1'), TitForTat('Tit4Tat_2'), Random('Drunk_1'), Random('Drunk_2'))
    return res


def instantiate_6_prisoners_CB() -> tuple[Prisoner, ...]:
    res = (TitForTat('Tit4Tat_1'),
           TitForTat('Tit4Tat_2'),
           Random('Drunk_1'),
           Random('Drunk_2'),
           WinStayLoseShift('WSLS_1'),
           WinStayLoseShift('WSLS_2'))
    return res


def instantiate_2_prisoners_CB() -> tuple[Prisoner, ...]:
    res = (TitForTat('Tit4Tat_1'),
           OpenAI('gpt-4o-mini_1', 'gpt-4o-mini'))  # Fixed model name
    return res


def instantiate_4_prisoners_with_AI_CB() -> tuple[Prisoner, ...]:
    res = (TitForTat('Tit4Tat_1'),
           OpenAI('gpt-4.1-nano_1', 'gpt-4.1-nano'),
           OpenAI('gpt-4.1-nano_2', 'gpt-4.1-nano'),
           OpenAI('gpt-4.1-nano_3', 'gpt-4.1-nano'),
           OpenAI('gpt-4.1-nano_4', 'gpt-4.1-nano'),
           Random('Random_1'))
    return res


def main() -> None:
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    payoff: Payoff = Payoff(reward=3, punishment=1, temptation=5, sucker=0)
    tournament: Tournament = Tournament(instantiate_4_prisoners_with_AI_CB,
                                        payoff=payoff,
                                        # termination_prob=0.004047,
                                        termination_prob=0.0341,
                                        max_rounds=20)

    tournament.play_one_round_robin_tournament(seed=31415, max_concurrent_games=16)

    for k, v in tournament.games_score.items():
        game_length = len(tournament.history[k])
        print(f'{k} : {v} after {game_length} turns')
    print()
    for k, v in tournament.prisoners_score.items():
        print(f'{k} : {v}')


if __name__ == '__main__':
    main()
