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
# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     raise ValueError("Please set the OPENAI_API_KEY environment variable.")

max_API_call_attempts = 5


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
    async def _choose_one_move(self,
                               payoff: Payoff,
                               termination_prob: float,
                               max_turns: int,
                               history: list[Move],
                               opponent_history: list[Move],
                               log_file: str) -> Move:
        ...

    async def choose_one_move(self,
                              payoff: Payoff,
                              termination_prob: float,
                              max_turns: int,
                              history: list[Move],
                              opponent_history: list[Move],
                              log_file: str) -> Move | None:

        for _ in range(max_API_call_attempts):
            try:
                move = await self._choose_one_move(payoff,
                                                   termination_prob,
                                                   max_turns,
                                                   history,
                                                   opponent_history,
                                                   log_file)
                return move

            except Exception as e:
                print(f'Got exception from `_choose_one_move()` for prisoner `{self.name}`: {e}')

        print(f'Got too many errors calling `_choose_one_move()` for prisoner `{self.name}`; giving up')
        return None


class Random(Prisoner):
    async def _choose_one_move(self,
                               payoff: Payoff,
                               termination_prob: float,
                               max_turns: int,
                               history: list[Move],
                               opponent_history: list[Move],
                               log_file: str) -> Move:
        my_move = random.choice((Move.COOPERATE, Move.DEFECT))
        return my_move


class TitForTat(Prisoner):
    async def _choose_one_move(self,
                               payoff: Payoff,
                               termination_prob: float,
                               max_turns: int,
                               history: list[Move],
                               opponent_history: list[Move],
                               log_file: str) -> Move:
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
    async def _choose_one_move(self,
                               payoff: Payoff,
                               termination_prob: float,
                               max_turns: int,
                               history: list[Move],
                               opponent_history: list[Move],
                               log_file: str) -> Move:

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
        """
        Note: o4-mini can use a couple of thousand tokens for its internal reasoning, which makes it even more expensive
        :param name:
        :param model_name:
        """

        super().__init__(name)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable.")

        self._llm_client = ChatOpenAI(openai_api_key=api_key,
                                      model_name=model_name,
                                      temperature=1.,  # adjust for creativity (0.0–1.0)
                                      max_tokens=5000)

    async def _choose_one_move(self,
                               payoff: Payoff,
                               termination_prob: float,
                               max_turns: int,
                               history: list[Move],
                               opponent_history: list[Move],
                               log_file: str) -> Move | None:
        # Format the payoff matrix
        payoff_text = (f"Reward (C,C): {payoff.reward}, "
                       f"Punishment (D,D): {payoff.punishment}, "
                       f"Temptation (D,C): {payoff.temptation}, "
                       f"Sucker (C,D): {payoff.sucker}.")
        # Prepare history strings
        me = ' '.join('C' if m == Move.COOPERATE else 'D' for m in history) or 'none'
        opp = ' '.join('C' if m == Move.COOPERATE else 'D' for m in opponent_history) or 'none'
        # Construct prompt
        prompt = ('You are playing the Iterated Prisoner\'s Dilemma. '
                  'The payoff matrix is: ' + payoff_text + '\n'
                                                           f'After each turn, the game terminates with probability {termination_prob}. '
                                                           'The payoff matrix and termination probability remain constant throughout the game. '
                                                           f'The game will not last more than {max_turns} turns.\n'
                                                           f'Previous moves — You: {me}. Opponent: {opp}.\n'
                                                           'Your goal is to maximize your total score across all turns.\n'
                                                           'Choose your next move between cooperate and defect.'
                                                           'Provide a brief reasoning for the choice of move, then on a new line output your move as a single character, without any emphasis (like bold or italic): `C` to cooperate or `D` to defect.'
                                                           'Ensure the last line of your output contains one character (either `C` or `D`), and one character only.')
        # Invoke the model asynchronously
        response = await asyncio.to_thread(lambda: self._llm_client.invoke([HumanMessage(content=prompt)]))

        # The raw API response is tucked into `result.llm_output`.
        prompt_tokens = response.usage_metadata.get('input_tokens', 'N/A')
        completion_tokens = response.usage_metadata.get('output_tokens', 'N/A')
        total_tokens = response.usage_metadata.get('total_tokens', 'N/A')

        decision = None
        with open(log_file, 'a') as the_log:
            the_log.write(f'MOVES COUNT: {len(history) + 1}\n')
            the_log.write(f'TIME STAMP: {get_time_stamp()}\n')
            the_log.write(f'PROMPT/COMPLETION/TOTAL TOKENS: {prompt_tokens}/{completion_tokens}/{total_tokens}\n')
            the_log.write('PROMPT:\n')
            the_log.write(prompt + '\n')
            the_log.write('RESPONSE:\n')
            the_log.write(response.content + '\n')
            # Parse the decision from the last line. If it raises an IndexError,
            # then catch it and set the decision to ''
            try:
                decision = response.content.strip().splitlines()[-1].strip().upper()
            except IndexError:
                decision = None
            if decision not in ('C', 'D'):
                the_log.write(f'ERROR IN PARSING DECISION: `{decision}`\n')
            the_log.write('\n-----------------------------------------------\n')

        match decision:
            case 'C':
                return Move.COOPERATE
            case 'D':
                return Move.DEFECT

        raise ValueError(f'Parsed invalid decision `{decision}` from response `{response.content}`')


class Match:
    counter = 0

    def __init__(self, prisoner: Prisoner, opponent: Prisoner) -> None:
        assert prisoner.name != opponent.name  # Prisoner names must be unique
        # Store the two Prisoners between self.prisoner_1 and self.prisoner_2 in alphabetical order by name
        self.prisoner_1, self.prisoner_2 = (prisoner, opponent) if prisoner.name < opponent.name else (opponent,
                                                                                                       prisoner)
        cls = type(self)
        cls.counter += 1
        self.match_counter = cls.counter
        time_stamp = get_time_stamp()
        self.log_files = [f'logs/log_{self.prisoner_1.name} match_{self.match_counter} {time_stamp}.txt',
                          f'logs/log_{self.prisoner_2.name} match_{self.match_counter} {time_stamp}.txt']
        self.history = ([], [])
        self.scores = [0, 0]


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
        self.scores = defaultdict(float)
        n_prisoners = len(self.prisoners)
        # Make all pairs of integers from 0 to n_prisoners-1 included, where the first integer is < second integer
        self.matches_idx = list(combinations(range(0, n_prisoners), 2))
        self.matches = [Match(self.prisoners[item[0]], self.prisoners[item[1]]) for item in self.matches_idx]

        self.moves_to_rewards: dict[tuple[Move, Move], tuple[float, float]] = {
            (Move.COOPERATE, Move.COOPERATE): (self.payoff.reward, self.payoff.reward),
            (Move.DEFECT, Move.DEFECT): (self.payoff.punishment, self.payoff.punishment),
            (Move.COOPERATE, Move.DEFECT): (self.payoff.sucker, self.payoff.temptation),
            (Move.DEFECT, Move.COOPERATE): (self.payoff.temptation, self.payoff.sucker)}

    async def play_one_turn(self, match: Match) -> tuple[tuple[Move, Move], tuple[float, float]]:
        # Run both players' decision-making concurrently
        move_1_task = match.prisoner_1.choose_one_move(self.payoff,
                                                       self.termination_prob,
                                                       self.max_rounds,
                                                       match.history[0],
                                                       match.history[1],
                                                       match.log_files[0])
        move_2_task = match.prisoner_2.choose_one_move(self.payoff,
                                                       self.termination_prob,
                                                       self.max_rounds,
                                                       match.history[1],
                                                       match.history[0],
                                                       match.log_files[1])
        move_1, move_2 = await asyncio.gather(move_1_task, move_2_task)

        if move_1 is None or move_2 is None:
            exc_msg = f'Error during match {match.match_counter}'
            if move_1 is None:
                exc_msg = f' Prisoner {match.prisoner_1.name} did not choose any move.'
            if move_2 is None:
                exc_msg = exc_msg + f' Prisoner {match.prisoner_2.name} did not choose any move.'
            raise ValueError(exc_msg)

        rewards: tuple[float, float] = self.moves_to_rewards[(move_1, move_2)]
        return (move_1, move_2), rewards

    async def play_one_match(self, match: Match) -> list[float]:
        assert match.scores == [0, 0]
        for _ in range(self.max_rounds):
            try:
                # Play one turn (Prisoners make a simultaneous move)
                (move_1, move_2), (move_1_reward, move_2_reward) = await self.play_one_turn(match)
            except Exception as e:
                print(f'Stopping match no. {match.match_counter} because of unrecoverable error: {e}')
                break
            else:  # else block needed just to suppress PyCharm bogus warnings
                # Update history and scores in the Match
                match.history[0].append(move_1)
                match.history[1].append(move_2)
                match.scores[0] += move_1_reward
                match.scores[1] += move_2_reward
                if random.random() < self.termination_prob:
                    break
        return match.scores

    def play_one_round_robin_tournament(self, seed=None, max_concurrent_games=32) -> None:
        async def _async_tournament():
            if seed is not None:
                random.seed(seed)

            # Create a semaphore to limit concurrent games
            semaphore = asyncio.Semaphore(max_concurrent_games)

            async def play_game_with_semaphore(match):
                async with semaphore:
                    return await self.play_one_match(match)

            # Create tasks for all matches
            tasks = [play_game_with_semaphore(match) for match in self.matches]

            # Run all games concurrently with progress bar
            await tqdm.gather(*tasks, desc="Playing matches")

            # Calculate the overall score of each prisoner based on the scores after every match
            # The score for each match is normalized by dividing it by the number of turns in the match
            for match in self.matches:
                # In case a match stopped during the first turn because of an error, the history is still empty
                # Make sure there is no division by zero thereafter
                denominator_1 = len(match.history[0]) if match.history[0] else 1
                denominator_2 = len(match.history[1]) if match.history[1] else 1
                self.scores[match.prisoner_1.name] += match.scores[0] / denominator_1
                self.scores[match.prisoner_2.name] += match.scores[1] / denominator_2

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


def instantiate_6_prisoners_with_AI_CB() -> tuple[Prisoner, ...]:
    res = (TitForTat('Tit4Tat_1'),
           OpenAI('gpt-4.1-nano_1', 'gpt-4.1-nano'),
           OpenAI('gpt-4.1-nano_2', 'gpt-4.1-nano'),
           OpenAI('gpt-4.1-nano_3', 'gpt-4.1-nano'),
           OpenAI('gpt-4.1-nano_4', 'gpt-4.1-nano'),
           Random('Random_1'))
    return res


def instantiate_4_prisoners_with_AI_CB() -> tuple[Prisoner, ...]:
    res = (OpenAI('gpt-4.1-nano_1', 'gpt-4.1-nano'),
           OpenAI('gpt-4.1-nano_2', 'gpt-4.1-nano'),
           OpenAI('gpt-4.1-nano_3', 'gpt-4.1-nano'),
           OpenAI('gpt-4.1-nano_4', 'gpt-4.1-nano'))
    return res


def instantiate_misc_models_CB() -> tuple[Prisoner, ...]:
    res = (OpenAI('gpt-4.1-nano_1', 'gpt-4.1-nano'),
           OpenAI('gpt-4o-mini_1', 'gpt-4o-mini'),
           OpenAI('o4-mini_1', 'o4-mini'),
           Random('Random_1'),
           TitForTat('Tit4Tat_1'))
    return res


def main() -> None:
    payoff: Payoff = Payoff(reward=3, punishment=1, temptation=5, sucker=0)
    tournament: Tournament = Tournament(instantiate_misc_models_CB,
                                        payoff=payoff,
                                        termination_prob=0.0341,
                                        # termination_prob=0.06697,  # 0.0341 is for 20 rounds
                                        max_rounds=20)

    tournament.play_one_round_robin_tournament(seed=31415, max_concurrent_games=16)

    print('\nTOURNAMENT MATCHES')
    for match in tournament.matches:
        print(
            f'Match No. {match.match_counter}: {match.prisoner_1.name} Vs. {match.prisoner_2.name} ended in {len(match.history[0])} round(s) with scores {match.scores}')

    print('\nPRISONERS AND THEIR NORMALIZED SCORE')
    for prisoner_name, score in tournament.scores.items():
        print(prisoner_name, score)


if __name__ == '__main__':
    main()

# TODO

"""
- Make the temperature and max token no. easily configurable by AI model
- Load parameters from config. file

"""