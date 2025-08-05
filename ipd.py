from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations
from enum import Enum, auto
from typing import NamedTuple
import random
import os
from pathlib import Path
from datetime import datetime
import asyncio
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from omegaconf import OmegaConf, DictConfig

# To line profile:
# kernprof -l -v  main_async.py
# The add `@profile` before the methods/functions of interest


config_file = 'config.yaml'


class Move(Enum):
    COOPERATE = auto()
    DEFECT = auto()


class Payoff(NamedTuple):
    reward: float
    punishment: float
    temptation: float
    sucker: float


def get_time_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class Prisoner(ABC):
    def __init__(self, name: str, seed: int = None) -> None:
        self.name: str = name
        self.rng = random.Random(seed)

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
                              log_file: str,
                              max_API_call_attempts: int) -> Move | None:

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
    def __init__(self, name: str, seed: int = None) -> None:
        super().__init__(name, seed)

    async def _choose_one_move(self,
                               payoff: Payoff,
                               termination_prob: float,
                               max_turns: int,
                               history: list[Move],
                               opponent_history: list[Move],
                               log_file: str) -> Move:
        my_move = self.rng.choice((Move.COOPERATE, Move.DEFECT))
        return my_move


class TitForTat(Prisoner):
    def __init__(self, name: str, seed: int = None) -> None:
        super().__init__(name, seed)

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
    def __init__(self, name: str, seed: int = None) -> None:
        super().__init__(name, seed)

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


class LLM(Prisoner):
    def __init__(self,
                 llm_client: str,
                 name: str,
                 model_name: str,
                 temperature: float, max_tokens: int,
                 seed: int = None) -> None:
        super().__init__(name, seed)
        # Get the API key to be used with the LLM API
        load_dotenv()
        api_key_env_var = f"{llm_client.upper()}_API_KEY"
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise ValueError(f"Please set the {api_key_env_var} environment variable.")

        # Instantiate the LangChain `BaseChatModel` specific for the LLM at hand
        self._llm_client = globals()[llm_client](api_key=api_key,
                                                 model_name=model_name,
                                                 temperature=temperature,
                                                 max_tokens=max_tokens)

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
                                                           f'In any case, the game will not last more than {max_turns} turns. '
                                                           f'This is turn number {len(history) + 1}.'
                                                           'The payoff matrix and termination probability remain constant throughout the game. '
                                                           f'Previous moves — You: {me}. Opponent: {opp}.\n'
                                                           'Your goal is to maximize your total score across all turns.\n'
                                                           'Choose your next move between cooperate and defect.'
                                                           'Provide a brief reasoning for the choice of move, then on a new line output your move for this turn as a single character, without any emphasis (like bold or italic): `C` to cooperate or `D` to defect.'
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

    def __init__(self,
                 prisoner: Prisoner,
                 opponent: Prisoner,
                 log_dir: str,
                 max_API_call_attempts: int = 1,
                 seed: int = None) -> None:
        assert prisoner.name != opponent.name  # Prisoner names must be unique
        # Store the two Prisoners between self.prisoner_1 and self.prisoner_2 in alphabetical order by name
        self.prisoner_1, self.prisoner_2 = (prisoner, opponent) if prisoner.name < opponent.name else (opponent,
                                                                                                       prisoner)
        cls = type(self)
        cls.counter += 1
        self.match_counter = cls.counter
        self.log_files = [f'{log_dir}/{self.prisoner_1.name} match_{self.match_counter:04d}.txt',
                          f'{log_dir}/{self.prisoner_2.name} match_{self.match_counter:04d}.txt']
        self.history = ([], [])
        self.scores = [0, 0]
        self.normalised_scores = [.0, .0]

        self.rng = random.Random(seed)
        self.max_API_call_attempts = max_API_call_attempts


class Tournament:
    def __init__(self, config: DictConfig) -> None:
        # config_as_dict = OmegaConf.to_container(config, resolve=True)
        assert 0 <= config.common.termination_probability <= 1
        payoff = Payoff(reward=config.common.payoff.reward,
                        punishment=config.common.payoff.punishment,
                        temptation=config.common.payoff.temptation,
                        sucker=config.common.payoff.sucker)

        progressive_seed = config.common.seed
        self.payoff = payoff
        self.termination_prob = config.common.termination_probability
        self.max_rounds = config.common.max_rounds
        self.n_rematches = config.common.n_rematches
        self.log_dir = f'logs/{get_time_stamp()}'
        Path(self.log_dir).mkdir(exist_ok=True)
        self.scores = defaultdict(float)
        self.normalised_scores = defaultdict(float)
        self.max_concurrent_matches = config.common.max_concurrent_matches
        n_prisoners = len(config.prisoners)
        # Make all pairs of integers from 0 to n_prisoners-1 included, where the first integer is < second integer
        self.matches_idx = list(combinations(range(0, n_prisoners), 2))
        # Instantiate the matches, giving to each its own seed for its RNG
        self.matches = []
        config_as_dict: dict = OmegaConf.to_container(config, resolve=True)
        for i_rematch in range(self.n_rematches):
            for item in self.matches_idx:
                config_prisoner_1 = config_as_dict['prisoners'][item[0]]
                config_prisoner_2 = config_as_dict['prisoners'][item[1]]
                # Augment the parameters for the Prisoner ctor with the seed for RNG
                params_with_seed_1 = config_prisoner_1['params'] | {'seed': progressive_seed}
                if progressive_seed is not None:
                    progressive_seed += 1
                params_with_seed_2 = config_prisoner_2['params'] | {'seed': progressive_seed}
                if progressive_seed is not None:
                    progressive_seed += 1
                prisoner_1 = globals()[config_prisoner_1['prisoner_class']](**params_with_seed_1)
                prisoner_2 = globals()[config_prisoner_2['prisoner_class']](**params_with_seed_2)
                self.matches.append(Match(prisoner_1,
                                          prisoner_2,
                                          log_dir=self.log_dir,
                                          max_API_call_attempts=config.common.max_API_call_attempts,
                                          seed=progressive_seed))

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
                                                       match.log_files[0],
                                                       match.max_API_call_attempts)
        move_2_task = match.prisoner_2.choose_one_move(self.payoff,
                                                       self.termination_prob,
                                                       self.max_rounds,
                                                       match.history[1],
                                                       match.history[0],
                                                       match.log_files[1],
                                                       match.max_API_call_attempts)
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
        assert match.normalised_scores == [.0, .0]
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
                # Use the Match's own RNG
                if match.rng.random() < self.termination_prob:
                    break

        if len(match.history[0]) > 0:
            match.normalised_scores[0] = match.scores[0] / len(match.history[0])
        if len(match.history[1]) > 0:
            match.normalised_scores[1] = match.scores[1] / len(match.history[1])

        return match.scores

    def play_one_round_robin_tournament(self) -> None:
        async def _async_tournament():
            # Create a semaphore to limit concurrent games
            semaphore = asyncio.Semaphore(self.max_concurrent_matches)

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
                self.scores[match.prisoner_1.name] += match.scores[0]
                self.scores[match.prisoner_2.name] += match.scores[1]
                self.normalised_scores[match.prisoner_1.name] += match.normalised_scores[0]
                self.normalised_scores[match.prisoner_2.name] += match.normalised_scores[1]

        # Run the async tournament in a new event loop
        asyncio.run(_async_tournament())


def main() -> None:
    config = OmegaConf.load(config_file)
    tournament = Tournament(config=config)

    tournament.play_one_round_robin_tournament()

    print('\nTOURNAMENT MATCHES')
    for match in tournament.matches:
        print(
            f'Match No. {match.match_counter}: {match.prisoner_1.name} Vs. {match.prisoner_2.name} ended in {len(match.history[0])} round(s) with scores {match.scores}')

    print('\nPRISONERS AND THEIR SCORE')
    for prisoner_name, score in tournament.scores.items():
        print(f"{prisoner_name} {score:.3f}")

    print('\nPRISONERS AND THEIR NORMALIZED SCORE')
    for prisoner_name, score in tournament.normalised_scores.items():
        print(f"{prisoner_name} {score:.3f}")


if __name__ == '__main__':
    main()
