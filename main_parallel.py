from __future__ import annotations
from collections.abc import Callable
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations
from enum import Enum, auto
from typing import NamedTuple
import random
import os
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import pickle

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


@dataclass
class GameResult:
    """Result of a single game between two prisoners"""
    prisoner1_name: str
    prisoner2_name: str
    prisoner1_score: float
    prisoner2_score: float
    moves_history: list[tuple[Move, Move]]


class Prisoner(ABC):
    def __init__(self, name: str) -> None:
        self.name: str = name

    @abstractmethod
    def choose_one_move(self,
                        payoff: Payoff,
                        termination_prob: float,
                        max_turns: int,
                        history: list[Move],
                        opponent_history: list[Move]) -> Move:
        ...


class Random(Prisoner):
    def choose_one_move(self,
                        payoff: Payoff,
                        termination_prob: float,
                        max_turns: int,
                        history: list[Move],
                        opponent_history: list[Move]) -> Move:
        my_move = random.choice((Move.COOPERATE, Move.DEFECT))
        return my_move


class TitForTat(Prisoner):
    def choose_one_move(self,
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
    def choose_one_move(self,
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
        last_payoff = classify_payoff(history[-1], opponent_history[-1])
        if last_payoff in ('reward', 'temptation'):
            return history[-1]
        # Otherwise, switch to the other move
        return Move.COOPERATE if history[-1] == Move.DEFECT else Move.DEFECT


from datetime import datetime


def get_time_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class OpenAI(Prisoner):
    def __init__(self, name: str, model_name: str) -> None:
        super().__init__(name)
        self.model_name = model_name
        time_stamp = get_time_stamp()
        self.log_file_name = f'logs/log_{name}_{time_stamp}.txt'
        self.moves_count = 0
        self._llm_client = None  # Will be initialized when needed

    def _get_client(self):
        """Lazy initialization of the LLM client for multiprocessing compatibility"""
        if self._llm_client is None:
            self._llm_client = ChatOpenAI(
                openai_api_key=api_key,
                model_name=self.model_name,
                temperature=0.2,
                max_tokens=512
            )
        return self._llm_client

    def choose_one_move(self,
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
        prompt = (
                "You are playing the Iterated Prisoner's Dilemma. "
                "The payoff matrix is: " + payoff_text + "\n"
                                                         f"After each turn, the game terminates with probability {termination_prob}. "
                                                         "The payoff matrix and termination probability remain constant throughout the game. "
                                                         f"The game will not last more than {max_turns} turns.\n"
                                                         f"Previous moves — You: {me}. Opponent: {opp}.\n"
                                                         "Your goal is to maximize your total score across all turns.\n"
                                                         "Provide a brief reasoning, then on a new line output a single character with your next move, without any emphasis (like bold or italic): 'C' to cooperate or 'D' to defect."
                                                         "Make sure the last line always contains exactly one character, either 'C' or 'D', and nothing else; make sure it doesn't contain anything extra like `My decision:`."
        )
        # Invoke the model
        client = self._get_client()
        response = client.invoke([HumanMessage(content=prompt)])

        # The raw API response is tucked into `result.llm_output`.
        prompt_tokens = response.usage_metadata.get('input_tokens', 'N/A')
        completion_tokens = response.usage_metadata.get('output_tokens', 'N/A')
        total_tokens = response.usage_metadata.get('total_tokens', 'N/A')

        self.moves_count += 1

        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)

        with open(self.log_file_name, 'a') as the_log:
            the_log.write(f'MOVES COUNT: {self.moves_count}\n')
            the_log.write(f'TIME STAMP: {get_time_stamp()}\n')
            the_log.write(f'PROMPT/COMPLETION/TOTAL TOKENS: {prompt_tokens}/{completion_tokens}/{total_tokens}\n')
            the_log.write('PROMPT:\n')
            the_log.write(prompt + '\n')
            the_log.write('RESPONSE:\n')
            the_log.write(response.content + '\n\n')

        # Parse the decision from the last line
        decision = response.content.strip().splitlines()[-1].strip().upper()
        assert decision in ('C', 'D')
        return Move.COOPERATE if decision == 'C' else Move.DEFECT


def play_single_game(prisoner1_data, prisoner2_data, payoff: Payoff,
                     termination_prob: float, max_rounds: int, game_seed: int) -> GameResult:
    """
    Play a single game between two prisoners. This function runs in a separate process.

    Args:
        prisoner1_data: Tuple of (prisoner_class, args, kwargs) for prisoner 1
        prisoner2_data: Tuple of (prisoner_class, args, kwargs) for prisoner 2
        payoff: Payoff structure
        termination_prob: Probability of game termination after each round
        max_rounds: Maximum number of rounds
        game_seed: Random seed for this specific game

    Returns:
        GameResult object containing the game outcome
    """
    # Set the random seed for this process
    random.seed(game_seed)

    # Reconstruct prisoners from the serialized data
    prisoner1_class, prisoner1_args, prisoner1_kwargs = prisoner1_data
    prisoner2_class, prisoner2_args, prisoner2_kwargs = prisoner2_data

    prisoner1 = prisoner1_class(*prisoner1_args, **prisoner1_kwargs)
    prisoner2 = prisoner2_class(*prisoner2_args, **prisoner2_kwargs)

    # Set up the game
    moves_to_rewards = {
        (Move.COOPERATE, Move.COOPERATE): (payoff.reward, payoff.reward),
        (Move.DEFECT, Move.DEFECT): (payoff.punishment, payoff.punishment),
        (Move.COOPERATE, Move.DEFECT): (payoff.sucker, payoff.temptation),
        (Move.DEFECT, Move.COOPERATE): (payoff.temptation, payoff.sucker)
    }

    game_score_1, game_score_2 = 0, 0
    history = []

    for _ in range(max_rounds):
        # Get move histories for each prisoner
        prisoner1_history = [move for (move, _) in history]
        prisoner2_history = [move for (_, move) in history]

        # Get moves from both prisoners
        move_1 = prisoner1.choose_one_move(payoff, termination_prob, max_rounds,
                                           prisoner1_history, prisoner2_history)
        move_2 = prisoner2.choose_one_move(payoff, termination_prob, max_rounds,
                                           prisoner2_history, prisoner1_history)

        # Calculate rewards
        move_score_1, move_score_2 = moves_to_rewards[(move_1, move_2)]

        # Update scores and history
        game_score_1 += move_score_1
        game_score_2 += move_score_2
        history.append((move_1, move_2))

        # Check for termination
        if random.random() < termination_prob:
            break

    return GameResult(
        prisoner1_name=prisoner1.name,
        prisoner2_name=prisoner2.name,
        prisoner1_score=game_score_1,
        prisoner2_score=game_score_2,
        moves_history=history
    )


def serialize_prisoner(prisoner: Prisoner) -> tuple:
    """
    Serialize a prisoner object for multiprocessing.
    Returns (class, args, kwargs) tuple.
    """
    if isinstance(prisoner, Random):
        return (Random, (prisoner.name,), {})
    elif isinstance(prisoner, TitForTat):
        return (TitForTat, (prisoner.name,), {})
    elif isinstance(prisoner, WinStayLoseShift):
        return (WinStayLoseShift, (prisoner.name,), {})
    elif isinstance(prisoner, OpenAI):
        return (OpenAI, (prisoner.name, prisoner.model_name), {})
    else:
        raise ValueError(f"Unknown prisoner type: {type(prisoner)}")


class Tournament:
    def __init__(self,
                 instantiate_prisoners_CB: Callable[[], tuple[Prisoner, ...]],
                 payoff: Payoff,
                 termination_prob: float,
                 max_rounds: int,
                 max_workers: int = None) -> None:
        assert 0 <= termination_prob <= 1
        self.prisoners = instantiate_prisoners_CB()
        self.payoff = payoff
        self.termination_prob = termination_prob
        self.max_rounds = max_rounds
        self.max_workers = max_workers
        self.games_score: dict[tuple[str, str], tuple[float, float]] = {}
        self.prisoners_score = defaultdict(float)
        self.history = defaultdict(list)

    def play_one_round_robin_tournament(self, seed=None) -> None:
        """
        Play a round-robin tournament using parallel processing.
        """
        if seed is not None:
            random.seed(seed)

        n_prisoners = len(self.prisoners)
        matches = list(combinations(range(0, n_prisoners), 2))

        # Serialize prisoners for multiprocessing
        serialized_prisoners = [serialize_prisoner(p) for p in self.prisoners]

        # Prepare game parameters
        game_tasks = []
        for i, (p1_idx, p2_idx) in enumerate(matches):
            # Generate a unique seed for each game
            game_seed = (seed or 0) + i * 1000 + p1_idx * 100 + p2_idx

            game_tasks.append((
                serialized_prisoners[p1_idx],
                serialized_prisoners[p2_idx],
                self.payoff,
                self.termination_prob,
                self.max_rounds,
                game_seed
            ))

        print(
            f"Starting tournament with {len(matches)} matches using up to {self.max_workers or 'all available'} processes...")

        # Execute games in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all games
            future_to_game = {
                executor.submit(play_single_game, *task): i
                for i, task in enumerate(game_tasks)
            }

            # Collect results with progress bar
            results = []
            for future in tqdm(as_completed(future_to_game), total=len(matches), desc="Playing matches"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    game_idx = future_to_game[future]
                    print(f'Game {game_idx} generated an exception: {exc}')
                    raise

        # Process results
        for result in results:
            key = (result.prisoner1_name, result.prisoner2_name)
            self.games_score[key] = (result.prisoner1_score, result.prisoner2_score)
            self.history[key] = result.moves_history

        # Calculate overall scores
        self._calculate_overall_scores()

    def _calculate_overall_scores(self):
        """Calculate the overall score of each prisoner based on game results."""
        # Calculate the overall score of each prisoner based on the scores after every game
        # The overall score is per move and per game
        for prisoners, scores in self.games_score.items():
            game_length = len(self.history[prisoners])
            if game_length > 0:  # Avoid division by zero
                self.prisoners_score[prisoners[0]] += scores[0] / game_length
                self.prisoners_score[prisoners[1]] += scores[1] / game_length

        # Normalize by number of games played
        n_games = len(self.games_score)
        if n_games > 0:
            for prisoner in self.prisoners_score:
                self.prisoners_score[prisoner] /= n_games


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
           OpenAI('gpt-4.1-nano_1', 'gpt-4.1-nano'))
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
    payoff: Payoff = Payoff(reward=3, punishment=1, temptation=5, sucker=0)

    # Create tournament with configurable max_workers
    # Set max_workers=None to use all available processors
    # Set max_workers=4 to limit to 4 concurrent processes, etc.
    tournament: Tournament = Tournament(
        instantiate_4_prisoners_with_AI_CB,
        payoff=payoff,
        termination_prob=0.0341,
        max_rounds=20,
        max_workers=32  # Adjust this value as needed
    )

    tournament.play_one_round_robin_tournament(seed=31415)

    print("\nGame Results:")
    for k, v in tournament.games_score.items():
        game_length = len(tournament.history[k])
        print(f'{k} : {v} after {game_length} turns')

    print("\nOverall Prisoner Scores:")
    for k, v in tournament.prisoners_score.items():
        print(f'{k} : {v:.4f}')


if __name__ == '__main__':
    main()

# TODO
"""
Indicate in the prompt to keep into consideration a comparison between the prisoner previous moves and the opponent's 
previous move, to see if it is possible to infer the opponent strategy and use the information to the prisoner advantage
"""
