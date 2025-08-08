import pytest
import asyncio
import os
import tempfile
from unittest.mock import Mock, AsyncMock, patch, mock_open
from pathlib import Path

# Import the classes we need to test
from ipd import LLM, Move, Payoff, _rate_limiter_registry


class TestLLM:
    """Test suite for the LLM class."""

    @pytest.fixture
    def payoff(self):
        """Standard payoff matrix for tests."""
        return Payoff(reward=3, punishment=1, temptation=5, sucker=0)

    @pytest.fixture
    def mock_chat_client(self):
        """Mock ChatOpenAI/ChatAnthropic client."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = "I will cooperate to build trust.\nC"
        mock_response.usage_metadata = {
            'input_tokens': 150,
            'output_tokens': 25,
            'total_tokens': 175
        }
        mock_client.ainvoke.return_value = mock_response
        return mock_client

    @pytest.fixture
    def temp_log_file(self):
        """Create a temporary log file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            yield f.name
        # Cleanup
        try:
            os.unlink(f.name)
        except FileNotFoundError:
            pass

    @pytest.fixture(autouse=True)
    def setup_env_vars(self):
        """Set up environment variables for testing."""
        with patch.dict(os.environ, {
            'CHATOPENAI_API_KEY': 'test-openai-key',
            'CHATANTHROPIC_API_KEY': 'test-anthropic-key'
        }):
            yield

    @pytest.fixture(autouse=True)
    def reset_rate_limiter_registry(self):
        """Reset the global rate limiter registry between tests."""
        _rate_limiter_registry._limiters.clear()
        yield
        _rate_limiter_registry._limiters.clear()

    def test_llm_initialization_openai(self):
        """Test LLM initialization with OpenAI client."""
        with patch('ipd.globals') as mock_globals:
            mock_chat_class = Mock()
            mock_globals.return_value = {'ChatOpenAI': mock_chat_class}

            llm = LLM(
                llm_client='ChatOpenAI',
                name='test-gpt',
                model_name='gpt-4',
                temperature=0.7,
                max_tokens=100,
                max_rate_per_minute=60
            )

            assert llm.name == 'test-gpt'
            assert llm.llm_client == 'ChatOpenAI'
            assert llm.max_rate_per_minute == 60
            assert llm.client_key == 'ChatOpenAI_gpt-4'
            mock_chat_class.assert_called_once_with(
                api_key='test-openai-key',
                model_name='gpt-4',
                temperature=0.7,
                max_tokens=100
            )

    def test_llm_initialization_anthropic(self):
        """Test LLM initialization with Anthropic client."""
        with patch('ipd.globals') as mock_globals:
            mock_chat_class = Mock()
            mock_globals.return_value = {'ChatAnthropic': mock_chat_class}

            llm = LLM(
                llm_client='ChatAnthropic',
                name='test-claude',
                model_name='claude-3-haiku',
                temperature=1.0,
                max_tokens=500,
                max_rate_per_minute=30
            )

            assert llm.name == 'test-claude'
            assert llm.llm_client == 'ChatAnthropic'
            assert llm.max_rate_per_minute == 30
            assert llm.client_key == 'ChatAnthropic_claude-3-haiku'

    def test_llm_initialization_missing_api_key(self):
        """Test that LLM raises ValueError when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Please set the CHATOPENAI_API_KEY environment variable"):
                LLM(
                    llm_client='ChatOpenAI',
                    name='test-gpt',
                    model_name='gpt-4',
                    temperature=0.7,
                    max_tokens=100
                )

    @pytest.mark.asyncio
    async def test_choose_one_move_cooperate(self, payoff, temp_log_file, mock_chat_client):
        """Test choosing a COOPERATE move."""
        with patch('ipd.globals') as mock_globals:
            mock_globals.return_value = {'ChatOpenAI': lambda **kwargs: mock_chat_client}

            llm = LLM(
                llm_client='ChatOpenAI',
                name='test-gpt',
                model_name='gpt-4',
                temperature=0.7,
                max_tokens=100
            )
            llm._llm_client = mock_chat_client

            move = await llm._choose_one_move(
                payoff=payoff,
                termination_prob=0.1,
                max_turns=10,
                history=[],
                opponent_history=[],
                log_file=temp_log_file
            )

            assert move == Move.COOPERATE
            mock_chat_client.ainvoke.assert_called_once()

            # Verify log file contents
            with open(temp_log_file, 'r') as f:
                log_content = f.read()
                assert 'MOVES COUNT: 1' in log_content
                assert 'RATE LIMIT: 9999 req/min for ChatOpenAI_gpt-4' in log_content
                assert 'PROMPT/COMPLETION/TOTAL TOKENS: 150/25/175' in log_content

    @pytest.mark.asyncio
    async def test_choose_one_move_defect(self, payoff, temp_log_file, mock_chat_client):
        """Test choosing a DEFECT move."""
        # Modify mock response to return defect
        mock_chat_client.ainvoke.return_value.content = "Strategic defection is optimal here.\nD"

        with patch('ipd.globals') as mock_globals:
            mock_globals.return_value = {'ChatOpenAI': lambda **kwargs: mock_chat_client}

            llm = LLM(
                llm_client='ChatOpenAI',
                name='test-gpt',
                model_name='gpt-4',
                temperature=0.7,
                max_tokens=100
            )
            llm._llm_client = mock_chat_client

            move = await llm._choose_one_move(
                payoff=payoff,
                termination_prob=0.1,
                max_turns=10,
                history=[Move.COOPERATE],
                opponent_history=[Move.DEFECT],
                log_file=temp_log_file
            )

            assert move == Move.DEFECT

    @pytest.mark.asyncio
    async def test_choose_one_move_with_history(self, payoff, temp_log_file, mock_chat_client):
        """Test move selection with game history."""
        with patch('ipd.globals') as mock_globals:
            mock_globals.return_value = {'ChatOpenAI': lambda **kwargs: mock_chat_client}

            llm = LLM(
                llm_client='ChatOpenAI',
                name='test-gpt',
                model_name='gpt-4',
                temperature=0.7,
                max_tokens=100
            )
            llm._llm_client = mock_chat_client

            history = [Move.COOPERATE, Move.DEFECT, Move.COOPERATE]
            opponent_history = [Move.DEFECT, Move.COOPERATE, Move.DEFECT]

            await llm._choose_one_move(
                payoff=payoff,
                termination_prob=0.05,
                max_turns=20,
                history=history,
                opponent_history=opponent_history,
                log_file=temp_log_file
            )

            # Verify the prompt contains history
            call_args = mock_chat_client.ainvoke.call_args[0][0][0].content
            assert 'Previous moves — You: C D C. Opponent: D C D.' in call_args
            assert 'This is turn number 4.' in call_args

    @pytest.mark.asyncio
    async def test_choose_one_move_empty_history(self, payoff, temp_log_file, mock_chat_client):
        """Test move selection with empty history."""
        with patch('ipd.globals') as mock_globals:
            mock_globals.return_value = {'ChatOpenAI': lambda **kwargs: mock_chat_client}

            llm = LLM(
                llm_client='ChatOpenAI',
                name='test-gpt',
                model_name='gpt-4',
                temperature=0.7,
                max_tokens=100
            )
            llm._llm_client = mock_chat_client

            await llm._choose_one_move(
                payoff=payoff,
                termination_prob=0.1,
                max_turns=10,
                history=[],
                opponent_history=[],
                log_file=temp_log_file
            )

            # Verify the prompt contains "none" for empty history
            call_args = mock_chat_client.ainvoke.call_args[0][0][0].content
            assert 'Previous moves — You: none. Opponent: none.' in call_args
            assert 'This is turn number 1.' in call_args

    @pytest.mark.asyncio
    async def test_invalid_move_response(self, payoff, temp_log_file, mock_chat_client):
        """Test handling of invalid move responses."""
        # Mock response with invalid move
        mock_chat_client.ainvoke.return_value.content = "I'm confused.\nX"

        with patch('ipd.globals') as mock_globals:
            mock_globals.return_value = {'ChatOpenAI': lambda **kwargs: mock_chat_client}

            llm = LLM(
                llm_client='ChatOpenAI',
                name='test-gpt',
                model_name='gpt-4',
                temperature=0.7,
                max_tokens=100
            )
            llm._llm_client = mock_chat_client

            with pytest.raises(ValueError, match="Parsed invalid decision"):
                await llm._choose_one_move(
                    payoff=payoff,
                    termination_prob=0.1,
                    max_turns=10,
                    history=[],
                    opponent_history=[],
                    log_file=temp_log_file
                )

            # Verify error is logged
            with open(temp_log_file, 'r') as f:
                log_content = f.read()
                assert 'ERROR IN PARSING DECISION: `X`' in log_content

    @pytest.mark.asyncio
    async def test_empty_response(self, payoff, temp_log_file, mock_chat_client):
        """Test handling of empty response."""
        mock_chat_client.ainvoke.return_value.content = ""

        with patch('ipd.globals') as mock_globals:
            mock_globals.return_value = {'ChatOpenAI': lambda **kwargs: mock_chat_client}

            llm = LLM(
                llm_client='ChatOpenAI',
                name='test-gpt',
                model_name='gpt-4',
                temperature=0.7,
                max_tokens=100
            )
            llm._llm_client = mock_chat_client

            with pytest.raises(ValueError, match="Parsed invalid decision"):
                await llm._choose_one_move(
                    payoff=payoff,
                    termination_prob=0.1,
                    max_turns=10,
                    history=[],
                    opponent_history=[],
                    log_file=temp_log_file
                )

    @pytest.mark.asyncio
    async def test_case_insensitive_moves(self, payoff, temp_log_file, mock_chat_client):
        """Test that moves are case insensitive."""
        mock_chat_client.ainvoke.return_value.content = "Cooperating for mutual benefit.\nc"

        with patch('ipd.globals') as mock_globals:
            mock_globals.return_value = {'ChatOpenAI': lambda **kwargs: mock_chat_client}

            llm = LLM(
                llm_client='ChatOpenAI',
                name='test-gpt',
                model_name='gpt-4',
                temperature=0.7,
                max_tokens=100
            )
            llm._llm_client = mock_chat_client

            move = await llm._choose_one_move(
                payoff=payoff,
                termination_prob=0.1,
                max_turns=10,
                history=[],
                opponent_history=[],
                log_file=temp_log_file
            )

            assert move == Move.COOPERATE

    @pytest.mark.asyncio
    async def test_multiline_response_parsing(self, payoff, temp_log_file, mock_chat_client):
        """Test parsing moves from multiline responses."""
        mock_chat_client.ainvoke.return_value.content = """Based on the game theory analysis,
        I believe cooperation is the optimal strategy.
        This will maximize long-term payoffs.

        C"""

        with patch('ipd.globals') as mock_globals:
            mock_globals.return_value = {'ChatOpenAI': lambda **kwargs: mock_chat_client}

            llm = LLM(
                llm_client='ChatOpenAI',
                name='test-gpt',
                model_name='gpt-4',
                temperature=0.7,
                max_tokens=100
            )
            llm._llm_client = mock_chat_client

            move = await llm._choose_one_move(
                payoff=payoff,
                termination_prob=0.1,
                max_turns=10,
                history=[],
                opponent_history=[],
                log_file=temp_log_file
            )

            assert move == Move.COOPERATE

    @pytest.mark.asyncio
    async def test_rate_limiting(self, payoff, temp_log_file):
        """Test that rate limiting is applied correctly."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = "Strategic thinking required.\nD"
        mock_response.usage_metadata = {'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110}
        mock_client.ainvoke.return_value = mock_response

        with patch('ipd.globals') as mock_globals:
            mock_globals.return_value = {'ChatOpenAI': lambda **kwargs: mock_client}

            llm = LLM(
                llm_client='ChatOpenAI',
                name='test-gpt',
                model_name='gpt-4',
                temperature=0.7,
                max_tokens=100,
                max_rate_per_minute=30
            )
            llm._llm_client = mock_client

            # Test that rate limiter is created with correct parameters
            limiter = _rate_limiter_registry.get_limiter('ChatOpenAI_gpt-4', 30)
            assert limiter.max_rate == 30

            move = await llm._choose_one_move(
                payoff=payoff,
                termination_prob=0.1,
                max_turns=10,
                history=[],
                opponent_history=[],
                log_file=temp_log_file
            )

            assert move == Move.DEFECT

    @pytest.mark.asyncio
    async def test_api_exception_handling(self, payoff, temp_log_file):
        """Test handling of API exceptions through the parent choose_one_move method."""
        mock_client = AsyncMock()
        mock_client.ainvoke.side_effect = Exception("API Error")

        with patch('ipd.globals') as mock_globals:
            mock_globals.return_value = {'ChatOpenAI': lambda **kwargs: mock_client}

            llm = LLM(
                llm_client='ChatOpenAI',
                name='test-gpt',
                model_name='gpt-4',
                temperature=0.7,
                max_tokens=100
            )
            llm._llm_client = mock_client

            # Test with max_API_call_attempts = 1
            result = await llm.choose_one_move(
                payoff=payoff,
                termination_prob=0.1,
                max_turns=10,
                history=[],
                opponent_history=[],
                log_file=temp_log_file,
                max_API_call_attempts=1
            )

            assert result is None  # Should return None after exhausting attempts

    @pytest.mark.asyncio
    async def test_prompt_content_verification(self, payoff, temp_log_file, mock_chat_client):
        """Test that the prompt contains all expected content."""
        with patch('ipd.globals') as mock_globals:
            mock_globals.return_value = {'ChatOpenAI': lambda **kwargs: mock_chat_client}

            llm = LLM(
                llm_client='ChatOpenAI',
                name='test-gpt',
                model_name='gpt-4',
                temperature=0.7,
                max_tokens=100
            )
            llm._llm_client = mock_chat_client

            await llm._choose_one_move(
                payoff=payoff,
                termination_prob=0.0341,
                max_turns=15,
                history=[Move.COOPERATE, Move.DEFECT],
                opponent_history=[Move.COOPERATE, Move.COOPERATE],
                log_file=temp_log_file
            )

            # Get the actual prompt
            call_args = mock_chat_client.ainvoke.call_args[0][0][0].content

            # Verify all expected components are in the prompt
            assert 'Iterated Prisoner\'s Dilemma' in call_args
            assert 'Reward (C,C): 3' in call_args
            assert 'Punishment (D,D): 1' in call_args
            assert 'Temptation (D,C): 5' in call_args
            assert 'Sucker (C,D): 0' in call_args
            assert 'terminates with probability 0.0341' in call_args
            assert 'not last more than 15 turns' in call_args
            assert 'This is turn number 3' in call_args
            assert 'You: C D. Opponent: C C' in call_args
            assert 'maximize your total score' in call_args
            assert '`C` to cooperate or `D` to defect' in call_args

    def test_rate_limiter_registry_singleton(self):
        """Test that the rate limiter registry works as expected."""
        # Get limiter for same client/model should return same instance
        limiter1 = _rate_limiter_registry.get_limiter('ChatOpenAI_gpt-4', 60)
        limiter2 = _rate_limiter_registry.get_limiter('ChatOpenAI_gpt-4', 60)
        assert limiter1 is limiter2

        # Different client/model should return different instance
        limiter3 = _rate_limiter_registry.get_limiter('ChatAnthropic_claude', 30)
        assert limiter3 is not limiter1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])