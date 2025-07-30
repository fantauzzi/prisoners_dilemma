import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

# Import from your main module
from main_async import Move, Payoff, OpenAI


class MockLLMResponse:
    """Mock response object that mimics the structure of ChatOpenAI response."""

    def __init__(self, content: str, input_tokens: int = 100, output_tokens: int = 50, total_tokens: int = 150):
        self.content = content
        self.usage_metadata = {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens
        }


class TestOpenAIPrisoner:
    """Test the OpenAI prisoner class with mocked API calls."""

    @pytest.fixture
    def mock_api_key(self):
        """Mock API key for testing."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'}):
            yield 'test-api-key'

    @pytest.fixture
    def openai_prisoner(self, mock_api_key):
        """Create an OpenAI prisoner for testing."""
        with patch('main_async.ChatOpenAI') as mock_chat:
            # Mock the ChatOpenAI client
            mock_client = Mock()
            mock_chat.return_value = mock_client

            prisoner = OpenAI("TestGPT", "gpt-4o-mini")
            prisoner._llm_client = mock_client
            return prisoner

    @pytest.fixture
    def test_payoff(self):
        """Create a standard payoff matrix for testing."""
        return Payoff(reward=3, punishment=1, temptation=5, sucker=0)

    def test_openai_initialization(self, mock_api_key):
        """Test OpenAI prisoner initialization."""
        with patch('main_async.ChatOpenAI') as mock_chat:
            mock_client = Mock()
            mock_chat.return_value = mock_client

            prisoner = OpenAI("TestGPT", "gpt-4o-mini")

            # Verify ChatOpenAI was called with correct parameters
            mock_chat.assert_called_once_with(
                openai_api_key='test-api-key',
                model_name='gpt-4o-mini',
                temperature=1.0,
                max_tokens=5000
            )

            assert prisoner.name == "TestGPT"
            assert prisoner._llm_client == mock_client

    @pytest.mark.asyncio
    async def test_openai_cooperate_response(self, openai_prisoner, test_payoff):
        """Test OpenAI prisoner choosing to cooperate."""
        # Mock response that should result in cooperation
        mock_response = MockLLMResponse(
            content="I think cooperation is the best strategy here.\nC",
            input_tokens=150,
            output_tokens=25,
            total_tokens=175
        )

        # Mock the invoke method to return our mock response
        openai_prisoner._llm_client.invoke = Mock(return_value=mock_response)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            move = await openai_prisoner._choose_one_move(
                payoff=test_payoff,
                termination_prob=0.1,
                max_turns=20,
                history=[],
                opponent_history=[],
                log_file=log_file
            )

            assert move == Move.COOPERATE

            # Verify the LLM was called
            openai_prisoner._llm_client.invoke.assert_called_once()

            # Check that log file was written
            with open(log_file, 'r') as f:
                log_content = f.read()
                assert "MOVES COUNT: 1" in log_content
                assert "PROMPT/COMPLETION/TOTAL TOKENS: 150/25/175" in log_content
                assert "I think cooperation is the best strategy here." in log_content

        finally:
            os.unlink(log_file)

    @pytest.mark.asyncio
    async def test_openai_defect_response(self, openai_prisoner, test_payoff):
        """Test OpenAI prisoner choosing to defect."""
        mock_response = MockLLMResponse(
            content="Given the opponent's history, I should defect.\nD",
            input_tokens=160,
            output_tokens=30,
            total_tokens=190
        )

        openai_prisoner._llm_client.invoke = Mock(return_value=mock_response)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            move = await openai_prisoner._choose_one_move(
                payoff=test_payoff,
                termination_prob=0.05,
                max_turns=50,
                history=[Move.COOPERATE, Move.DEFECT],
                opponent_history=[Move.DEFECT, Move.DEFECT],
                log_file=log_file
            )

            assert move == Move.DEFECT

            # Check that history was properly formatted in the prompt
            call_args = openai_prisoner._llm_client.invoke.call_args
            prompt_message = call_args[0][0][0].content
            assert "You: C D" in prompt_message
            assert "Opponent: D D" in prompt_message

        finally:
            os.unlink(log_file)

    @pytest.mark.asyncio
    async def test_openai_case_insensitive_parsing(self, openai_prisoner, test_payoff):
        """Test that OpenAI prisoner handles lowercase responses correctly."""
        mock_response = MockLLMResponse(content="I'll cooperate this time.\nc")
        openai_prisoner._llm_client.invoke = Mock(return_value=mock_response)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            move = await openai_prisoner._choose_one_move(
                payoff=test_payoff,
                termination_prob=0.1,
                max_turns=20,
                history=[],
                opponent_history=[],
                log_file=log_file
            )

            assert move == Move.COOPERATE
        finally:
            os.unlink(log_file)

    @pytest.mark.asyncio
    async def test_openai_multiline_response_parsing(self, openai_prisoner, test_payoff):
        """Test parsing when decision is in a multiline response."""
        mock_response = MockLLMResponse(
            content="""Looking at the payoff matrix and history, I need to consider:
            1. The opponent has been mostly cooperative
            2. The reward for mutual cooperation is good
            3. I should reciprocate their cooperation

            Therefore, my decision is:
            C"""
        )
        openai_prisoner._llm_client.invoke = Mock(return_value=mock_response)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            move = await openai_prisoner._choose_one_move(
                payoff=test_payoff,
                termination_prob=0.1,
                max_turns=20,
                history=[],
                opponent_history=[],
                log_file=log_file
            )

            assert move == Move.COOPERATE
        finally:
            os.unlink(log_file)

    @pytest.mark.asyncio
    async def test_openai_prompt_formatting(self, openai_prisoner, test_payoff):
        """Test that the prompt is formatted correctly with all required information."""
        mock_response = MockLLMResponse(content="C")
        openai_prisoner._llm_client.invoke = Mock(return_value=mock_response)

        history = [Move.COOPERATE, Move.DEFECT, Move.COOPERATE]
        opponent_history = [Move.DEFECT, Move.COOPERATE, Move.DEFECT]

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            await openai_prisoner._choose_one_move(
                payoff=test_payoff,
                termination_prob=0.15,
                max_turns=25,
                history=history,
                opponent_history=opponent_history,
                log_file=log_file
            )

            # Extract the prompt that was sent
            call_args = openai_prisoner._llm_client.invoke.call_args
            prompt = call_args[0][0][0].content

            # Verify all expected components are in the prompt
            assert "Iterated Prisoner's Dilemma" in prompt
            assert "Reward (C,C): 3" in prompt
            assert "Punishment (D,D): 1" in prompt
            assert "Temptation (D,C): 5" in prompt
            assert "Sucker (C,D): 0" in prompt
            assert "probability 0.15" in prompt
            assert "25 turns" in prompt
            assert "You: C D C" in prompt
            assert "Opponent: D C D" in prompt
            assert "maximize your total score" in prompt
            assert "Choose your next move between cooperate and defect" in prompt

        finally:
            os.unlink(log_file)

    @pytest.mark.asyncio
    async def test_openai_empty_history_formatting(self, openai_prisoner, test_payoff):
        """Test prompt formatting when history is empty."""
        mock_response = MockLLMResponse(content="C")
        openai_prisoner._llm_client.invoke = Mock(return_value=mock_response)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            await openai_prisoner._choose_one_move(
                payoff=test_payoff,
                termination_prob=0.1,
                max_turns=20,
                history=[],
                opponent_history=[],
                log_file=log_file
            )

            call_args = openai_prisoner._llm_client.invoke.call_args
            prompt = call_args[0][0][0].content

            # Should show "none" for empty histories
            assert "You: none" in prompt
            assert "Opponent: none" in prompt

        finally:
            os.unlink(log_file)

    @pytest.mark.asyncio
    async def test_openai_invalid_response_raises_error(self, openai_prisoner, test_payoff):
        """Test that invalid responses raise ValueError."""
        mock_response = MockLLMResponse(content="I choose to cooperate\nX")  # Invalid decision
        openai_prisoner._llm_client.invoke = Mock(return_value=mock_response)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            with pytest.raises(ValueError, match="Parsed invalid decision"):
                await openai_prisoner._choose_one_move(
                    payoff=test_payoff,
                    termination_prob=0.1,
                    max_turns=20,
                    history=[],
                    opponent_history=[],
                    log_file=log_file
                )

            # Check that error was logged
            with open(log_file, 'r') as f:
                log_content = f.read()
                assert "ERROR IN PARSING DECISION: `X`" in log_content

        finally:
            os.unlink(log_file)

    @pytest.mark.asyncio
    async def test_openai_whitespace_in_response(self, openai_prisoner, test_payoff):
        """Test that responses with extra whitespace are handled correctly."""
        mock_response = MockLLMResponse(content="  \n  D  \n  ")  # Lots of whitespace
        openai_prisoner._llm_client.invoke = Mock(return_value=mock_response)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            move = await openai_prisoner._choose_one_move(
                payoff=test_payoff,
                termination_prob=0.1,
                max_turns=20,
                history=[],
                opponent_history=[],
                log_file=log_file
            )

            assert move == Move.DEFECT
        finally:
            os.unlink(log_file)

    @pytest.mark.asyncio
    async def test_openai_log_file_content(self, openai_prisoner, test_payoff):
        """Test that log file contains all expected information."""
        mock_response = MockLLMResponse(
            content="Strategic analysis here\nC",
            input_tokens=200,
            output_tokens=40,
            total_tokens=240
        )
        openai_prisoner._llm_client.invoke = Mock(return_value=mock_response)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            # Mock datetime for consistent testing
            with patch('main_async.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "2024-01-01_12-00-00"

                await openai_prisoner._choose_one_move(
                    payoff=test_payoff,
                    termination_prob=0.1,
                    max_turns=20,
                    history=[Move.COOPERATE],
                    opponent_history=[Move.DEFECT],
                    log_file=log_file
                )

            # Read and verify log content
            with open(log_file, 'r') as f:
                log_content = f.read()

            # Check all expected log components
            assert "MOVES COUNT: 2" in log_content  # len(history) + 1
            assert "TIME STAMP: 2024-01-01_12-00-00" in log_content
            assert "PROMPT/COMPLETION/TOTAL TOKENS: 200/40/240" in log_content
            assert "PROMPT:" in log_content
            assert "RESPONSE:" in log_content
            assert "Strategic analysis here" in log_content
            assert "Iterated Prisoner's Dilemma" in log_content  # Part of prompt
            assert "-----------------------------------------------" in log_content

        finally:
            os.unlink(log_file)

    @pytest.mark.asyncio
    async def test_openai_different_model_names(self, mock_api_key):
        """Test that different model names are handled correctly."""
        models_to_test = ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]

        for model_name in models_to_test:
            with patch('main_async.ChatOpenAI') as mock_chat:
                mock_client = Mock()
                mock_chat.return_value = mock_client

                prisoner = OpenAI(f"Test_{model_name}", model_name)

                # Verify correct model was requested
                mock_chat.assert_called_once_with(
                    openai_api_key='test-api-key',
                    model_name=model_name,
                    temperature=1.0,
                    max_tokens=5000
                )

    @pytest.mark.asyncio
    async def test_openai_async_execution(self, openai_prisoner, test_payoff):
        """Test that the OpenAI prisoner works correctly in async context."""
        mock_response = MockLLMResponse(content="D")

        # Mock asyncio.to_thread to simulate async execution
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = mock_response

            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
                log_file = temp_log.name

            try:
                move = await openai_prisoner._choose_one_move(
                    payoff=test_payoff,
                    termination_prob=0.1,
                    max_turns=20,
                    history=[],
                    opponent_history=[],
                    log_file=log_file
                )

                assert move == Move.DEFECT
                # Verify asyncio.to_thread was called
                mock_to_thread.assert_called_once()

            finally:
                os.unlink(log_file)

    @pytest.mark.asyncio
    async def test_openai_usage_metadata_missing(self, openai_prisoner, test_payoff):
        """Test handling when usage metadata is missing or incomplete."""
        # Create response with incomplete metadata
        mock_response = Mock()
        mock_response.content = "C"
        mock_response.usage_metadata = {}  # Empty metadata

        openai_prisoner._llm_client.invoke = Mock(return_value=mock_response)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            move = await openai_prisoner._choose_one_move(
                payoff=test_payoff,
                termination_prob=0.1,
                max_turns=20,
                history=[],
                opponent_history=[],
                log_file=log_file
            )

            assert move == Move.COOPERATE

            # Check that N/A is logged for missing metadata
            with open(log_file, 'r') as f:
                log_content = f.read()
                assert "PROMPT/COMPLETION/TOTAL TOKENS: N/A/N/A/N/A" in log_content

        finally:
            os.unlink(log_file)


class TestOpenAIPrisonerEdgeCases:
    """Test edge cases and error conditions for OpenAI prisoner."""

    @pytest.fixture
    def openai_prisoner(self):
        """Create an OpenAI prisoner with mocked client."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            with patch('main_async.ChatOpenAI') as mock_chat:
                mock_client = Mock()
                mock_chat.return_value = mock_client
                prisoner = OpenAI("EdgeCaseGPT", "gpt-4o-mini")
                prisoner._llm_client = mock_client
                return prisoner

    @pytest.fixture
    def test_payoff(self):
        return Payoff(reward=3, punishment=1, temptation=5, sucker=0)

    @pytest.mark.asyncio
    async def test_openai_empty_response(self, openai_prisoner, test_payoff):
        """Test handling of completely empty response."""
        mock_response = MockLLMResponse(content="")
        openai_prisoner._llm_client.invoke = Mock(return_value=mock_response)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            with pytest.raises(ValueError):
                await openai_prisoner._choose_one_move(
                    payoff=test_payoff,
                    termination_prob=0.1,
                    max_turns=20,
                    history=[],
                    opponent_history=[],
                    log_file=log_file
                )
        finally:
            os.unlink(log_file)

    @pytest.mark.asyncio
    async def test_openai_only_whitespace_response(self, openai_prisoner, test_payoff):
        """Test handling of response with only whitespace."""
        mock_response = MockLLMResponse(content="   \n\t  \n  ")
        openai_prisoner._llm_client.invoke = Mock(return_value=mock_response)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_log:
            log_file = temp_log.name

        try:
            with pytest.raises(ValueError):
                await openai_prisoner._choose_one_move(
                    payoff=test_payoff,
                    termination_prob=0.1,
                    max_turns=20,
                    history=[],
                    opponent_history=[],
                    log_file=log_file
                )
        finally:
            os.unlink(log_file)
