import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path

# Import the classes we need to test
from main_async import LLM, Move, Payoff


class MockLLMResponse:
    """Mock response object that mimics the structure of LangChain LLM responses"""

    def __init__(self, content: str, input_tokens: int = 100, output_tokens: int = 50):
        self.content = content
        self.usage_metadata = {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens
        }


class TestLLM:
    """Test suite for the LLM class"""

    @pytest.fixture
    def payoff(self):
        """Standard payoff matrix for testing"""
        return Payoff(reward=3.0, punishment=1.0, temptation=5.0, sucker=0.0)

    @pytest.fixture
    def temp_log_file(self):
        """Create a temporary log file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            yield f.name
        # Cleanup
        if os.path.exists(f.name):
            os.unlink(f.name)

    @pytest.fixture
    def mock_openai_env(self):
        """Mock environment variable for OpenAI API key"""
        with patch.dict(os.environ, {'CHATOPENAI_API_KEY': 'test-key'}):
            yield

    @pytest.fixture
    def mock_anthropic_env(self):
        """Mock environment variable for Anthropic API key"""
        with patch.dict(os.environ, {'CHATANTHROPIC_API_KEY': 'test-key'}):
            yield

    @patch('main_async.load_dotenv')
    @patch('main_async.ChatOpenAI')
    def test_llm_initialization_openai(self, mock_chat_openai, mock_load_dotenv, mock_openai_env):
        """Test LLM initialization with OpenAI client"""
        mock_client = Mock()
        mock_chat_openai.return_value = mock_client

        llm = LLM(
            llm_client='ChatOpenAI',
            name='test_llm',
            model_name='gpt-4o-mini',
            temperature=0.7,
            max_tokens=500
        )

        assert llm.name == 'test_llm'
        assert llm._llm_client == mock_client
        mock_chat_openai.assert_called_once_with(
            api_key='test-key',
            model_name='gpt-4o-mini',
            temperature=0.7,
            max_tokens=500
        )

    @patch('main_async.load_dotenv')
    @patch('main_async.ChatAnthropic')
    def test_llm_initialization_anthropic(self, mock_chat_anthropic, mock_load_dotenv, mock_anthropic_env):
        """Test LLM initialization with Anthropic client"""
        mock_client = Mock()
        mock_chat_anthropic.return_value = mock_client

        llm = LLM(
            llm_client='ChatAnthropic',
            name='test_claude',
            model_name='claude-3-5-haiku-20241022',
            temperature=1.0,
            max_tokens=500
        )

        assert llm.name == 'test_claude'
        assert llm._llm_client == mock_client
        mock_chat_anthropic.assert_called_once_with(
            api_key='test-key',
            model_name='claude-3-5-haiku-20241022',
            temperature=1.0,
            max_tokens=500
        )

    @patch('main_async.load_dotenv')
    def test_llm_initialization_missing_api_key(self, mock_load_dotenv):
        """Test that LLM raises error when API key is missing"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Please set the CHATOPENAI_API_KEY environment variable"):
                LLM(
                    llm_client='ChatOpenAI',
                    name='test_llm',
                    model_name='gpt-4o-mini',
                    temperature=0.7,
                    max_tokens=500
                )

    @patch('main_async.load_dotenv')
    @patch('main_async.ChatOpenAI')
    @pytest.mark.asyncio
    async def test_choose_one_move_cooperate(self, mock_chat_openai, mock_load_dotenv, mock_openai_env, payoff,
                                             temp_log_file):
        """Test LLM choosing to cooperate"""
        # Setup mock
        mock_client = Mock()
        mock_response = MockLLMResponse("I should cooperate to build trust.\n\nC")
        mock_client.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_client

        # Create LLM instance
        llm = LLM('ChatOpenAI', 'test_llm', 'gpt-4o-mini', 0.7, 500)

        # Mock asyncio.to_thread to return the mock response directly
        with patch('asyncio.to_thread', return_value=mock_response):
            move = await llm._choose_one_move(
                payoff=payoff,
                termination_prob=0.1,
                max_turns=10,
                history=[],
                opponent_history=[],
                log_file=temp_log_file
            )

        assert move == Move.COOPERATE

        # Verify log file was written
        with open(temp_log_file, 'r') as f:
            log_content = f.read()
            assert 'MOVES COUNT: 1' in log_content
            assert 'C' in log_content

    @patch('main_async.load_dotenv')
    @patch('main_async.ChatOpenAI')
    @pytest.mark.asyncio
    async def test_choose_one_move_defect(self, mock_chat_openai, mock_load_dotenv, mock_openai_env, payoff,
                                          temp_log_file):
        """Test LLM choosing to defect"""
        # Setup mock
        mock_client = Mock()
        mock_response = MockLLMResponse("I should defect to maximize my score.\n\nD")
        mock_client.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_client

        # Create LLM instance
        llm = LLM('ChatOpenAI', 'test_llm', 'gpt-4o-mini', 0.7, 500)

        # Mock asyncio.to_thread
        with patch('asyncio.to_thread', return_value=mock_response):
            move = await llm._choose_one_move(
                payoff=payoff,
                termination_prob=0.1,
                max_turns=10,
                history=[Move.COOPERATE],
                opponent_history=[Move.DEFECT],
                log_file=temp_log_file
            )

        assert move == Move.DEFECT

    @patch('main_async.load_dotenv')
    @patch('main_async.ChatOpenAI')
    @pytest.mark.asyncio
    async def test_choose_one_move_invalid_response(self, mock_chat_openai, mock_load_dotenv, mock_openai_env, payoff,
                                                    temp_log_file):
        """Test LLM handling invalid response format"""
        # Setup mock with invalid response
        mock_client = Mock()
        mock_response = MockLLMResponse("I'm confused.\n\nX")  # Invalid move
        mock_client.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_client

        # Create LLM instance
        llm = LLM('ChatOpenAI', 'test_llm', 'gpt-4o-mini', 0.7, 500)

        # Mock asyncio.to_thread
        with patch('asyncio.to_thread', return_value=mock_response):
            with pytest.raises(ValueError, match="Parsed invalid decision"):
                await llm._choose_one_move(
                    payoff=payoff,
                    termination_prob=0.1,
                    max_turns=10,
                    history=[],
                    opponent_history=[],
                    log_file=temp_log_file
                )

    @patch('main_async.load_dotenv')
    @patch('main_async.ChatOpenAI')
    @pytest.mark.asyncio
    async def test_choose_one_move_empty_response(self, mock_chat_openai, mock_load_dotenv, mock_openai_env, payoff,
                                                  temp_log_file):
        """Test LLM handling empty response"""
        # Setup mock with empty response
        mock_client = Mock()
        mock_response = MockLLMResponse("")  # Empty response
        mock_client.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_client

        # Create LLM instance
        llm = LLM('ChatOpenAI', 'test_llm', 'gpt-4o-mini', 0.7, 500)

        # Mock asyncio.to_thread
        with patch('asyncio.to_thread', return_value=mock_response):
            with pytest.raises(ValueError, match="Parsed invalid decision"):
                await llm._choose_one_move(
                    payoff=payoff,
                    termination_prob=0.1,
                    max_turns=10,
                    history=[],
                    opponent_history=[],
                    log_file=temp_log_file
                )

    @patch('main_async.load_dotenv')
    @patch('main_async.ChatOpenAI')
    @pytest.mark.asyncio
    async def test_choose_one_move_case_insensitive(self, mock_chat_openai, mock_load_dotenv, mock_openai_env, payoff,
                                                    temp_log_file):
        """Test that LLM handles lowercase responses correctly"""
        # Setup mock with lowercase response
        mock_client = Mock()
        mock_response = MockLLMResponse("Let me cooperate.\n\nc")  # lowercase 'c'
        mock_client.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_client

        # Create LLM instance
        llm = LLM('ChatOpenAI', 'test_llm', 'gpt-4o-mini', 0.7, 500)

        # Mock asyncio.to_thread
        with patch('asyncio.to_thread', return_value=mock_response):
            move = await llm._choose_one_move(
                payoff=payoff,
                termination_prob=0.1,
                max_turns=10,
                history=[],
                opponent_history=[],
                log_file=temp_log_file
            )

        assert move == Move.COOPERATE

    @patch('main_async.load_dotenv')
    @patch('main_async.ChatOpenAI')
    @pytest.mark.asyncio
    async def test_prompt_construction_first_move(self, mock_chat_openai, mock_load_dotenv, mock_openai_env, payoff,
                                                  temp_log_file):
        """Test that the prompt is constructed correctly for the first move"""
        # Setup mock
        mock_client = Mock()
        mock_response = MockLLMResponse("First move, let's cooperate.\n\nC")
        mock_client.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_client

        # Create LLM instance
        llm = LLM('ChatOpenAI', 'test_llm', 'gpt-4o-mini', 0.7, 500)

        # Mock asyncio.to_thread to actually call the lambda function
        def mock_to_thread(func):
            return func()

        with patch('asyncio.to_thread', side_effect=mock_to_thread):
            await llm._choose_one_move(
                payoff=payoff,
                termination_prob=0.1,
                max_turns=10,
                history=[],
                opponent_history=[],
                log_file=temp_log_file
            )

        # Check that invoke was called with correct message structure
        invoke_call = mock_client.invoke.call_args[0][0]
        assert len(invoke_call) == 1
        prompt_content = invoke_call[0].content

        # Verify key elements in prompt
        assert "Iterated Prisoner's Dilemma" in prompt_content
        assert "Reward (C,C): 3.0" in prompt_content
        assert "Punishment (D,D): 1.0" in prompt_content
        assert "Temptation (D,C): 5.0" in prompt_content
        assert "Sucker (C,D): 0.0" in prompt_content
        assert "turn number 1" in prompt_content
        assert "You: none" in prompt_content
        assert "Opponent: none" in prompt_content

    @patch('main_async.load_dotenv')
    @patch('main_async.ChatOpenAI')
    @pytest.mark.asyncio
    async def test_logging_functionality(self, mock_chat_openai, mock_load_dotenv, mock_openai_env, payoff,
                                         temp_log_file):
        """Test that logging works correctly"""
        # Setup mock
        mock_client = Mock()
        mock_response = MockLLMResponse(
            "Let me think about this carefully.\n\nC",
            input_tokens=150,
            output_tokens=75
        )
        mock_client.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_client

        # Create LLM instance
        llm = LLM('ChatOpenAI', 'test_llm', 'gpt-4o-mini', 0.7, 500)

        # Mock asyncio.to_thread and datetime for deterministic timestamps
        with patch('asyncio.to_thread', return_value=mock_response), \
                patch('main_async.get_time_stamp', return_value='2024-01-01_12-00-00'):
            await llm._choose_one_move(
                payoff=payoff,
                termination_prob=0.1,
                max_turns=10,
                history=[Move.COOPERATE],
                opponent_history=[Move.COOPERATE],
                log_file=temp_log_file
            )

        # Read and verify log content
        with open(temp_log_file, 'r') as f:
            log_content = f.read()

        assert 'MOVES COUNT: 2' in log_content
        assert 'TIME STAMP: 2024-01-01_12-00-00' in log_content
        assert 'PROMPT/COMPLETION/TOTAL TOKENS: 150/75/225' in log_content
        assert 'PROMPT:' in log_content
        assert 'RESPONSE:' in log_content
        assert 'Let me think about this carefully.' in log_content
        assert '-----------------------------------------------' in log_content

    @patch('main_async.load_dotenv')
    @patch('main_async.ChatOpenAI')
    @pytest.mark.asyncio
    async def test_error_logging_for_invalid_decision(self, mock_chat_openai, mock_load_dotenv, mock_openai_env, payoff,
                                                      temp_log_file):
        """Test that invalid decisions are logged as errors"""
        # Setup mock with invalid response
        mock_client = Mock()
        mock_response = MockLLMResponse("I'm confused.\n\nMAYBE")
        mock_client.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_client

        # Create LLM instance
        llm = LLM('ChatOpenAI', 'test_llm', 'gpt-4o-mini', 0.7, 500)

        # Mock asyncio.to_thread
        with patch('asyncio.to_thread', return_value=mock_response):
            with pytest.raises(ValueError):
                await llm._choose_one_move(
                    payoff=payoff,
                    termination_prob=0.1,
                    max_turns=10,
                    history=[],
                    opponent_history=[],
                    log_file=temp_log_file
                )

        # Verify error was logged
        with open(temp_log_file, 'r') as f:
            log_content = f.read()

        assert 'ERROR IN PARSING DECISION: `MAYBE`' in log_content

    @patch('main_async.load_dotenv')
    @patch('main_async.ChatOpenAI')
    @pytest.mark.asyncio
    async def test_choose_one_move_with_retries(self, mock_chat_openai, mock_load_dotenv, mock_openai_env, payoff,
                                                temp_log_file):
        """Test the retry mechanism in choose_one_move"""
        # Setup mock that fails twice then succeeds
        mock_client = Mock()
        mock_chat_openai.return_value = mock_client

        # Create LLM instance
        llm = LLM('ChatOpenAI', 'test_llm', 'gpt-4o-mini', 0.7, 500)

        # Mock _choose_one_move to fail twice then succeed
        call_count = 0

        async def mock_choose_one_move(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("API error")
            return Move.COOPERATE

        llm._choose_one_move = mock_choose_one_move

        # Test the retry logic
        move = await llm.choose_one_move(
            payoff=payoff,
            termination_prob=0.1,
            max_turns=10,
            history=[],
            opponent_history=[],
            log_file=temp_log_file
        )

        assert move == Move.COOPERATE
        assert call_count == 3  # Failed twice, succeeded on third attempt

    @patch('main_async.load_dotenv')
    @patch('main_async.ChatOpenAI')
    @pytest.mark.asyncio
    async def test_choose_one_move_max_retries_exceeded(self, mock_chat_openai, mock_load_dotenv, mock_openai_env,
                                                        payoff, temp_log_file):
        """Test behavior when max retries are exceeded"""
        # Setup mock
        mock_client = Mock()
        mock_chat_openai.return_value = mock_client

        # Create LLM instance
        llm = LLM('ChatOpenAI', 'test_llm', 'gpt-4o-mini', 0.7, 500)

        # Mock _choose_one_move to always fail
        async def mock_choose_one_move(*args, **kwargs):
            raise Exception("Persistent API error")

        llm._choose_one_move = mock_choose_one_move

        # Test that None is returned after max attempts
        with patch('main_async.max_API_call_attempts', 3):
            move = await llm.choose_one_move(
                payoff=payoff,
                termination_prob=0.1,
                max_turns=10,
                history=[],
                opponent_history=[],
                log_file=temp_log_file
            )

        assert move is None

    @patch('main_async.load_dotenv')
    @patch('main_async.ChatAnthropic')
    @pytest.mark.asyncio
    async def test_anthropic_client_integration(self, mock_chat_anthropic, mock_load_dotenv, mock_anthropic_env, payoff,
                                                temp_log_file):
        """Test that Anthropic client works correctly"""
        # Setup mock
        mock_client = Mock()
        mock_response = MockLLMResponse("I'll cooperate this turn.\n\nC")
        mock_client.invoke.return_value = mock_response
        mock_chat_anthropic.return_value = mock_client

        # Create LLM instance with Anthropic
        llm = LLM('ChatAnthropic', 'test_claude', 'claude-3-5-haiku-20241022', 1.0, 500)

        # Mock asyncio.to_thread
        with patch('asyncio.to_thread', return_value=mock_response):
            move = await llm._choose_one_move(
                payoff=payoff,
                termination_prob=0.1,
                max_turns=10,
                history=[],
                opponent_history=[],
                log_file=temp_log_file
            )

        assert move == Move.COOPERATE

    @patch('main_async.load_dotenv')
    @patch('main_async.ChatOpenAI')
    @pytest.mark.asyncio
    async def test_whitespace_handling_in_response(self, mock_chat_openai, mock_load_dotenv, mock_openai_env, payoff,
                                                   temp_log_file):
        """Test that whitespace in responses is handled correctly"""
        # Setup mock with extra whitespace
        mock_client = Mock()
        mock_response = MockLLMResponse("Let me decide.\n\n  D  \n\n")  # Extra whitespace around D
        mock_client.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_client

        # Create LLM instance
        llm = LLM('ChatOpenAI', 'test_llm', 'gpt-4o-mini', 0.7, 500)

        # Mock asyncio.to_thread
        with patch('asyncio.to_thread', return_value=mock_response):
            move = await llm._choose_one_move(
                payoff=payoff,
                termination_prob=0.1,
                max_turns=10,
                history=[],
                opponent_history=[],
                log_file=temp_log_file
            )

        assert move == Move.DEFECT

    @patch('main_async.load_dotenv')
    @patch('main_async.ChatOpenAI')
    @pytest.mark.asyncio
    async def test_multiline_response_parsing(self, mock_chat_openai, mock_load_dotenv, mock_openai_env, payoff,
                                              temp_log_file):
        """Test parsing of multi-line responses"""
        # Setup mock with multi-line response
        mock_client = Mock()
        mock_response = MockLLMResponse(
            "Given the current situation, I need to analyze:\n"
            "1. The opponent's previous moves\n"
            "2. The expected value of cooperation vs defection\n"
            "3. The probability of game continuation\n\n"
            "After careful consideration, my move is:\n"
            "D"
        )
        mock_client.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_client

        # Create LLM instance
        llm = LLM('ChatOpenAI', 'test_llm', 'gpt-4o-mini', 0.7, 500)

        # Mock asyncio.to_thread
        with patch('asyncio.to_thread', return_value=mock_response):
            move = await llm._choose_one_move(
                payoff=payoff,
                termination_prob=0.1,
                max_turns=10,
                history=[],
                opponent_history=[],
                log_file=temp_log_file
            )

        assert move == Move.DEFECT

    @patch('main_async.load_dotenv')
    def test_llm_inheritance(self, mock_load_dotenv):
        """Test that LLM properly inherits from Prisoner"""
        with patch.dict(os.environ, {'CHATOPENAI_API_KEY': 'test-key'}), \
                patch('main_async.ChatOpenAI'):
            llm = LLM('ChatOpenAI', 'test_llm', 'gpt-4o-mini', 0.7, 500)

            # Test that it has the required abstract method
            assert hasattr(llm, '_choose_one_move')
            assert hasattr(llm, 'choose_one_move')
            assert hasattr(llm, 'name')
            assert llm.name == 'test_llm'

    @patch('main_async.load_dotenv')
    @patch('main_async.ChatOpenAI')
    @pytest.mark.asyncio
    async def test_token_usage_logging(self, mock_chat_openai, mock_load_dotenv, mock_openai_env, payoff,
                                       temp_log_file):
        """Test that token usage is logged correctly"""
        # Setup mock with specific token counts
        mock_client = Mock()
        mock_response = MockLLMResponse(
            "Analysis complete.\n\nC",
            input_tokens=200,
            output_tokens=25
        )
        mock_client.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_client

        # Create LLM instance
        llm = LLM('ChatOpenAI', 'test_llm', 'gpt-4o-mini', 0.7, 500)

        # Mock asyncio.to_thread
        with patch('asyncio.to_thread', return_value=mock_response):
            await llm._choose_one_move(
                payoff=payoff,
                termination_prob=0.1,
                max_turns=10,
                history=[],
                opponent_history=[],
                log_file=temp_log_file
            )

        # Verify token usage is logged
        with open(temp_log_file, 'r') as f:
            log_content = f.read()

        assert 'PROMPT/COMPLETION/TOTAL TOKENS: 200/25/225' in log_content

    @patch('main_async.load_dotenv')
    @patch('main_async.ChatOpenAI')
    @pytest.mark.asyncio
    async def test_missing_usage_metadata(self, mock_chat_openai, mock_load_dotenv, mock_openai_env, payoff,
                                          temp_log_file):
        """Test handling of missing usage metadata"""
        # Setup mock without usage metadata
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = "I choose to cooperate.\n\nC"
        mock_response.usage_metadata = {}  # Empty usage metadata
        mock_client.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_client

        # Create LLM instance
        llm = LLM('ChatOpenAI', 'test_llm', 'gpt-4o-mini', 0.7, 500)

        # Mock asyncio.to_thread
        with patch('asyncio.to_thread', return_value=mock_response):
            move = await llm._choose_one_move(
                payoff=payoff,
                termination_prob=0.1,
                max_turns=10,
                history=[],
                opponent_history=[],
                log_file=temp_log_file
            )

        assert move == Move.COOPERATE

        # Verify N/A is logged for missing token info
        with open(temp_log_file, 'r') as f:
            log_content = f.read()

        assert 'PROMPT/COMPLETION/TOTAL TOKENS: N/A/N/A/N/A' in log_content


# Configuration fixture for tests that might need it
@pytest.fixture
def test_config_content():
    """Sample configuration content for testing"""
    return """
common:
  termination_probability: 0.1
  max_rounds: 10
  max_API_call_attempts: 5
  max_concurrent_matches: 4
  payoff: { reward: 3, punishment: 1, temptation: 5, sucker: 0 }
prisoners:
  - { prisoner_class: 'LLM', params: { llm_client: 'ChatOpenAI',
                                       name: 'test_llm',
                                       model_name: 'gpt-4o-mini',
                                       temperature: 1,
                                       max_tokens: 500 } }
"""


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
