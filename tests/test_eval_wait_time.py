import time
from unittest.mock import Mock, patch
from queue import Queue
from typing import List

# Import the functions we want to test
from open_instruct.grpo_fast import maybe_evaluate, vllm_generate_thread


class TestEvalWaitTime:
    """Test the eval wait time tracking functionality."""
    
    def test_eval_wait_time_tracking(self):
        """Test that eval wait time is properly tracked and logged."""
        
        # Mock wandb
        with patch('open_instruct.grpo_fast.wandb') as mock_wandb:
            mock_wandb.run = Mock()
            mock_wandb.log = Mock()
            
            # Mock args
            args = Mock()
            args.with_tracking = True
            args.num_training_steps = 10
            args.eval_freq = 2
            
            # Mock other dependencies
            tokenizer = Mock()
            tokenizer.batch_decode.return_value = ["test response"]
            tokenizer.pad_token = "<pad>"
            
            eval_prompt_token_ids: List[int] = [1, 2, 3]
            eval_ground_truths = ["test ground truth"]
            eval_dataset_names = ["test_dataset"]
            
            # Mock reward function
            async def mock_reward_fn(*args, **kwargs):
                return [1.0], {"test_metric": 1.0}
            
            reward_fn = Mock()
            reward_fn.return_value = mock_reward_fn()
            
            # Mock writer
            writer = Mock()
            
            # Create queues
            evaluation_inference_results_Q = Queue()
            
            # Test that eval wait time is tracked
            # First eval - should not log wait time since it's the first one
            maybe_evaluate(
                args=args,
                training_step=2,
                evaluation_inference_results_Q=evaluation_inference_results_Q,
                tokenizer=tokenizer,
                eval_prompt_token_ids=eval_prompt_token_ids,
                eval_ground_truths=eval_ground_truths,
                eval_dataset_names=eval_dataset_names,
                reward_fn=reward_fn,
                episode=1,
                writer=writer,
            )
            
            # Put some mock data in the queue for the eval
            evaluation_inference_results_Q.put((
                [[1, 2, 3, 4]],  # responses
                ["stop"],  # finish_reasons
                [[1, 1, 1, 1]],  # masks
                ([0], [0], [""], [""], [0], [False])  # infos
            ))
            
            # Call maybe_evaluate again to process the data
            maybe_evaluate(
                args=args,
                training_step=2,
                evaluation_inference_results_Q=evaluation_inference_results_Q,
                tokenizer=tokenizer,
                eval_prompt_token_ids=eval_prompt_token_ids,
                eval_ground_truths=eval_ground_truths,
                eval_dataset_names=eval_dataset_names,
                reward_fn=reward_fn,
                episode=1,
                writer=writer,
            )
            
            # Verify that the eval finish time was recorded
            from open_instruct.grpo_fast import last_eval_finish_time
            assert last_eval_finish_time is not None
            
            # Now test the vllm_generate_thread function
            # Mock the generate_with_engines function
            def mock_generate_with_engines(prompts, sampling_params):
                return (
                    [[1, 2, 3, 4]],  # response_ids
                    ["stop"],  # finish_reasons
                    [[1, 1, 1, 1]],  # masks
                    ([0], [0], [""], [""], [0], [False])  # infos
                )
            
            # Mock vLLM engines
            mock_engines = [Mock()]
            
            # Mock generation config
            generation_config = Mock()
            eval_generation_config = Mock()
            
            # Create queues
            inference_results_Q = Queue()
            param_prompt_Q = Queue()
            evaluation_inference_results_Q = Queue()
            
            # Put some data in the param_prompt_Q
            param_prompt_Q.put((None, [[1, 2, 3]]))
            
            # Test that the eval wait time is logged when the next eval is queued
            with patch('open_instruct.grpo_fast.generate_with_engines', mock_generate_with_engines):
                # This should trigger an eval at step 2
                vllm_generate_thread(
                    vllm_engines=mock_engines,
                    generation_config=generation_config,
                    eval_generation_config=eval_generation_config,
                    inference_results_Q=inference_results_Q,
                    param_prompt_Q=param_prompt_Q,
                    num_training_steps=5,
                    eval_prompt_token_ids=eval_prompt_token_ids,
                    evaluation_inference_results_Q=evaluation_inference_results_Q,
                    eval_freq=2,
                    resume_training_step=2,
                    tool_use=False,
                )
            
            # Verify that wandb.log was called with the eval wait time
            mock_wandb.log.assert_called()
            call_args = mock_wandb.log.call_args_list
            eval_wait_time_calls = [call for call in call_args if "eval/wait_time_between_evals" in str(call)]
            assert len(eval_wait_time_calls) > 0, "Eval wait time should be logged to wandb"