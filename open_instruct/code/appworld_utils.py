import logging
import traceback
from typing import Optional, List

from appworld import AppWorld
from appworld.task import load_task_ids

logger = logging.getLogger(__name__)


def resolve_task_id(task_id: Optional[str], dataset_name: str) -> str:
    """Resolve the task ID, either from input or by selecting from dataset."""
    if task_id is None:
        try:
            task_ids = load_task_ids(dataset_name)
            if not task_ids:
                raise ValueError(f"No tasks found in dataset '{dataset_name}'")
            return task_ids[0]
        except Exception as e:
            logger.error(f"Error loading task IDs: {e}")
            raise
    else:
        return task_id


def extract_task_info(world: AppWorld, task_id: str) -> dict:
    """Extract task information from AppWorld instance."""
    try:
        task_info = {
            "task_id": task_id,
            "instruction": world.task.instruction,
            "supervisor": {
                "first_name": world.task.supervisor.first_name,
                "last_name": world.task.supervisor.last_name,
                "email": world.task.supervisor.email,
            }
        }
        return task_info
    except Exception as e:
        logger.error(f"Error extracting task info: {e}")
        raise


def run_evaluation(world: AppWorld) -> Optional[dict]:
    """Run AppWorld evaluation and return structured results."""
    try:
        evaluation = world.evaluate()
        result = {
            "success": evaluation.success,
            "pass_count": evaluation.pass_count,
            "fail_count": evaluation.fail_count,
            "num_tests": evaluation.num_tests
        }
        return result
    except Exception as eval_error:
        logger.warning(f"Evaluation failed: {eval_error}")
        return {"error": str(eval_error)}


def run_custom_tests(world: AppWorld, tests: Optional[List[str]]) -> List[dict]:
    """Run custom tests in the AppWorld environment."""
    if not tests:
        return []
    
    custom_test_results = []
    for i, test in enumerate(tests):
        try:
            test_output = world.execute(test)
            success = "Execution successful" in test_output or not test_output.startswith("Execution failed")
            custom_test_results.append({
                "test": test,
                "output": test_output,
                "success": success
            })
        except Exception as test_error:
            custom_test_results.append({
                "test": test,
                "output": str(test_error),
                "success": False
            })
    return custom_test_results


def format_response(task_info: dict, execution_output: str, task_completed: bool, 
                   evaluation_result: Optional[dict], custom_test_results: List[dict], 
                   error: Optional[str] = None) -> dict:
    """Format the final response dictionary."""
    try:
        response = {
            "task_info": task_info,
            "execution_output": execution_output,
            "task_completed": task_completed,
            "evaluation": evaluation_result,
            "custom_tests": custom_test_results,
            "success": task_completed and (evaluation_result is None or evaluation_result.get("success", False)),
            **({"error": error} if error else {})
        }
        return response
    except Exception as e:
        logger.error(f"Error formatting response: {e}")
        # Return a minimal response if formatting fails
        return {
            "task_info": {"task_id": "unknown", "instruction": "unknown", "supervisor": {}},
            "execution_output": str(execution_output) if execution_output else "unknown",
            "task_completed": False,
            "evaluation": None,
            "custom_tests": [],
            "success": False,
            "error": f"Response formatting failed: {e}"
        }


def execute_program_in_world(world: AppWorld, program: str, tests: Optional[List[str]]) -> dict:
    """Execute the program in AppWorld and handle all related logic."""
    try:
        task_info = extract_task_info(world, world.task_id)
        
        # Execute the main program
        execution_output = world.execute(program)
        
        # Check if task was completed
        task_completed = world.task_completed()
        
        # Run evaluation if task completed
        evaluation_result = None
        if task_completed:
            evaluation_result = run_evaluation(world)
        
        # Run custom tests
        custom_test_results = run_custom_tests(world, tests)
        
        result = format_response(
            task_info, execution_output, task_completed, 
            evaluation_result, custom_test_results
        )
        return result
        
    except Exception as exec_error:
        logger.error(f"Error in execute_program_in_world: {exec_error}")
        
        # Try to get task info even if execution failed
        try:
            task_info = extract_task_info(world, world.task_id)
        except Exception as task_info_error:
            logger.error(f"Failed to extract task info after error: {task_info_error}")
            task_info = {"task_id": world.task_id, "instruction": "Failed to extract", "supervisor": {}}
            
        return format_response(
            task_info, str(exec_error), False, None, [], str(exec_error)
        )


