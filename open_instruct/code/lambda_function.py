import json
from typing import List
import traceback


def get_successful_tests(program: str, tests: List[str], max_execution_time: float = 1.0) -> List[int]:
    """Run a program against a list of tests, if the program exited successfully then we consider
    the test to be passed. Note that you SHOULD ONLY RUN THIS FUNCTION IN A VIRTUAL ENVIRONMENT
    as we do not guarantee the safety of the program provided.

    Parameter:
        program: a string representation of the python program you want to run
        tests: a list of assert statements which are considered to be the test cases
        max_execution_time: the number of second each individual test can run before
            it is considered failed and terminated

    Return:
        a list of 0/1 indicating passed or not"""
    test_ct = len(tests)
    if test_ct == 0:
        return []
    
    # Alternative implementation without shared memory
    results = []
    execution_context = {"__builtins__": __builtins__}
    
    print(f"Testing program: {program}")
    try:
        exec(program, execution_context)
    except Exception:
        return [0] * len(tests)
    
    print(f"Testing tests: {tests}")
    for test in tests:
        try:
            print(f"Testing test: {test}")
            # Create a fresh copy of the execution context with the program
            test_context = execution_context.copy()
            exec(test, test_context)
            results.append(1)
        except Exception:
            results.append(0)
    
    return results

def lambda_handler(event, context):
    """
    AWS Lambda handler for the test_program endpoint
    """
    try:
        request_id = context.aws_request_id
        # Parse the request body
        body = json.loads(event['body'])
        program = body.get('program', '')
        tests = body.get('tests', [])
        max_execution_time = body.get('max_execution_time', 1.0)

        # Execute tests
        results = get_successful_tests(
            program=program,
            tests=tests,
            max_execution_time=max_execution_time
        )

        print(f"Completed test run for request {request_id}")

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps({
                'results': results
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps({
                'error': str(e),
                'traceback': traceback.format_exc()
            })
        } 