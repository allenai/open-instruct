# from RestrictedPython import compile_restricted

    
def format_function_code(function_code):

    if "```python" in function_code:
        function_code = function_code.split("```python")[1].replace("```python", "").split("```")[0].strip()
    else:
        function_code = function_code.strip()
    
    return function_code



def run_test_cases(function_code, test_cases):
    """
    Generic test runner that executes a Python function against given test cases.
    
    Args:
        function_code (str): String containing the Python function definition
        test_cases (list): List of test case assertions as strings
    
    Returns:
        bool: True if all tests pass, False if any test fails or encounters an error
    """
    # Create a namespace to store the function
    namespace = {}

    for name in ['exec', 'exit', 'quit']:
        if name in function_code:
            return False

    
    try:
        # Execute the function code in the namespace
        # byte_code = compile_restricted(
        #     function_code,
        #     filename='<inline code>',
        #     mode='exec'
        # )
        exec(function_code, namespace)
        
        # Run each test case
        for test in test_cases:
            try:
                # Execute the test assertion in the namespace
                # compile_restricted(test, '<test>', 'exec')
                exec(test, namespace)
            except AssertionError:
                print(f"Test failed: {test}")
                return False
            except Exception as e:
                print(f"Error executing test: {test}")
                print(f"Error message: {str(e)}")
                return False
                
        return True
        
    except Exception as e:
        print(f"Error in function definition:")
        print(f"Error message: {str(e)}")
        return False


# from RestrictedPython.Guards import safer_getattr, guarded_iter_unpack_sequence
# from RestrictedPython.PrintCollector import PrintCollector
# from RestrictedPython.Eval import default_guarded_getiter, default_guarded_getitem

# def run_test_cases(function_code, test_cases):
#     """
#     Generic test runner that executes a Python function against given test cases.
    
#     Args:
#         function_code (str): String containing the Python function definition
#         test_cases (list): List of test case assertions as strings
    
#     Returns:
#         bool: True if all tests pass, False if any test fails or encounters an error
#     """
#     # Create a namespace to store the function
#     namespace = {}
#     safe_builtins = {
#         # Basic types and constructors
#         'bool': bool,
#         'int': int,
#         'float': float,
#         'str': str,
#         'list': list,
#         'dict': dict,
#         'tuple': tuple,
#         'set': set,
#         'frozenset': frozenset,
        
#         # Built-in functions
#         'abs': abs,
#         'all': all,
#         'any': any,
#         'ascii': ascii,
#         'bin': bin,
#         'chr': chr,
#         'divmod': divmod,
#         'enumerate': enumerate,
#         'filter': filter,
#         'format': format,
#         'getattr': getattr,
#         'hasattr': hasattr,
#         'hash': hash,
#         'hex': hex,
#         'id': id,
#         'isinstance': isinstance,
#         'issubclass': issubclass,
#         'iter': iter,
#         'len': len,
#         'map': map,
#         'max': max,
#         'min': min,
#         'next': next,
#         'oct': oct,
#         'ord': ord,
#         'pow': pow,
#         'range': range,
#         'repr': repr,
#         'reversed': reversed,
#         'round': round,
#         'slice': slice,
#         'sorted': sorted,
#         'sum': sum,
#         'zip': zip,
        
#         # Constants
#         'True': True,
#         'False': False,
#         'None': None,
        
#         # Exception handling
#         'BaseException': BaseException,
#         'Exception': Exception,
#         'TypeError': TypeError,
#         'ValueError': ValueError,
#         'ArithmeticError': ArithmeticError,
#         'AssertionError': AssertionError,
        
#         # Built-in modules
#         '__import__': __import__,
#     }
#     restricted_globals = {
#         '__builtins__': safe_builtins,
#         '_getattr_': safer_getattr,  # Add the safer_getattr guard
#         '_print_': PrintCollector,  # Optional: for handling print statements
#         '_getiter_': default_guarded_getiter,  # Changed to recommended guard
#         '_iter_unpack_sequence_': guarded_iter_unpack_sequence,  # Added sequence unpacking guard
#         '_getitem_': default_guarded_getitem,  # Added back getitem guard
#     }

#     # namespace = restricted_globals.copy()
#     namespace = {}
    
#     try:
#         # Execute the function code in the namespace
#         byte_code = compile_restricted(
#             function_code,
#             filename='<inline code>',
#             mode='exec'
#         )
#         exec(function_code, namespace)
        
#         # Run each test case
#         for test in test_cases:
#             try:
#                 # Execute the test assertion in the namespace
#                 exec(compile_restricted(test, '<test>', 'exec'), namespace)
#             except AssertionError:
#                 print(f"Test failed: {test}")
#                 return False
#             except Exception as e:
#                 print(f"Error executing test: {test}")
#                 print(f"Error message: {str(e)}")
#                 return False
                
#         return True
        
#     except Exception as e:
#         print(f"Error in function definition:")
#         print(f"Error message: {str(e)}")
#         return False