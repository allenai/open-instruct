"""This script is used to create Azure OpenAI batch API requests for code editing across multiple datasets.

Usage:

Cd into the directory of this file and run:
```
python batch_code_edit.py --datasets "user/dataset1,user/dataset2" --num-errors 3
```

"""

import argparse
import dataclasses
import json
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List

import datasets
import openai
import tiktoken

# Set up logging with file name and line number
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Model costs are in USD per million tokens.
MODEL_COSTS_PER_1M_TOKENS = {
    "o3-batch": {
        "input": 10 * 0.5,
        "output": 40 * 0.5,
    },
    "o3": {
        "input": 10 * 0.5,
        "output": 40 * 0.5,
    },
}

# Maximum number of prompts per batch file
MAX_PROMPTS_PER_BATCH = 50_000

# Dataset-specific column mappings
# TODO: Fill out these mappings for each dataset
DATASET_COLUMN_MAPPINGS = {
    "saurabh5/llama-nemotron-rlvr": {
        "code_column": "rewritten_solution",
        "test_column": "ground_truth",
        "description_index_fn": lambda x: x,
        "description_column": "rewritten_input",
        "id_column": "id",
        "language": "python",
    },
    "saurabh5/the-algorithm-python": {
        "code_column": "reference_solution",
        "test_column": "ground_truth",
        "description_column": "messages",
        "description_index_fn": lambda x: x[0]["content"],
        "id_column": "python_index",
        "language": "python",
    },
    "saurabh5/open-code-reasoning-rlvr": {
        "code_column": "rewritten_solution",
        "test_column": "ground_truth",
        "description_column": "input",
        "id_column": "id",
        "language": "python",
    },
    "saurabh5/tulu-3-personas-code-rlvr": {
        "code_column": "rewritten_solution",
        "test_column": "ground_truth",
        "description_column": "rewritten_input",
        "id_column": "id",
        "language": "python",
    },
    "saurabh5/rlvr-code-data": {
        "code_column": "translated_solution",
        "test_column": "ground_truth",
        "description_column": "translated_problem",
        "id_column": "id",
    },
}


@dataclasses.dataclass
class CodeError:
    tag: str
    prompt: str


SUPPORTED_LANGUAGES = ["python", "javascript", "java", "cpp", "rust", "bash",
                       "go", "swift", "kotlin", "haskell", "lean", "typescript"]

# Language-specific coding errors
LANGUAGE_ERRORS: dict[str, list[CodeError]] = {
    "python": [
        CodeError(
            tag="eq-asgn",
            prompt="Replace a '==' equality test in a conditional expression with a single '=' assignment.",
        ),
        CodeError(
            tag="offby1",
            prompt="Adjust a loop boundary (e.g., change range(n) to range(n + 1) or the reverse) so the loop runs one iteration off.",
        ),
        CodeError(
            tag="miss-colon",
            prompt="Remove the trailing colon from a control-flow or function header (if/for/while/def/class).",
        ),
        CodeError(
            tag="unclose-par",
            prompt="Delete a closing parenthesis from a multi-token expression, leaving it unmatched.",
        ),
        CodeError(
            tag="bad-indent",
            prompt="Dedent an inner code block (such as a loop or conditional body) by two spaces, breaking its alignment.",
        ),
        CodeError(
            tag="str-quotes",
            prompt="Change a closing double quote \") to a single quote ' in a string literal, leaving the opening quote unchanged.",
        ),
        CodeError(
            tag="illegal-kw",
            prompt="Rename an identifier to a reserved keyword (for example, change a variable name to 'class').",
        ),
        CodeError(
            tag="ill-comment",
            prompt="Insert a stray '#' mid-line so the remainder becomes an unintended comment.",
        ),
        CodeError(
            tag="var-uninit",
            prompt="Delete the assignment of exactly one variable that is later used, leaving it uninitialised.",
        ),
        CodeError(
            tag="none-deref",
            prompt="Assign None to an object immediately before calling one of its methods.",
        ),
        CodeError(
            tag="idx-oob",
            prompt="Shift an index expression by +1 (e.g., arr[i] → arr[i+1]), risking an out-of-bounds access.",
        ),
        CodeError(
            tag="div-zero",
            prompt="Introduce a division by zero (e.g., replace x / y with x / 0).",
        ),
        CodeError(
            tag="inf-loop",
            prompt="Modify an existing loop condition so it never becomes false, creating an infinite loop.",
        ),
        CodeError(
            tag="long-loop",
            prompt="Double the iteration count of an existing loop (such as multiplying its upper bound by 2).",
        ),
        CodeError(
            tag="float-prec",
            prompt="Insert an equality comparison between two floats that differ only by floating-point precision error.",
        ),
        CodeError(
            tag="mut-default",
            prompt="Give a function a mutable default argument like {} or [] instead of None.",
        ),
        CodeError(
            tag="slow-loop",
            prompt="Invoke an expensive computation twice within the same iteration of a nested loop.",
        ),
        CodeError(
            tag="magic-num",
            prompt="Replace a variable or parameter with a hard-coded constant (a 'magic number') reused in multiple places.",
        ),
    ],
    "javascript": [
        CodeError(tag="missing-import", prompt="Remove a required import/require statement."),
        CodeError(tag="undefined-var", prompt="Replace a variable with an undefined one."),
        CodeError(tag="ill-decl", prompt="Change a function declaration to an assignment."),
        CodeError(tag="miss-semi", prompt="Delete a semicolon."),
        CodeError(tag="bad-class", prompt="Replace a class declaration with a variable assignment."),
        CodeError(tag="miss-ret", prompt="Remove the return statement."),
        CodeError(tag="ill-arr", prompt="Change an array literal to a function call."),
        CodeError(tag="global-leak", prompt="Drop let/const, leaking the variable to global scope."),
        CodeError(tag="str-concat", prompt="Swap a template literal for \"+\" string concatenation."),
        CodeError(tag="loose-eq", prompt="Replace === with == in a comparison."),
        CodeError(tag="assign-in-cond", prompt="Substitute = for === inside an if condition."),
        CodeError(tag="miss-await", prompt="Call an async function without await."),
        CodeError(tag="unhandled-promise", prompt="Remove the catch() from a promise chain."),
        CodeError(tag="callback-err-ignore", prompt="Omit the err argument in a Node-style callback."),
        CodeError(tag="missing-break", prompt="Delete the break in a switch case."),
        CodeError(tag="offby1", prompt="Change loop bound i < n to i <= n."),
        CodeError(tag="NaN-comp", prompt="Compare a value to NaN with ===."),
        CodeError(tag="parseInt-no-radix", prompt="Call parseInt(str) without the radix."),
        CodeError(tag="for-in-array", prompt="Iterate over an array with for…in instead of for…of."),
        CodeError(tag="hoist-use-before-def", prompt="Use a variable before declaring it."),
        CodeError(tag="idx-oob", prompt="Replace arr[i] with arr[i + 1]."),
        CodeError(tag="json-circular", prompt="JSON.stringify a self-referential object."),
        CodeError(tag="insecure-eval", prompt="Pass raw user input to eval()."),
        CodeError(tag="typo-length", prompt="Misspell .length as .lenght."),
        CodeError(tag="const-reassign", prompt="Reassign a const variable."),
        CodeError(tag="shadow-var", prompt="Declare an inner variable that hides an outer one."),
        CodeError(tag="ill-comment", prompt="Insert an unterminated /* comment."),
        CodeError(tag="array-copy-ref", prompt="Assign one array to another with = instead of slice()."),
        CodeError(tag="float-prec", prompt="Compare two floats with ==."),
        CodeError(tag="miss-brace", prompt="Delete a closing } from a block."),
        CodeError(tag="inf-loop", prompt="Remove the increment in a while loop, creating an infinite loop."),
    ],
    "java": [
        CodeError(tag="missing-import", prompt="Add a missing import statement."),
        CodeError(
            tag="undefined-var", prompt="Replace a variable with an undefined one."
        ),
        CodeError(
            tag="ill-decl",
            prompt="Change a method declaration to a variable assignment.",
        ),
        CodeError(tag="miss-semi", prompt="Insert a missing semicolon."),
        CodeError(
            tag="bad-class",
            prompt="Change a class declaration to a variable assignment.",
        ),
        CodeError(tag="miss-ret", prompt="Remove the return statement."),
        CodeError(tag="ill-arr", prompt="Change an array literal to a function call."),
        CodeError(tag="miss-access", prompt="Insert a missing access modifier."),
        CodeError(
            tag="str-concat",
            prompt="Swap one string concatenation for a template literal.",
        ),
        CodeError(
            tag="miss-try", prompt="Insert a try/catch block around the function."
        ),
        CodeError(tag="eq-asgn", prompt="Replace == with = inside an if condition."),
        CodeError(tag="offby1", prompt="Change loop bound i < n to i <= n."),
        CodeError(tag="str-eq", prompt="Compare two strings with == instead of .equals()."),
        CodeError(tag="miss-brace", prompt="Delete a closing } from a method or control block."),
        CodeError(tag="unclose-par", prompt="Remove a closing parenthesis from a method call."),
        CodeError(tag="nullptr", prompt="Set obj = null just before calling obj.method()."),
        CodeError(tag="idx-oob", prompt="Replace arr[i] with arr[i + 1]."),
        CodeError(tag="var-uninit", prompt="Declare int x; and read x before assigning it."),
        CodeError(tag="ill-comment", prompt="Insert an unterminated /* comment."),
        CodeError(tag="illegal-kw", prompt="Rename a local variable to a reserved word like enum."),
        CodeError(tag="res-leak", prompt="Replace a try-with-resources block with manual new FileInputStream(...) and omit close()."),
        CodeError(tag="magic-num", prompt="Inline the constant 0.75 multiple times."),
        CodeError(tag="path-traversal", prompt="Pass raw user input to new File(userInputPath)."),
        CodeError(tag="n2-slop", prompt="Move an expensive query() call inside a nested loop, creating O(N²) work."),
        CodeError(tag="inf-loop", prompt="Remove the increment i++ from a while (i < n) loop."),
        CodeError(tag="div-zero", prompt="Compute value / (den - den) after setting den at runtime."),
        CodeError(tag="shadow-var", prompt="Declare a variable inside a method that hides an outer variable."),
        CodeError(tag="shallow-copy", prompt="Assign one array to another with = instead of calling .clone()."),
        CodeError(tag="float-prec", prompt="Compare two double values with ==."),
        CodeError(tag="mutable-static", prompt="Make a static List<?> cache and mutate it from multiple threads without synchronization."),
    ],
    "cpp": [
        CodeError(
            tag="eq-asgn",
            prompt="Replace == with = inside an if condition.",
        ),
        CodeError(
            tag="offby1",
            prompt="Change loop bound i < n to i <= n.",
        ),
        CodeError(
            tag="missing-semi",
            prompt="Delete the semicolon at the end of a statement.",
        ),
        CodeError(
            tag="unclosed-brace",
            prompt="Remove the closing } of a function or class.",
        ),
        CodeError(
            tag="unclose-par",
            prompt="Drop the closing ) of a function call.",
        ),
        CodeError(
            tag="bad-break",
            prompt="Omit break; in one case of a switch.",
        ),
        CodeError(
            tag="nullptr",
            prompt="Set a pointer to nullptr just before dereferencing it.",
        ),
        CodeError(
            tag="idx-oob",
            prompt="Change vec[i] to vec[i + 1].",
        ),
        CodeError(
            tag="var-uninit",
            prompt="Declare int x; and read x before assigning it.",
        ),
        CodeError(
            tag="double-delete",
            prompt="delete the same pointer twice.",
        ),
        CodeError(
            tag="mem-leak",
            prompt="new an object but never delete it.",
        ),
        CodeError(
            tag="dangling-ref",
            prompt="Return a reference to a local variable.",
        ),
        CodeError(
            tag="signed-mix",
            prompt="Compare int vs size_t without cast.",
        ),
        CodeError(
            tag="int-overflow",
            prompt="Multiply two large ints without widening.",
        ),
        CodeError(
            tag="shadow-var",
            prompt="Declare a local size that hides the member size.",
        ),
        CodeError(
            tag="const-corrupt",
            prompt="Remove const from a parameter and mutate it.",
        ),
        CodeError(
            tag="race-cond",
            prompt="Eliminate the mutex lock around shared data.",
        ),
        CodeError(
            tag="exc-leak",
            prompt="Throw inside a constructor without RAII guards.",
        ),
        CodeError(
            tag="iter-invalid",
            prompt="erase() an element from a vector while range-iterating it.",
        ),
        CodeError(
            tag="float-prec",
            prompt="Compare two double values with ==.",
        ),
        CodeError(
            tag="ub-reinterpret",
            prompt="Cast an object to an unrelated type with reinterpret_cast.",
        ),
        CodeError(
            tag="magic-num",
            prompt="Inline the constant 0.75 multiple times.",
        ),
    ],
    "rust": [
        CodeError(
            tag="eq-asgn",
            prompt="Replace == with = inside an if (won't compile—ownership tests still catch it).",
        ),
        CodeError(
            tag="offby1",
            prompt="Change loop bound i < n to i <= n in a for/while.",
        ),
        CodeError(
            tag="missing-semi",
            prompt="Delete the trailing ; so an expression becomes the function's return.",
        ),
        CodeError(
            tag="unclosed-brace",
            prompt="Drop the closing } of a match arm block.",
        ),
        CodeError(
            tag="unclose-par",
            prompt="Remove a ) from a function call.",
        ),
        CodeError(
            tag="bad-break",
            prompt="Omit break inside a loop { … }, causing an infinite loop.",
        ),
        CodeError(
            tag="nullptr",
            prompt="Use std::ptr::null() and then unsafe { *ptr }.",
        ),
        CodeError(
            tag="idx-oob",
            prompt="Change vec[i] to vec[i + 1] (runtime panic).",
        ),
        CodeError(
            tag="var-uninit",
            prompt="Declare let x: i32; and use x before assignment (compile-error).",
        ),
        CodeError(
            tag="move-after",
            prompt="Move let v = s; and later read s again.",
        ),
        CodeError(
            tag="immut-mut",
            prompt="Hold an immutable borrow let r = &val; then call val.push(…) without ending r.",
        ),
        CodeError(
            tag="borrow-alias",
            prompt="Borrow &mut x twice concurrently in the same scope.",
        ),
        CodeError(
            tag="raw-ub",
            prompt="unsafe { *(0xdead as *const i32) } to cause undefined behavior.",
        ),
        CodeError(
            tag="unwrap-err",
            prompt="Call .unwrap() on an Err result.",
        ),
        CodeError(
            tag="expect-none",
            prompt=".expect('val') on a None option.",
        ),
        CodeError(
            tag="match-miss",
            prompt="Write a match with one enum variant unhandled, fallback to _ => unreachable!().",
        ),
        CodeError(
            tag="panic-div0",
            prompt="Divide by 0 using integer types.",
        ),
        CodeError(
            tag="float-prec",
            prompt="Compare two f64 values with ==.",
        ),
        CodeError(tag="string-slice", prompt="Slice &s[0..1] on a UTF-8 multi-byte char (runtime panic)."),
        CodeError(
            tag="shadow-var", prompt="let count = …; … { let count = 0; … } hides the outer count."
        ),
        CodeError(
            tag="magic-num",
            prompt="Hard-code 0.75 multiple times instead of a const.",
        ),
        CodeError(
            tag="mem-leak",
            prompt="Call std::mem::forget(obj) so Drop isn't run.",
        ),
        CodeError(tag="error-ignore", prompt="Assign let _ = fallible_call(); and ignore the Result."),
    ],
    "bash": [
        CodeError(
            tag="unquoted-expansion",
            prompt="Remove the double-quotes around \"$var\" so it becomes $var.",
        ),
        CodeError(
            tag="glob-accident",
            prompt="Replace a literal path \"$file\" with $file so any * and ? expand.",
        ),
        CodeError(
            tag="offby1-seq",
            prompt="Change seq 0 $((n-1)) to seq 1 $n.",
        ),
        CodeError(
            tag="wrong-test",
            prompt="Swap [[ $a -eq $b ]] for [ $a == $b ] (string-vs-integer mix-up).",
        ),
        CodeError(
            tag="backtick-old",
            prompt="Convert $(cmd) to obsolete `cmd`.",
        ),
        CodeError(
            tag="pipe-fail",
            prompt="Delete set -o pipefail, letting the left side of pipes fail silently.",
        ),
        CodeError(
            tag="unset-var",
            prompt="Remove set -u, allowing reads of undeclared variables.",
        ),
        CodeError(
            tag="ifs-leak",
            prompt="Insert IFS=$'\\n' at function start and forget to restore it.",
        ),
        CodeError(
            tag="subshell-loss",
            prompt="Wrap a critical assignment in ( … ), losing the value outside.",
        ),
        CodeError(
            tag="tmp-unsafe",
            prompt="Replace mktemp with tmp=/tmp/file.$$ (race-condition temp file).",
        ),
        CodeError(
            tag="command-inject",
            prompt="Build eval \"$cmd $user_input\" from unchecked $user_input.",
        ),
        CodeError(
            tag="null-glob",
            prompt="Enable shopt -s nullglob but assume globs always return something.",
        ),
        CodeError(
            tag="array-as-string",
            prompt="Output echo $arr instead of echo \"${arr[@]}\".",
        ),
        CodeError(
            tag="bad-shebang",
            prompt="Use #!/bin/sh yet rely on Bash-only features like [[ ]].",
        ),
        CodeError(
            tag="var-global",
            prompt="Omit local before a variable inside a function, clobbering globals.",
        ),
        CodeError(
            tag="missing-exit",
            prompt="Drop exit 1 after a detected fatal error, script continues.",
        ),
        CodeError(
            tag="piped-err-ignore",
            prompt="somecmd | grep foo without || exit—errors swallowed.",
        ),
        CodeError(
            tag="2>&1-order",
            prompt="Write cmd 2>&1 >out (stderr still goes to tty).",
        ),
        CodeError(
            tag="trap-sigint",
            prompt="Set trap 'cleanup $1' INT inside a function that lacks $1.",
        ),
        CodeError(
            tag="line-endings",
            prompt="Paste a CRLF line, adding hidden ^M that breaks bash.",
        ),
        CodeError(
            tag="missing-quotes-path",
            prompt="rm $dir/* where $dir might contain spaces.",
        ),
        CodeError(
            tag="here-doc-tabs",
            prompt="Use tabs before the delimiter in a here-doc, making it never terminate.",
        ),
        CodeError(
            tag="arith-str",
            prompt="Do count=$((count + \"one\"))—string in arithmetic context.",
        ),
        CodeError(
            tag="race-mkdir",
            prompt="mkdir /tmp/job without -p or mktemp, risking concurrent runs.",
        ),
    ],
    "go": [
        CodeError(tag="missing-import", prompt="Remove a needed import line."),
        CodeError(tag="undefined-var", prompt="Replace a variable with an undefined identifier."),
        CodeError(tag="ill-decl", prompt="Change a func declaration into a var assignment."),
        CodeError(tag="miss-comma", prompt="Delete a trailing comma in a multi-line composite literal."),
        CodeError(tag="miss-ret", prompt="Omit the required return statement from a non-void function."),
        CodeError(tag="nil-deref", prompt="Set a pointer to nil immediately before using it."),
        CodeError(tag="idx-oob", prompt="Replace slice[i] with slice[i+1]."),
        CodeError(tag="slice-copy-ref", prompt="Assign one slice to another with = and then mutate it."),
        CodeError(tag="shadow-var", prompt="Declare an inner variable that hides an outer one."),
        CodeError(tag="err-ignore", prompt="Call a function and discard the returned error."),
        CodeError(tag="race-access", prompt="Remove the mutex around a shared variable used by goroutines."),
        CodeError(tag="deadlock-chan", prompt="Block both sender and receiver on an unbuffered channel."),
        CodeError(tag="miss-go", prompt="Call a long-running function without the go keyword."),
        CodeError(tag="leak-goroutine", prompt="Launch a goroutine in a loop that captures the loop variable."),
        CodeError(tag="panic-assert", prompt="Use a type assertion without checking the ok value."),
        CodeError(tag="wrong-fmt", prompt="Print a string with %d instead of %s."),
        CodeError(tag="float-prec", prompt="Compare two float64 values with ==."),
        CodeError(tag="miss-defer-close", prompt="Open a file and forget defer file.Close()."),
        CodeError(tag="miss-struct-field", prompt="Omit a required field in a struct composite literal."),
        CodeError(tag="global-leak", prompt="Declare a package-level var that should be local."),
        CodeError(tag="ill-arr", prompt="Replace a slice literal with a meaningless function call."),
        CodeError(tag="fallthrough-misuse", prompt="Add an unnecessary fallthrough in a switch case."),
        CodeError(tag="ill-comment", prompt="Insert an unterminated /* comment."),
        CodeError(tag="magic-num", prompt="Inline the constant 0.75 multiple times instead of using a const."),
        CodeError(tag="offby1", prompt="Change loop condition i < n to i <= n."),
        CodeError(tag="unclose-par", prompt="Remove a closing parenthesis from a function call."),
        CodeError(tag="eq-asgn", prompt="Use = in an if condition where a comparison was intended."),
        CodeError(tag="chan-close-send", prompt="Close a channel and then send to it."),
        CodeError(tag="inf-loop", prompt="Remove the increment in for i < n {}, creating an infinite loop."),
    ],
    "swift": [
        CodeError(tag="missing-import", prompt="Delete a needed import statement."),
        CodeError(tag="undefined-var", prompt="Replace a variable with an undefined identifier."),
        CodeError(tag="ill-decl", prompt="Change a func declaration into a let assignment."),
        CodeError(tag="miss-ret", prompt="Omit the required return statement from a non-Void function."),
        CodeError(tag="miss-try", prompt="Call a throwing function without try/try? or try!."),
        CodeError(tag="force-unwrap", prompt="Add a ! to force-unwrap an optional that might be nil."),
        CodeError(tag="optional-chain", prompt="Replace optional chaining with force-unwraps."),
        CodeError(tag="mutating-miss", prompt="Modify a struct property inside a non-mutating method."),
        CodeError(tag="idx-oob", prompt="Replace array[i] with array[i + 1]."),
        CodeError(tag="offby1", prompt="Change loop bound i < n to i <= n."),
        CodeError(tag="inf-loop", prompt="Remove the increment in while i < n {}, creating an infinite loop."),
        CodeError(tag="nil-deref", prompt="Set an optional to nil just before accessing it."),
        CodeError(tag="shadow-var", prompt="Declare an inner variable that hides an outer one."),
        CodeError(tag="weak-miss", prompt="Capture self strongly in a escaping closure, causing a retain cycle."),
        CodeError(tag="unowned-use", prompt="Use an unowned reference after the object has deallocated."),
        CodeError(tag="dispatch-deadlock", prompt="Call DispatchQueue.main.sync inside code already on the main queue."),
        CodeError(tag="err-ignore", prompt="Use try! instead of handling the thrown error."),
        CodeError(tag="panic-assert", prompt="Force-cast with as! without checking the result."),
        CodeError(tag="float-prec", prompt="Compare two Double values with ==."),
        CodeError(tag="string-eq", prompt="Compare Strings with === instead of ==."),
        CodeError(tag="miss-default", prompt="Omit the default case in a switch over a non-exhaustive enum."),
        CodeError(tag="fallthrough-misuse", prompt="Add an unnecessary fallthrough in a switch case."),
        CodeError(tag="miss-breakpoint", prompt="Forget break inside a for-case pattern, executing unintended code."),
        CodeError(tag="miss-defer-close", prompt="Open a file and omit defer file.close()."),
        CodeError(tag="global-leak", prompt="Declare a global var that should be local."),
        CodeError(tag="magic-num", prompt="Inline the constant 0.75 multiple times."),
        CodeError(tag="ill-comment", prompt="Insert an unterminated /* comment."),
        CodeError(tag="res-leak", prompt="Create a Timer without invalidating it, leaking the resource."),
        CodeError(tag="path-traversal", prompt="Pass raw user input to FileManager.default.contents(atPath:)."),
        CodeError(tag="eq-asgn", prompt="Use = inside an if condition where == was intended (requires var, compiles with warning)."),
    ],
    "kotlin": [
        CodeError(tag="missing-import", prompt="Delete a needed import statement."),
        CodeError(tag="undefined-var", prompt="Replace a variable with an unresolved identifier."),
        CodeError(tag="ill-decl", prompt="Change a fun declaration into a val assignment."),
        CodeError(tag="miss-ret", prompt="Omit the required return statement from a non-Unit function."),
        CodeError(tag="force-unwrap", prompt="Add !! to a nullable value that may be null."),
        CodeError(tag="null-deref", prompt="Set a variable to null just before calling a method via !!."),
        CodeError(tag="idx-oob", prompt="Replace list[i] with list[i + 1]."),
        CodeError(tag="offby1", prompt="Change loop bound i < n to i <= n."),
        CodeError(tag="var-over-val", prompt="Declare a var where a val would suffice."),
        CodeError(tag="shadow-var", prompt="Declare an inner variable that hides an outer one."),
        CodeError(tag="lateinit-not-init", prompt="Access a lateinit var before it is initialized."),
        CodeError(tag="miss-when-else", prompt="Omit the else branch in a non-exhaustive when expression."),
        CodeError(tag="string-ref-eq", prompt="Compare two Strings with === instead of ==."),
        CodeError(tag="float-prec", prompt="Compare two Double/Float values with ==."),
        CodeError(tag="miss-null-check", prompt="Replace safe ?. with a force unwrap or direct access."),
        CodeError(tag="err-ignore", prompt="Call a function returning Result or throwing errors and ignore failure."),
        CodeError(tag="miss-use", prompt="Open a resource without wrapping it in use { … }."),
        CodeError(tag="res-leak", prompt="Create a Cursor/FileInputStream and forget to close it."),
        CodeError(tag="coroutine-leak", prompt="Launch a coroutine in a loop capturing the loop variable and never cancel it."),
        CodeError(tag="race-access", prompt="Remove synchronization around shared mutable state."),
        CodeError(tag="deadlock-channel", prompt="Send and receive on the same unbuffered channel from the same coroutine."),
        CodeError(tag="magic-num", prompt="Inline the constant 0.75 multiple times instead of using a const val."),
        CodeError(tag="ill-comment", prompt="Insert an unterminated /* comment."),
        CodeError(tag="unclose-par", prompt="Remove a closing parenthesis from a function call."),
        CodeError(tag="miss-brace", prompt="Delete a closing } from a block."),
        CodeError(tag="inf-loop", prompt="Remove the increment or exit condition in a while/for loop."),
        CodeError(tag="unsafe-cast", prompt="Force-cast with as without checking the result."),
        CodeError(tag="assign-in-cond", prompt="Replace == with = inside an if condition (compiles to Unit, logic bug)."),
        CodeError(tag="global-leak", prompt="Declare a top-level mutable var that should be local."),
        CodeError(tag="path-traversal", prompt="Pass raw user input to File(userInputPath)."),
    ],
    "haskell": [
        CodeError(tag="missing-import", prompt="Delete a required import."),
        CodeError(tag="undefined-var", prompt="Replace an identifier with one that's not in scope."),
        CodeError(tag="ill-decl", prompt="Turn a function definition into a plain value binding."),
        CodeError(tag="miss-dollar", prompt="Remove a $ causing precedence/parentheses errors."),
        CodeError(tag="wrong-type-sig", prompt="Change a type signature so it no longer matches the definition."),
        CodeError(tag="missing-type-sig", prompt="Strip a top-level type signature."),
        CodeError(tag="type-mismatch", prompt="Replace an Int literal with a String literal (or similar) to break type inference."),
        CodeError(tag="nonexhaustive-case", prompt="Omit at least one pattern in a case expression."),
        CodeError(tag="partial-head", prompt="Call head on a potentially empty list."),
        CodeError(tag="partial-fromJust", prompt="Use fromJust on a Maybe Nothing."),
        CodeError(tag="idx-oob", prompt="Replace xs!!i with xs!!(i+1)."),
        CodeError(tag="pattern-shadow", prompt="Introduce a pattern variable that hides an outer one."),
        CodeError(tag="lazy-io-leak", prompt="Read a file with readFile and drop the handle before forcing the contents."),
        CodeError(tag="infinite-rec", prompt="Remove the base case of a recursive function."),
        CodeError(tag="bang-miss", prompt="Delete a BangPattern (!) on a strict field, re-introducing thunks."),
        CodeError(tag="monad-mismatch", prompt="Use an IO action inside pure code without lifting."),
        CodeError(tag="missing-return", prompt="Drop return in do-notation, yielding the raw value."),
        CodeError(tag="let-in-do", prompt="Add a let binding in do but forget the in keyword."),
        CodeError(tag="do-vs-pure", prompt="Write do notation for a pure (non-Monad) expression."),
        CodeError(tag="ambiguous-type", prompt="Use read \"123\" without an explicit type annotation."),
        CodeError(tag="err-ignored", prompt="Call error or undefined and leave it reachable."),
        CodeError(tag="unsafePerformIO", prompt="Wrap side-effecting code in unsafePerformIO."),
        CodeError(tag="nonstrict-eval", prompt="Build an infinite list and attempt to fully evaluate it."),
        CodeError(tag="wrong-eq", prompt="Replace == with = inside a guard or if."),
        CodeError(tag="ill-comment", prompt="Insert an unterminated {- comment."),
        CodeError(tag="magic-num", prompt="Inline the constant 0.75 multiple times instead of a named value."),
        CodeError(tag="module-cycle", prompt="Create two modules that import each other."),
        CodeError(tag="space-leak", prompt="Accumulate a large lazy list instead of streaming via foldl'."),
        CodeError(tag="deadlock-mvar", prompt="Take and never put an MVar, blocking indefinitely."),
    ],
    "lean": [
        CodeError(tag="missing-import", prompt="Delete a required import."),
        CodeError(tag="undefined-ident", prompt="Swap an identifier for one that's not in scope."),
        CodeError(tag="ill-decl", prompt="Turn a def into a let inside a tactic block."),
        CodeError(tag="wrong-type-sig", prompt="Change a declaration's type so it no longer matches the body."),
        CodeError(tag="missing-by", prompt="Remove the by keyword before a proof term."),
        CodeError(tag="sorry-proof", prompt="Replace a proof with by sorry."),
        CodeError(tag="partial-func", prompt="Define a recursive function without a termination proof."),
        CodeError(tag="nonexhaustive-match", prompt="Omit a pattern in a match expression."),
        CodeError(tag="idx-oob", prompt="Replace list.nth xs i with list.nth xs (i + 1)."),
        CodeError(tag="nat-int-confuse", prompt="Apply Nat.succ to an Int."),
        CodeError(tag="eq-asgn", prompt="Use := where = was meant in a proof step."),
        CodeError(tag="rewrite-dir", prompt="Use rw the wrong way around, reversing the equality."),
        CodeError(tag="simproc-loop", prompt="Add a simp lemma that rewrites a to a, causing a loop."),
        CodeError(tag="shadow-var", prompt="Introduce a local binding that hides an outer variable."),
        CodeError(tag="magic-num", prompt="Inline the constant 37 multiple times instead of naming it."),
        CodeError(tag="ill-comment", prompt="Insert an unterminated /- comment."),
        CodeError(tag="missing-open", prompt="Call simp without open Nat, leaving names unresolved."),
        CodeError(tag="term-hole", prompt="Leave a _ hole in a term."),
        CodeError(tag="unqualified-name", prompt="Use map instead of List.map without open List."),
        CodeError(tag="implicit-miss", prompt="Omit required implicit argument brackets {} in a call."),
        CodeError(tag="typeclass-miss", prompt="Delete an implicit instance argument [inst]."),
        CodeError(tag="inf-loop-tactic", prompt="Write repeat exact?, leading to non-termination."),
        CodeError(tag="coercion-confuse", prompt="Rely on a nonexistent coercion from Nat to Int."),
        CodeError(tag="rfl-abuse", prompt="Prove a non-trivial equality with plain rfl."),
        CodeError(tag="dec_trivial", prompt="Use decide on an undecidable proposition."),
        CodeError(tag="prop-vs-type", prompt="Supply a Type where Prop is expected."),
        CodeError(tag="unreachable", prompt="Add code after exact—it compiles but is dead."),
        CodeError(tag="unfold-loop", prompt="Mark a lemma @[simp, unfold], causing infinite unfolding."),
    ],
    "typescript": [
        CodeError(tag="missing-import", prompt="Remove a required import statement."),
        CodeError(tag="undefined-var", prompt="Replace a variable with an undeclared one."),
        CodeError(tag="ill-decl", prompt="Turn a function foo() into const foo = () => but leave downstream calls untouched."),
        CodeError(tag="miss-semi", prompt="Delete a semicolon."),
        CodeError(tag="bad-class", prompt="Replace a class Foo {} with const Foo = {};."),
        CodeError(tag="miss-ret", prompt="Remove the return in a non-void function."),
        CodeError(tag="ill-arr", prompt="Change [1, 2] to Array(1, 2)."),
        CodeError(tag="global-leak", prompt="Drop let/const, leaking to the global scope."),
        CodeError(tag="str-concat", prompt="Swap a template literal for + concatenation."),
        CodeError(tag="loose-eq", prompt="Replace === with ==."),
        CodeError(tag="assign-in-cond", prompt="Use = instead of === inside an if."),
        CodeError(tag="miss-await", prompt="Call an async fn without await."),
        CodeError(tag="unhandled-promise", prompt="Delete .catch(...) from a promise chain."),
        CodeError(tag="missing-break", prompt="Remove break in a switch case."),
        CodeError(tag="offby1", prompt="Change i < n to i <= n."),
        CodeError(tag="NaN-comp", prompt="Compare to NaN with ===."),
        CodeError(tag="parseInt-no-radix", prompt="Call parseInt(str) without the radix."),
        CodeError(tag="for-in-array", prompt="Iterate over an array with for…in."),
        CodeError(tag="hoist-use-before-def", prompt="Use a var before it's declared."),
        CodeError(tag="idx-oob", prompt="Replace arr[i] with arr[i + 1]."),
        CodeError(tag="json-circular", prompt="JSON.stringify a self-referential object."),
        CodeError(tag="insecure-eval", prompt="Pipe raw input to eval()."),
        CodeError(tag="typo-length", prompt="Misspell .length as .lenght."),
        CodeError(tag="shadow-var", prompt="Declare an inner variable hiding an outer one."),
        CodeError(tag="ill-comment", prompt="Insert an unterminated /* comment."),
        CodeError(tag="array-copy-ref", prompt="Do b = a instead of a.slice()."),
        CodeError(tag="float-prec", prompt="Compare floats with ==."),
        CodeError(tag="miss-brace", prompt="Delete a closing }."),
        CodeError(tag="inf-loop", prompt="Drop the increment in a while loop."),
        CodeError(tag="implicit-any", prompt="Omit a type so it defaults to any with noImplicitAny on."),
        CodeError(tag="any-abuse", prompt="Change a typed variable to any."),
        CodeError(tag="wrong-type-assign", prompt="Assign a string to a number-typed variable."),
        CodeError(tag="interface-mismatch", prompt="Return an object missing a required interface field."),
        CodeError(tag="excess-prop", prompt="Pass an object with an extra property to a strictly typed function."),
        CodeError(tag="wrong-generic-arg", prompt="Supply the wrong type parameter to a generic (e.g., Promise<string> → Promise<number>)."),
        CodeError(tag="enum-misuse", prompt="Assign 42 to a variable of enum type."),
        CodeError(tag="readonly-reassign", prompt="Write to a readonly property."),
        CodeError(tag="const-assertion-remove", prompt="Delete as const, widening the type."),
        CodeError(tag="non-null-misuse", prompt="Add ! to silence a nullable warning where value may be null."),
        CodeError(tag="type-cast-wrong", prompt="Force-cast (as) to an incompatible type."),
        CodeError(tag="decorator-misorder", prompt="Swap class and property decorator order."),
        CodeError(tag="ambient-missing", prompt="Use a package lacking @types and skip a manual declaration."),
        CodeError(tag="strict-null-off", prompt="Dereference undefined with strictNullChecks off."),
    ],
}


@dataclasses.dataclass
class PromptData:
    id: str
    prompt: str
    dataset_name: str
    original_row_id: str
    sampled_errors: List[str]


def get_dataset_columns(dataset_name: str) -> Dict[str, str]:
    """Get the column mappings for a specific dataset."""
    if dataset_name not in DATASET_COLUMN_MAPPINGS:
        raise ValueError(f"No column mapping found for dataset: {dataset_name}")
    return DATASET_COLUMN_MAPPINGS[dataset_name]


def sample_errors(language: str, num_errors: int) -> List[CodeError]:
    """Sample random errors for the given language."""
    if language not in LANGUAGE_ERRORS:
        raise ValueError(f"No errors defined for language: {language}")

    available_errors = LANGUAGE_ERRORS[language]
    if num_errors > len(available_errors):
        logger.warning(
            f"Requested {num_errors} errors but only {len(available_errors)} available for {language}"
        )
        num_errors = len(available_errors)

    return random.sample(available_errors, num_errors)


def create_prompt(code: str, errors: List[CodeError], language: str) -> str:
    """Create the prompt for code editing."""
    errors_str = "\n".join(
        [f"{i + 1}. {error.prompt}" for i, error in enumerate(errors)]
    )
    return (
        f"Please edit this code:\n\n{code}\nto insert the following "
        f"{len(errors)} errors:\n\n{errors_str}\n\nWhen you write the code "
        f"out at the end, make sure to surround it with ```{language} "
        f"and ```, like this: \n```{language}\n{{code}}\n```\n\n"
        "Do not change anything else about the code. Do not add any comments "
        " or mention the errors in the code. If some of the errors are not "
        "applicable to the code, ignore them. If you don't have any valid "
        "errors to apply, just print 'ERROR'."
    )


def create_batch_files(
    prompts: List[PromptData],
    base_batch_file_name: str,
    model: str,
    timestamp: int,
    max_completion_tokens: int,
) -> List[str]:
    """Create multiple batch files in the format required by Azure OpenAI Batch API.
    Returns a list of created batch file paths."""
    batch_file_paths = []

    # Split prompts into chunks of MAX_PROMPTS_PER_BATCH
    for i in range(0, len(prompts), MAX_PROMPTS_PER_BATCH):
        chunk = prompts[i : i + MAX_PROMPTS_PER_BATCH]
        batch_file_name = f"{base_batch_file_name}_{i // MAX_PROMPTS_PER_BATCH}.jsonl"

        with open(batch_file_name, "w") as f:
            for prompt in chunk:
                # Format each request according to batch API requirements
                batch_request = {
                    "custom_id": f"{timestamp}_{prompt.id}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt.prompt}],
                        "max_completion_tokens": max_completion_tokens,
                    },
                }
                f.write(json.dumps(batch_request) + "\n")

        batch_file_paths.append(batch_file_name)

    return batch_file_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Azure OpenAI batch API requests for code editing across multiple datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset configuration group
    dataset_group = parser.add_argument_group("Dataset Configuration")
    dataset_group.add_argument(
        "--datasets",
        type=str,
        required=True,
        help='Comma-separated list of dataset names (e.g., "user/dataset1,user/dataset2")',
    )
    dataset_group.add_argument(
        "--split",
        type=str,
        default="train",
        help='Dataset split to use (default: "train")',
    )
    dataset_group.add_argument(
        "--sample-limit",
        type=int,
        help="Limit the number of samples to process per dataset. If not specified, processes all samples.",
    )

    # Language and error configuration
    language_group = parser.add_argument_group("Language and Error Configuration")
    language_group.add_argument(
        "--num-errors",
        type=int,
        default=3,
        help="Number of errors to sample per code snippet (default: 3)",
    )

    # Model configuration group
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        type=str,
        default="o3-batch",
        choices=list(MODEL_COSTS_PER_1M_TOKENS.keys()),
        help=f"Model to use for completions. Available models: {', '.join(MODEL_COSTS_PER_1M_TOKENS.keys())}",
    )
    model_group.add_argument(
        "--max-completion-tokens",
        type=int,
        default=8192,
        help="Maximum number of completion tokens to use (default: 8192)",
    )

    # Runtime options group
    runtime_group = parser.add_argument_group("Runtime Options")
    runtime_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would happen without making any API calls",
    )
    runtime_group.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help='Directory to save batch files and metadata (default: "./output")',
    )

    args = parser.parse_args()

    # Validate arguments
    if args.sample_limit is not None and args.sample_limit <= 0:
        parser.error("--sample-limit must be a positive integer")

    if args.num_errors <= 0:
        parser.error("--num-errors must be a positive integer")

    return args


def get_language(dataset_name: str) -> str:
    """Get the language for a dataset."""
    if dataset_name.startswith("saurabh5/rlvr-code-data-"):
        language = dataset_name.split("-")[-1]
        if language in SUPPORTED_LANGUAGES:
            return language
        else:
            raise ValueError(f"Unknown language: {language}. Supported languages: {SUPPORTED_LANGUAGES}.")
    elif dataset_name in DATASET_COLUMN_MAPPINGS:
        return DATASET_COLUMN_MAPPINGS[dataset_name]["language"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def make_requests_for_dataset(
    dataset_name: str,
    num_errors: int,
    split: str,
    sample_limit: int | None,
    model: str,
    max_completion_tokens: int,
    dry_run: bool,
    tokenizer: tiktoken.Encoding,
    timestamp: int,
) -> tuple[List[PromptData], List[Dict[str, Any]], int, int]:
    """Make requests for a dataset."""
    logger.info(f"Making requests for dataset: {dataset_name}")

    # Load dataset
    dataset = datasets.load_dataset(dataset_name, split=split)
    logger.info(f"Loaded {len(dataset)} rows from {dataset_name}")

    # Get column mappings
    language = get_language(dataset_name)
    # Use a separate variable for column mapping lookup to avoid modifying the original dataset_name
    column_mapping_dataset_name = dataset_name
    if language in dataset_name:
        column_mapping_dataset_name = dataset_name.replace(f"-{language}", "")
        logging.info(f'Using column mapping for: {column_mapping_dataset_name}')
    column_mapping = get_dataset_columns(column_mapping_dataset_name)

    # Determine number of samples to process
    num_samples = (
        len(dataset) if sample_limit is None else min(sample_limit, len(dataset))
    )
    sampled_indices = random.sample(range(len(dataset)), num_samples)

    logger.info(f"Processing {num_samples} samples from {dataset_name}")

    prompts: List[PromptData] = []
    metadata: List[Dict[str, Any]] = []
    total_input_tokens, total_output_tokens = 0, 0
    rows_without_code = 0
    total_rows = 0
    for idx in sampled_indices:
        row = dataset[idx]
        total_rows += 1
        # Extract data using column mappings
        code = row[column_mapping["code_column"]]
        if code is None:
            logger.warning(f"Code is None for row {idx} in {dataset_name}")
            rows_without_code += 1
            continue
        description = row[column_mapping["description_column"]]
        if column_mapping["id_column"] == "python_index":
            row_id = idx
        else:
            row_id = row[column_mapping["id_column"]]

        # Sample errors
        sampled_errors = sample_errors(language, num_errors)

        # Create prompt
        prompt_text = create_prompt(code, sampled_errors, language)

        # Create PromptData object - use the original dataset_name
        prompt_data = PromptData(
            id=row_id,
            prompt=prompt_text,
            dataset_name=dataset_name,  # Use original dataset_name, not modified one
            original_row_id=row_id,
            sampled_errors=sampled_errors,
        )
        prompts.append(prompt_data)

        # Create metadata entry
        metadata_entry = {
            "prompt_id": row_id,
            "dataset_name": dataset_name,
            "original_row_id": row_id,
            "language": language,
            "sampled_errors": [error.tag for error in sampled_errors],
            "description": description,
            "timestamp": timestamp,
        }
        metadata.append(metadata_entry)

        # Count tokens
        prompt_tokens = len(tokenizer.encode(prompt_text))
        total_input_tokens += prompt_tokens

        # Estimate output tokens (rough estimate)
        estimated_output_tokens = (
            len(tokenizer.encode(code)) * 2
        )  # Assume output is roughly 2x input
        total_output_tokens += estimated_output_tokens

    logger.info(f"Total rows: {total_rows}, rows without code: {rows_without_code}")
    return prompts, metadata, total_input_tokens, total_output_tokens


def main(
    datasets: str,
    num_errors: int = 3,
    split: str = "train",
    sample_limit: int | None = None,
    model: str = "o3-batch",
    max_completion_tokens: int = 8192,
    dry_run: bool = False,
    output_dir: str = "./output",
) -> None:
    # Parse dataset list
    dataset_names = [name.strip().lower() for name in datasets.split(",")]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/batch_files", exist_ok=True)

    timestamp = int(time.time())
    base_batch_file_name = f"{output_dir}/batch_files/{timestamp}"

    # Set random seed for reproducibility
    random.seed(42)

    all_prompts: List[PromptData] = []
    all_metadata: List[Dict[str, Any]] = []
    tokenizer = tiktoken.encoding_for_model(model)
    total_input_tokens, total_output_tokens = 0, 0

    # Track batch IDs per dataset
    dataset_batch_ids: Dict[str, List[str]] = {}

    for dataset_name in dataset_names:
        prompts, metadata, input_tokens, output_tokens = make_requests_for_dataset(
            dataset_name,
            num_errors,
            split,
            sample_limit,
            model,
            max_completion_tokens,
            dry_run,
            tokenizer,
            timestamp,
        )
        logger.info(f'{len(prompts)} prompts for {dataset_name}.')
        prompt_names = set(p.dataset_name for p in prompts)
        assert len(prompt_names) == 1 and next(iter(prompt_names)) == dataset_name, f"Prompts for {dataset_name} have different dataset names: {prompt_names}"
        assert len(prompts) > 0, f"No prompts found for dataset: {dataset_name}"
        all_prompts.extend(prompts)
        all_metadata.extend(metadata)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

    # Calculate estimated cost
    estimated_cost = (
        total_input_tokens * MODEL_COSTS_PER_1M_TOKENS[model]["input"]
        + total_output_tokens * MODEL_COSTS_PER_1M_TOKENS[model]["output"]
    ) / 1_000_000

    logger.info("\nSummary:")
    logger.info(f"Total prompts: {len(all_prompts):,}")
    logger.info(
        f"Input tokens: {total_input_tokens:,} (estimated output tokens: {total_output_tokens:,})"
    )
    logger.info(f"Estimated cost: ${estimated_cost:,.2f}")

    metadata_path = f"{output_dir}/metadata.json"
    with open(metadata_path, "a") as f:
        json.dump(all_metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}.")

    if dry_run:
        logger.info("Dry run mode - exiting without making API calls. Prompts:")
        max_length = min(3, len(all_prompts))
        for prompt in all_prompts[:max_length]:
            logger.info(prompt.prompt)
        sys.exit(0)

    if estimated_cost > 100:
        logger.info(
            "\nWaiting 10 seconds to allow you to cancel the script if you don't want to proceed..."
        )
        time.sleep(10)

    # Initialize the client with your API key
    client = openai.AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2025-04-01-preview",
    )

    # Process each dataset separately to track batch IDs
    for dataset_name in dataset_names:
        logger.info(f"Creating batch files for dataset: {dataset_name}")
        
        # Filter prompts for this dataset
        dataset_prompts = [p for p in all_prompts if p.dataset_name == dataset_name]
        
        if not dataset_prompts:
            logger.warning(f"No prompts found for dataset: {dataset_name}")
            continue
            
        # Create batch files for this dataset
        dataset_base_name = f"{base_batch_file_name}_{dataset_name.replace('/', '_')}"
        batch_file_paths = create_batch_files(
            dataset_prompts, dataset_base_name, model, timestamp, max_completion_tokens
        )
        
        # Submit batch jobs for this dataset
        dataset_batch_ids[dataset_name] = []
        logger.info(f"Submitting {len(batch_file_paths)} batch jobs for {dataset_name}...")
        for batch_file_path in batch_file_paths:
            batch_file = client.files.create(
                file=open(batch_file_path, "rb"), purpose="batch"
            )

            batch_job = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            batch_id = batch_job.id
            dataset_batch_ids[dataset_name].append(batch_id)
            logger.info(f"Submitted batch job with ID: {batch_id} for {dataset_name}")

    # Print batch IDs in the requested format
    logger.info("\nBatch job IDs by dataset:")
    for dataset_name, batch_ids in dataset_batch_ids.items():
        logger.info(f"{dataset_name}: {', '.join(batch_ids)}")
    with open(f"{output_dir}/batch_ids.txt", "a") as f:
        for dataset_name, batch_ids in dataset_batch_ids.items():
            f.write(f"{dataset_name}: {', '.join(batch_ids)}\n")
    logger.info(f"Saved batch IDs to {output_dir}/batch_ids.txt")
    all_batch_ids = [batch_id for batch_ids in dataset_batch_ids.values() for batch_id in batch_ids]
    logger.info(f"\nAll batches: {','.join(all_batch_ids)}")
    logger.info("\nYou can check the status of your batch jobs using the IDs above.")


if __name__ == "__main__":
    args = parse_args()
    main(
        datasets=args.datasets,
        num_errors=args.num_errors,
        split=args.split,
        sample_limit=args.sample_limit,
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    )
