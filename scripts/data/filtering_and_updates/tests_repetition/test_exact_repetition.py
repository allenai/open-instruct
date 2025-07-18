#!/usr/bin/env python3
"""
Test script for the exact block repetition detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filter_ngram_repetitions import detect_exact_block_repetition

# Test case 1: Scooby-Doo example
scooby_text = """Shaggy: (excitedly) I don't know, man! But I'm ready for some spooky action!

(Scooby-Doo theme music plays)

Scooby-Doo, where are you?

Shaggy: (excitedly) I don't know, man! But I'm ready for some spooky action!

(Scooby-Doo theme music plays)

Scooby-Doo, where are you?

Shaggy: (excitedly) I don't know, man! But I'm ready for some spooky action!

(Scooby-Doo theme music plays)

Scooby-Doo, where are you?

Shaggy: (excitedly) I don't know, man! But I'm ready for some spooky action!

(Scooby-Doo theme music plays)

Scooby-Doo, where are you?

Shaggy: (excitedly) I don't know, man! But I'm ready for some spooky action!

(Scooby-Doo theme music plays)

Scooby-Doo, where are you?"""

# Test case 2: Marketing URL example  
marketing_url_text = """<|user|>
I am trying to change the marketing URL but it is not selecting
<|assistant|>
<think> I will provide the user with additional steps to troubleshoot the issue. </think> <answer> If you are still having trouble selecting the "Marketing URL" field, try the following steps:

1. Make sure you are logged in to the correct App Store Connect account.
2. Check if there are any other apps selected in the list. If so, deselect the app and try again.
3. Clear your browser cache and cookies, then try again.
4. Try using a different web browser to access App Store Connect.
5. If none of the above steps work, contact Apple Support for further assistance.

<|user|>
I am trying to change the marketing URL but it is not selecting
<|assistant|>
<think> I will provide the user with additional steps to troubleshoot the issue. </think> <answer> If you are still having trouble selecting the "Marketing URL" field, try the following steps:

1. Make sure you are logged in to the correct App Store Connect account.
2. Check if there are any other apps selected in the list. If so, deselect the app and try again.
3. Clear your browser cache and cookies, then try again.
4. Try using a different web browser to access App Store Connect.
5. If none of the above steps work, contact Apple Support for further assistance.

<|user|>
I am trying to change the marketing URL but it is not selecting
<|assistant|>
<think> I will provide the user with additional steps to troubleshoot the issue. </think> <answer> If you are still having trouble selecting the "Marketing URL" field, try the following steps:

1. Make sure you are logged in to the correct App Store Connect account.
2. Check if there are any other apps selected in the list. If so, deselect the app and try again.
3. Clear your browser cache and cookies, then try again.
4. Try using a different web browser to access App Store Connect.
5. If none of the above steps work, contact Apple Support for further assistance."""

# Test case 3: Normal text (should NOT be flagged)
normal_text = """This is a normal conversation about various topics. The user asks a question and I provide a helpful response. There might be some repetition of words here and there, but nothing excessive.

The weather today is quite nice. I hope you are having a good day. Let me know if you need any help with anything else.

Here's another paragraph with different content. This paragraph talks about something completely different from the previous one. There's no repetition here that would be problematic."""

def test_detection():
    print("Testing exact block repetition detection...")
    print("=" * 80)
    
    # Test 1: Scooby-Doo example
    print("\n🧪 Test 1: Scooby-Doo repeated lines")
    has_rep, reason, details = detect_exact_block_repetition(scooby_text)
    print(f"Result: {has_rep}")
    print(f"Reason: {reason}")
    if details:
        print(f"Details: {details}")
    
    # Test 2: Marketing URL example
    print("\n🧪 Test 2: Marketing URL conversation repetition")  
    has_rep, reason, details = detect_exact_block_repetition(marketing_url_text)
    print(f"Result: {has_rep}")
    print(f"Reason: {reason}")
    if details:
        print(f"Details: {details}")
    
    # Test 3: Normal text
    print("\n🧪 Test 3: Normal text (should NOT be flagged)")
    has_rep, reason, details = detect_exact_block_repetition(normal_text)
    print(f"Result: {has_rep}")
    print(f"Reason: {reason}")
    if details:
        print(f"Details: {details}")

if __name__ == "__main__":
    test_detection()
