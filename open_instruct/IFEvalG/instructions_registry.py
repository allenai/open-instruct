# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry of all instructions."""

from open_instruct.IFEvalG import instructions


_KEYWORD = "keywords:"

_LANGUAGE = "language:"

_LENGTH = "length_constraints:"

_CONTENT = "detectable_content:"

_FORMAT = "detectable_format:"

_MULTITURN = "multi-turn:"

_COMBINATION = "combination:"

_STARTEND = "startend:"

_CHANGE_CASES = "change_case:"

_PUNCTUATION = "punctuation:"

INSTRUCTION_DICT = {
    _KEYWORD + "existence": instructions.KeywordChecker,
    _KEYWORD + "frequency": instructions.KeywordFrequencyChecker,
    # TODO(jeffreyzhou): make a proper set of sentences to choose from
    # _KEYWORD + "key_sentences": instructions.KeySentenceChecker,
    _KEYWORD + "forbidden_words": instructions.ForbiddenWords,
    _KEYWORD + "letter_frequency": instructions.LetterFrequencyChecker,
    _LANGUAGE + "response_language": instructions.ResponseLanguageChecker,
    _LENGTH + "number_sentences": instructions.NumberOfSentences,
    _LENGTH + "number_paragraphs": instructions.ParagraphChecker,
    _LENGTH + "number_words": instructions.NumberOfWords,
    _LENGTH + "nth_paragraph_first_word": instructions.ParagraphFirstWordCheck,
    _CONTENT + "number_placeholders": instructions.PlaceholderChecker,
    _CONTENT + "postscript": instructions.PostscriptChecker,
    _FORMAT + "number_bullet_lists": instructions.BulletListChecker,
    # TODO(jeffreyzhou): Pre-create paragraph or use prompt to replace
    # _CONTENT + "rephrase_paragraph": instructions.RephraseParagraph,
    _FORMAT + "constrained_response": instructions.ConstrainedResponseChecker,
    _FORMAT + "number_highlighted_sections": (instructions.HighlightSectionChecker),
    _FORMAT + "multiple_sections": instructions.SectionChecker,
    # TODO(tianjianlu): Re-enable rephrasing with preprocessing the message.
    # _FORMAT + "rephrase": instructions.RephraseChecker,
    _FORMAT + "json_format": instructions.JsonFormat,
    _FORMAT + "title": instructions.TitleChecker,
    # TODO(tianjianlu): Re-enable with specific prompts.
    # _MULTITURN + "constrained_start": instructions.ConstrainedStartChecker,
    _COMBINATION + "two_responses": instructions.TwoResponsesChecker,
    _COMBINATION + "repeat_prompt": instructions.RepeatPromptThenAnswer,
    _STARTEND + "end_checker": instructions.EndChecker,
    _CHANGE_CASES + "capital_word_frequency": instructions.CapitalWordFrequencyChecker,
    _CHANGE_CASES + "english_capital": instructions.CapitalLettersEnglishChecker,
    _CHANGE_CASES + "english_lowercase": instructions.LowercaseLettersEnglishChecker,
    _PUNCTUATION + "no_comma": instructions.CommaChecker,
    _STARTEND + "quotation": instructions.QuotationChecker,
    # the following are added for ifeval_mt variants
    # "type:translate": instructions.LanguageChecker, 
    # "type:repeat": instructions.RepeatChecker, 
    # "type:repeat_N": instructions.RewriteSentChecker,
    # "type:num_words": instructions.NumWordsChecker,
    # "type:num_words_N": instructions.NumWordsSentChecker, 
    # "constraint:increment": instructions.IncrementChecker, 
    # "constraint:digits": instructions.DigitsChecker, 
    # "constraint:nonalpha": instructions.NonAlphaNumChecker, 
    # "constraint:palindrome": instructions.PalindromeCountChecker,  
    # "constraint:chars": instructions.DiffNumCharsChecker, 
    # "constraint:punctuation": instructions.NumPunctuationChecker, 
    # "constraint:nochar": instructions.NoLetterChecker, 
    # "constraint:conjunctions": instructions.NoConjunctionsChecker, 
    # "constraint:vowels": instructions.VowelStartWordsChecker, 
    # "constraint:top_bottom": instructions.BeginEndLetterChecker, 
    # back to ood
    "count:word_count_range": instructions.WordCountRangeChecker,
    "count:unique_word_count" : instructions.UniqueWordCountChecker,
    "ratio:stop_words" : instructions.StopWordPercentageChecker,
    "ratio:sentence_type" : instructions.SentTypeRatioChecker,
    "ratio:sentence_balance" : instructions.SentBalanceChecker,
    "count:conjunctions" : instructions.ConjunctionCountChecker,
    "count:person_names" : instructions.PersonNameCountChecker,
    "ratio:overlap" : instructions.NGramOverlapChecker,
    "count:numbers" : instructions.NumbersCountChecker,
    "words:alphabet" : instructions.AlphabetLoopChecker,
    "words:vowel" : instructions.SingleVowelParagraphChecker,
    "words:consonants" : instructions.ConsonantClusterChecker,
    "sentence:alliteration_increment" : instructions.IncrementingAlliterationChecker,
    "words:palindrome" : instructions.PalindromeChecker,
    "count:punctuation" : instructions.PunctuationCoverChecker,
    "format:parentheses" : instructions.NestedParenthesesChecker,
    "format:quotes" : instructions.NestedQuotesChecker,
    "words:prime_lengths" : instructions.PrimeLengthsChecker,
    "format:options" : instructions.OptionsResponseChecker,
    "format:newline" : instructions.NewLineWordsChecker,
    "format:emoji" : instructions.EmojiSentenceChecker,
    "ratio:sentence_words" : instructions.CharacterCountUniqueWordsChecker,
    "count:words_japanese" : instructions.NthWordJapaneseChecker,
    "words:start_verb" : instructions.StartWithVerbChecker,
    "words:repeats" : instructions.LimitedWordRepeatChecker,
    "sentence:keyword" : instructions.IncludeKeywordChecker,
    "count:pronouns" : instructions.PronounCountChecker,
    "words:odd_even_syllables" : instructions.AlternateParitySyllablesChecker,
    "words:last_first" : instructions.LastWordFirstNextChecker,
    "words:paragraph_last_first" : instructions.ParagraphLastFirstWordMatchChecker,
    "sentence:increment" : instructions.IncrementingWordCountChecker,
    "words:no_consecutive" : instructions.NoConsecutiveFirstLetterChecker,
    "format:line_indent" : instructions.IndentStairsChecker,
    "format:quote_unquote" : instructions.QuoteExplanationChecker,
    "format:list" : instructions.SpecialBulletPointsChecker,
    "format:thesis" : instructions.ItalicsThesisChecker,
    "format:sub-bullets" : instructions.SubBulletPointsChecker,
    "format:no_bullets_bullets" : instructions.SomeBulletPointsChecker,
    "custom:multiples" : instructions.PrintMultiplesChecker,
    "custom:mcq_count_length": instructions.MultipleChoiceQuestionsChecker,
    "custom:reverse_newline": instructions.ReverseNewlineChecker,
    "custom:word_reverse": instructions.WordReverseOrderChecker,
    "custom:character_reverse": instructions.CharacterReverseOrderChecker,
    "custom:sentence_alphabet": instructions.SentenceAlphabetChecker,
    "custom:european_capitals_sort": instructions.EuropeanCapitalsSortChecker,
    "custom:csv_city": instructions.CityCSVChecker,
    "custom:csv_special_character": instructions.SpecialCharacterCSVChecker,
    "custom:csv_quotes": instructions.QuotesCSVChecker,
    "custom:date_format_list": instructions.DateFormatListChecker,
    "count:keywords_multiple" : instructions.KeywordsMultipleChecker,
    "words:keywords_specific_position" : instructions.KeywordSpecificPositionChecker,
    "words:words_position" : instructions.WordsPositionChecker,
    "repeat:repeat_change" : instructions.RepeatChangeChecker,
    "repeat:repeat_simple" : instructions.RepeatSimpleChecker,
    "repeat:repeat_span" : instructions.RepeatSpanChecker,
    "format:title_case" : instructions.TitleCaseChecker,
    "format:output_template" : instructions.OutputTemplateChecker,
    "format:no_whitespace" : instructions.NoWhitespaceChecker,
}

INSTRUCTION_CONFLICTS = {
    _KEYWORD + "existence": {_KEYWORD + "existence"},
    _KEYWORD + "frequency": {_KEYWORD + "frequency"},
    # TODO(jeffreyzhou): make a proper set of sentences to choose from
    # _KEYWORD + "key_sentences": instructions.KeySentenceChecker,
    _KEYWORD + "forbidden_words": {_KEYWORD + "forbidden_words"},
    _KEYWORD + "letter_frequency": {_KEYWORD + "letter_frequency"},
    _LANGUAGE + "response_language": {
        _LANGUAGE + "response_language",
        _FORMAT + "multiple_sections",
        _KEYWORD + "existence",
        _KEYWORD + "frequency",
        _KEYWORD + "forbidden_words",
        _STARTEND + "end_checker",
        _CHANGE_CASES + "english_capital",
        _CHANGE_CASES + "english_lowercase",
    },
    _LENGTH + "number_sentences": {_LENGTH + "number_sentences"},
    _LENGTH + "number_paragraphs": {
        _LENGTH + "number_paragraphs",
        _LENGTH + "nth_paragraph_first_word",
        _LENGTH + "number_sentences",
        _LENGTH + "nth_paragraph_first_word",
    },
    _LENGTH + "number_words": {_LENGTH + "number_words"},
    _LENGTH + "nth_paragraph_first_word": {
        _LENGTH + "nth_paragraph_first_word",
        _LENGTH + "number_paragraphs",
    },
    _CONTENT + "number_placeholders": {_CONTENT + "number_placeholders"},
    _CONTENT + "postscript": {_CONTENT + "postscript"},
    _FORMAT + "number_bullet_lists": {_FORMAT + "number_bullet_lists"},
    # TODO(jeffreyzhou): Pre-create paragraph or use prompt to replace
    # _CONTENT + "rephrase_paragraph": instructions.RephraseParagraph,
    _FORMAT + "constrained_response": set(INSTRUCTION_DICT.keys()),
    _FORMAT + "number_highlighted_sections": {_FORMAT + "number_highlighted_sections"},
    _FORMAT + "multiple_sections": {
        _FORMAT + "multiple_sections",
        _LANGUAGE + "response_language",
        _FORMAT + "number_highlighted_sections",
    },
    # TODO(tianjianlu): Re-enable rephrasing with preprocessing the message.
    # _FORMAT + "rephrase": instructions.RephraseChecker,
    _FORMAT + "json_format": set(INSTRUCTION_DICT.keys()).difference(
        {_KEYWORD + "forbidden_words", _KEYWORD + "existence"}
    ),
    _FORMAT + "title": {_FORMAT + "title"},
    # TODO(tianjianlu): Re-enable with specific prompts.
    # _MULTITURN + "constrained_start": instructions.ConstrainedStartChecker,
    _COMBINATION + "two_responses": set(INSTRUCTION_DICT.keys()).difference(
        {
            _KEYWORD + "forbidden_words",
            _KEYWORD + "existence",
            _LANGUAGE + "response_language",
            _FORMAT + "title",
            _PUNCTUATION + "no_comma",
        }
    ),
    _COMBINATION + "repeat_prompt": set(INSTRUCTION_DICT.keys()).difference(
        {_KEYWORD + "existence", _FORMAT + "title", _PUNCTUATION + "no_comma"}
    ),
    _STARTEND + "end_checker": {_STARTEND + "end_checker"},
    _CHANGE_CASES + "capital_word_frequency": {
        _CHANGE_CASES + "capital_word_frequency",
        _CHANGE_CASES + "english_lowercase",
        _CHANGE_CASES + "english_capital",
    },
    _CHANGE_CASES + "english_capital": {_CHANGE_CASES + "english_capital"},
    _CHANGE_CASES + "english_lowercase": {
        _CHANGE_CASES + "english_lowercase",
        _CHANGE_CASES + "english_capital",
    },
    _PUNCTUATION + "no_comma": {_PUNCTUATION + "no_comma"},
    _STARTEND + "quotation": {_STARTEND + "quotation", _FORMAT + "title"},
    "count:keywords_multiple" : {"count:keywords_multiple"},
    "words:words_position" : {"words:words_position"},
    "repeat:repeat_change" : {"repeat:repeat_change"},
    "repeat:repeat_simple" : {"repeat:repeat_simple"},
"repeat:repeat_span" : {"repeat:repeat_span"},
    "format:title_case" : {"format:title_case"},
    "format:output_template" : {"format:output_template"},
    "format:no_whitespace" : {'format:no_whitespace'},
    "words:keywords_specific_position": {"words:keywords_specific_position"},
}


def conflict_make(conflicts):
    """Makes sure if A conflicts with B, B will conflict with A.

    Args:
      conflicts: Dictionary of potential conflicts where key is instruction id
        and value is set of instruction ids that it conflicts with.

    Returns:
      Revised version of the dictionary. All instructions conflict with
      themselves. If A conflicts with B, B will conflict with A.
    """
    for key in conflicts:
        for k in conflicts[key]:
            conflicts[k].add(key)
        conflicts[key].add(key)
    return conflicts
