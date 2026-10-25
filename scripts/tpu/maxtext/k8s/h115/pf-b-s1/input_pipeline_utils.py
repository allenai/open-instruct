# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Operations used by Grain"""

import dataclasses
import os
import re
import warnings
from threading import current_thread
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
  import datasets
  import tensorflow as tf

import grain.python as grain
import numpy as np
from grain._src.python.dataset.sources.tfrecord_dataset import _TFRecordReader, _TFRecordDatasetIterator  # pylint: disable=protected-access
from grain.experimental import TFRecordIterDataset
from maxtext.diffusion.block_diffusion import corruption as block_diffusion_corruption
from maxtext.input_pipeline.protos import example_pb2
from maxtext.input_pipeline import tokenizer
from maxtext.multimodal import processor as mm_processor
from maxtext.multimodal import utils as mm_utils
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
from transformers.utils import chat_template_utils

Features = dict[str, Any]
INPUT_TOKENS_KEY = "input_ids"

########## Functions used by TFDS pipeline


def normalize_features(x, column_name):
  return {"inputs": x[column_name], "targets": x[column_name]}


def get_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token=None):
  # Load tokenizer
  tokenizer_model = tokenizer.build_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token)
  return tokenizer_model


def truncate_to_max_allowable_length(x, max_length):
  return {k: v[:max_length] for k, v in x.items()}


def shift_data_by_truncation(x):
  x["inputs"] = x["inputs"][:-1]
  x["targets"] = x["targets"][1:]
  return x


def add_segmentation_and_position(x, data_columns, padding_token=0):
  import tensorflow as tf  # pylint: disable=import-outside-toplevel

  for data_column in data_columns:
    x[f"{data_column}_segmentation"] = tf.cast(x[data_column] != padding_token, tf.int32)
    x[f"{data_column}_position"] = tf.broadcast_to(
        tf.range(x[data_column].shape[-1], dtype=np.int32)[None, :],
        x[data_column].shape,
    )
  return x


def TokenizeOp(
    tokenizer_model,
    features: Features,
    data_keys: Iterable[str] = ("inputs", "targets"),
) -> Features:
  """Op for tokenization"""
  import tensorflow as tf  # pylint: disable=import-outside-toplevel

  def _process_string(string_tensor):
    # Extract string value and decode it if necessary
    string_value = string_tensor.numpy().decode("utf-8")
    # encode and extract the tokenized integers
    modified_string = tokenizer_model.encode(string_value)
    return [modified_string]

  for k in data_keys:
    features[k] = tf.py_function(_process_string, [features[k]], Tout=[tf.int32])[0]
  return features


########## Functions used by HF pipeline


def is_video_file(path):
  if not isinstance(path, str):
    return False
  video_extensions = (".mp4", ".avi", ".mkv", ".webm", ".mov")
  return path.lower().endswith(video_extensions)


def reformat_prompt(
    example,
    column,
    image_placeholder,
    model_name,
    video_placeholder="<|video|>",
):
  """reformat prompt for multimodal SFT"""
  media = example.get("images")
  if media is None:
    media = []
  elif not isinstance(media, list):
    media = [media]
  contains_video = bool(media) and is_video_file(media[0])

  example[column] = mm_processor.reformat_prompt(
      prompt=example[column],
      image_placeholder=image_placeholder,
      model_name=model_name,
      num_images=0 if contains_video else len(media),
      video_placeholder=video_placeholder,
      num_videos=len(media) if contains_video else 0,
  )
  return example


def reformat_response(example, column, model_name):
  """reformat response for multimodal SFT"""
  val = example[column]
  if not val:
    raise ValueError(f"Response column '{column}' cannot be empty or None: {val}")
  response = val[0] if isinstance(val, (list, tuple)) else val

  example[column] = mm_processor.reformat_response(response, model_name)
  return example


def merge_image_columns(example, image_columns, max_num_images_per_example):
  """Merge multiple image columns into a single list of images."""
  images = []
  for col in image_columns:
    if isinstance(example[col], list):
      images.extend(example[col])
    else:
      images.append(example[col])

  example["images"] = images[:max_num_images_per_example] if max_num_images_per_example > 0 else images
  return example


def pre_process_image_sft(example, image_column, config):
  """pre-process image for multimodal SFT"""

  def _process_image_fn(image):
    is_video = is_video_file(image) or (isinstance(image, list) and image and is_video_file(image[0]))
    if is_video:
      return mm_processor.preprocess_image_for_training(image, config)

    if isinstance(image, list):
      image = [np.array(mm_utils.convert_to_RGB(img)) for img in image]
    else:
      image = np.array(mm_utils.convert_to_RGB(image))

    image = mm_processor.preprocess_image_for_training(image, config)
    return image

  example[image_column] = _process_image_fn(example[image_column])
  return example


def prepare_text_for_image_fusion(example, column_name, config):
  """prepare text for image fusion for multimodal SFT"""
  example[column_name] = mm_processor.prepare_text_for_image_fusion(
      tokens=example[column_name], config=config, processor_output=example["images"]
  )
  return example


def combine_columns(example, columns, data_column):
  """Combine columns such as 'prompt' and 'completion' for sft training"""
  assert len(columns) > 1
  combined = []
  for i in range(len(example[columns[0]])):
    for c in columns:
      combined.append(example[c][i])
  example[data_column] = combined
  return example


def is_conversational(features, data_columns):
  """Check if data is in a conversational format.
  Examples:

  features = {'prompt': [{'content': Value(dtype='string', id=None), 'role': Value(dtype='string', id=None)}],
              'completion': [{'content': Value(dtype='string', id=None), 'role': Value(dtype='string', id=None)}]}
  data_columns = ["prompt", "completion"]
  is_conversational(features, data_columns) return True.

  features = {'prompt': [Value(dtype='string', id=None)], 'completion': [Value(dtype='string', id=None)]}
  data_columns = ["prompt", "completion"]
  is_conversational(features, data_columns) returns False.
  """
  import datasets  # pylint: disable=import-outside-toplevel

  for column in data_columns:
    messages = features[column]
    if isinstance(messages, datasets.Sequence):
      if (
          # pyrefly: ignore[missing-attribute]
          isinstance(messages.feature, dict)
          and "role" in messages.feature
          and "content" in messages.feature  # pyrefly: ignore[missing-attribute]
      ):
        return True

  return False


def extract_token_ids(tokens):
  """Extracts token IDs from various tokenizer output formats.

  This helper function standardizes the extraction of tokenized integer IDs
  from common return types of Hugging Face tokenizers, including
  `BatchEncoding` objects, dictionaries, or simple lists.

  Args:
    tokens: The object containing token IDs. Supported types include:
      - A list of integers.
      - A dictionary containing the `INPUT_TOKENS_KEY`.
      - An object (e.g., `BatchEncoding`) with an attribute named `INPUT_TOKENS_KEY`.

  Returns:
    A list of integer token IDs.

  Raises:
    ValueError: If the input type is not supported or does not contain the expected key.
  """
  # attention masks in BatchEncoding are effectively ignored
  if hasattr(tokens, INPUT_TOKENS_KEY):
    return getattr(tokens, INPUT_TOKENS_KEY)
  elif isinstance(tokens, dict) and INPUT_TOKENS_KEY in tokens:
    return tokens[INPUT_TOKENS_KEY]
  elif isinstance(tokens, list):
    return tokens
  else:
    raise ValueError(f"Can't extract token_ids from type {type(tokens)}")


def _assistant_span_is_clean(messages, message_idx, rendered, span_text):
  """Whether a derived assistant span lines up with its message: it covers the content, starts at it
  (not inside the assistant header) and does not run into later messages. Turns whose content the
  template rewrites are not checked. Mirrors open-instruct's _verify_assistant_spans_cover_content."""
  content = messages[message_idx].get("content")
  if not content or content not in rendered:
    return True
  if content not in span_text or not span_text.lstrip().startswith(content):
    return False
  tail = span_text[span_text.index(content) + len(content) :]
  return not any(m.get("content") and m["content"] in tail for m in messages[message_idx + 1 :])


def _segments_from_spans(full_tokens, spans):
  """(token ids, is_prompt) runs covering `full_tokens`, with the token ranges in `spans` trained."""
  segments = []
  pos = 0
  for start, end in spans:
    start = max(start, pos)
    if start > pos:
      segments.append((list(full_tokens[pos:start]), True))
    if end > start:
      segments.append((list(full_tokens[start:end]), False))
    pos = max(pos, end)
  if pos < len(full_tokens):
    segments.append((list(full_tokens[pos:]), True))
  return segments


_GENERATION_BLOCK = re.compile(r"\{%[-+]?\s*generation\s*[-+]?%\}")


def _generation_block_segments(tokenizer_model, conversation, template):
  """Segments for templates that mark assistant output with `{% generation %}` blocks, as open-instruct labels
  them: render once with the generation character ranges, tokenize the rendering with offsets, and train every
  token that overlaps a range (boundary merges included, zero-width offsets never). A conversation whose
  end-of-turn or end-of-sequence token sits just outside its block is not trained (open-instruct drops it)."""
  rendered_chats, generation_indices = chat_template_utils.render_jinja_template(
      conversations=[conversation],
      chat_template=template,
      return_assistant_tokens_mask=True,
      add_generation_prompt=False,
      **tokenizer_model.special_tokens_map,
  )
  rendered, ranges = rendered_chats[0], [tuple(r) for r in generation_indices[0]]
  enc = tokenizer_model(rendered, add_special_tokens=False, return_offsets_mapping=True)
  full_tokens, offsets = enc["input_ids"], np.asarray(enc["offset_mapping"]).reshape(-1, 2)
  if not any(start < end for start, end in ranges):
    return [(list(full_tokens), True)] if any(m["role"] == "assistant" for m in conversation) else []
  terminators = {t for t in (tokenizer_model.eos_token, "<|im_end|>") if t}
  for start, end in ranges:
    if start == end:
      continue
    t0 = end
    while t0 < len(rendered) and rendered[t0].isspace():
      t0 += 1
    for terminator in terminators:
      if rendered.startswith(terminator, t0):
        covered = t0
        for b0, b1 in sorted(ranges):
          if b0 <= covered < b1:
            covered = b1
        if covered < t0 + len(terminator):
          max_logging.log(f"Not training on a conversation whose {terminator!r} sits outside its generation block.")
          return [(list(full_tokens), True)]
  trained = np.zeros(len(full_tokens), dtype=bool)
  for start, end in ranges:
    if start < end:
      trained |= (offsets[:, 1] > start) & (offsets[:, 0] < end)
  trained &= offsets[:, 0] < offsets[:, 1]
  spans, i = [], 0
  while i < len(trained):
    if trained[i]:
      j = i
      while j < len(trained) and trained[j]:
        j += 1
      spans.append((i, j))
      i = j
    else:
      i += 1
  return _segments_from_spans(full_tokens, spans)


def _split_conversation_into_segments(tokenizer_model, conversation):
  """Splits a conversation into formatted (text, is_prompt) segments covering its whole rendering.

  The conversation is templated once, as a whole, and every segment is cut from that rendering, so
  turns appear exactly as the chat template renders the full conversation (one system prompt, no
  extra end-of-sequence tokens between turns). Assistant turn i is trained from the token count of
  the conversation before it rendered with the generation prompt, to the token count of the
  conversation through it. When every such prefix renders as a literal prefix of the whole, a token
  straddling the end of the generation prompt is trained too (open-instruct's char-offset spans);
  otherwise, e.g. a template closing the last message with end-of-sequence instead of end-of-turn,
  spans come from token counts alone and each is checked against its message. Messages after the
  last assistant turn stay as untrained context.

  If a checked span does not line up with its message's content (it would put header or later-message
  tokens in the loss), no token of the conversation is trained. open-instruct drops such conversations.

  Args:
    tokenizer_model: The tokenizer instance.
    conversation: The messages of one conversation.

  Returns:
    A list of (token ids, is_prompt) pairs. Token ids, not text: re-tokenizing a decoded slice need not give the
    same tokens back.
  """

  # Which labelling rule to follow is the recipe's, not the template's: open-instruct main labels
  # `{% generation %}` blocks (allenai/open-instruct#1879), while older branches (e.g. the Olmo 3.5 hero SFT
  # runs) derive spans from rendered prefixes even when the template has blocks. They differ on templates whose
  # generation prompt ends inside the block, e.g. Olmo 3.5's trailing <think>.
  template = tokenizer_model.get_chat_template()
  use_blocks = os.environ.get("MAXTEXT_SFT_LABELS", "generation_blocks") == "generation_blocks"
  if use_blocks and isinstance(template, str) and _GENERATION_BLOCK.search(template):
    return _generation_block_segments(tokenizer_model, conversation, template)

  def n_tokens(messages, add_generation_prompt):
    return len(
        extract_token_ids(
            tokenizer_model.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=True)
        )
    )

  full_tokens = extract_token_ids(
      tokenizer_model.apply_chat_template(conversation, add_generation_prompt=False, tokenize=True)
  )
  rendered = tokenizer_model.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
  decode = lambda tokens: tokenizer_model.decode(tokens, skip_special_tokens=False)
  assistant_indices = [idx for idx, message in enumerate(conversation) if message["role"] == "assistant"]
  # Conversations whose prefixes render as literal prefixes of the whole need no check; the rest (e.g. a
  # template closing the last message with end-of-sequence) are checked turn by turn, as in open-instruct.
  prefix_stable = all(
      rendered.startswith(
          tokenizer_model.apply_chat_template(conversation[: idx + 1], add_generation_prompt=False, tokenize=False)
      )
      and (
          idx == 0
          or rendered.startswith(
              tokenizer_model.apply_chat_template(conversation[:idx], add_generation_prompt=True, tokenize=False)
          )
      )
      for idx in assistant_indices
  )
  spans = []
  for idx in assistant_indices:
    if idx == 0:
      start = 0
    elif prefix_stable:
      # A token straddling the end of the generation prompt is trained, as open-instruct's char-offset
      # spans do: the span starts where the tokens of the full rendering first differ from the prompt's.
      prompt_tokens = extract_token_ids(
          tokenizer_model.apply_chat_template(conversation[:idx], add_generation_prompt=True, tokenize=True)
      )
      start = next((i for i, (x, y) in enumerate(zip(full_tokens, prompt_tokens)) if x != y), len(prompt_tokens))
    else:
      start = n_tokens(conversation[:idx], True)
    end = min(n_tokens(conversation[: idx + 1], False), len(full_tokens))
    if (
        not prefix_stable
        and start < end
        and not _assistant_span_is_clean(conversation, idx, rendered, decode(full_tokens[start:end]))
    ):
      max_logging.log(f"Not training on a conversation whose assistant span {idx} does not match its content.")
      return [(list(full_tokens), True)]
    spans.append((start, end))
  if not spans:
    return []
  return _segments_from_spans(full_tokens, spans)


def _lm_row_ids(tokenizer_model, text):
  """Token ids of a plain LM row: the text without the chat template, then one EOS (H099's rule).

  Like H099's builder, the text may not tokenize to EOS or any other added token: the data step drops such rows.
  """
  ids = list(extract_token_ids(tokenizer_model(text, add_special_tokens=False)))
  eos = tokenizer_model.eos_token_id
  added = set(tokenizer_model.added_tokens_decoder)
  if eos is None or added.intersection(ids):
    raise ValueError("An LM row needs a tokenizer EOS id, and its text must not tokenize to an added token.")
  return ids + [eos]


def apply_chat_template(example, tokenizer_model, data_column_name):
  """Formats conversational data by applying the tokenizer's chat template
  and identifying prompt/completion segments for SFT masking.

  Args:
    example: A dictionary containing conversational data. It is expected to have a key
      specified by `data_column_name` that holds a list of messages.
    tokenizer_model: The tokenizer instance associated with the language model,
      which contains the specific chat template.
    data_column_name: The name of the column in the `example` dictionary
      that contains the list of messages.

  Returns:
    The modified `example` dictionary.
      - The `data_column_name` column will be updated to a list of
        messages, each formatted according to the tokenizer's chat template.
      - A new column "is_prompt" is added, where `True` indicates the
        tokens contain the system message, user message, and generation
        prompt (if applicable). `False` indicates the expected LLM
        completion, excluding the assistant's start tokens.
  """
  conversation = list(example[data_column_name])
  # LM rows (H115's replay arm, 2026-10-23): with MAXTEXT_SFT_LM_ROLE=<role>, a conversation that is one message of
  # that role is trained on every token of its content plus one EOS. Every other conversation takes the path below.
  lm_role = os.environ.get("MAXTEXT_SFT_LM_ROLE", "")
  if lm_role and len(conversation) == 1 and conversation[0]["role"] == lm_role:
    example["is_prompt"] = [False]
    example[data_column_name] = [_lm_row_ids(tokenizer_model, conversation[0]["content"])]
    return example
  try:
    for idx, message in enumerate(conversation):
      if message["role"] == "system" and idx != 0:
        raise ValueError(f"System message found at index {idx}. System messages must be at index 0.")
    segments = _split_conversation_into_segments(tokenizer_model, conversation)
  except ValueError as e:
    max_logging.log(f"Unable to apply chat template: {e}")
    raise e
  example["is_prompt"] = [is_prompt for _, is_prompt in segments]
  example[data_column_name] = [text for text, _ in segments]
  return example


def tokenization(example, hf_tokenizer, truncation, max_length, column_names):
  """Tokenize a HuggingFace dataset"""
  for column_name in column_names:
    if isinstance(example[column_name], list):
      example[column_name] = [
          x if not isinstance(x, str) else hf_tokenizer(x, truncation=truncation, max_length=max_length)["input_ids"]
          for x in example[column_name]
      ]
    elif isinstance(example[column_name], str):
      example[column_name] = hf_tokenizer(example[column_name], truncation=truncation, max_length=max_length)["input_ids"]
  return example


@dataclasses.dataclass
class SFTPromptMasking(grain.MapTransform):
  """Construct inputs and targets for SFT training. Concat prompt and completion to generate inputs.
  For targets, if train on completion only, the prompt will be masked by unk_id. Otherwise the same as inputs.
  """

  def __init__(self, text_column_name, completion_only, max_target_length, unk_id=0):
    self.text_column_name = text_column_name
    self.completion_only = completion_only
    self.max_target_length = max_target_length
    self.unk_id = unk_id

  def map(self, element):
    """
    Maps a single dataset element to an SFT training instance.
    It concatenates the prompt and completion to form the `inputs` sequence.
    For the `targets` sequence:
    - If `self.completion_only` is `True`, the prompt portion of the
      concatenated sequence is masked using `self.unk_id`.
    - If `self.completion_only` is `False`, the target sequence is
      identical to the input sequence.
    """
    inputs, targets = [], []
    for i, text in enumerate(element[self.text_column_name]):
      inputs += text
      targets += [self.unk_id] * len(text) if self.completion_only and element["is_prompt"][i] else text
    return {
        "inputs": np.asarray(inputs[: self.max_target_length], dtype=np.int32),
        "targets": np.asarray(targets[: self.max_target_length], dtype=np.int32),
    }


@dataclasses.dataclass
class SFTPromptMaskingVision(grain.MapTransform):
  """SFT prompt masking for multimodal"""

  def __init__(self, query_column, response_column, max_target_length, pad_id):
    self.query_column = query_column
    self.response_column = response_column
    self.max_target_length = max_target_length
    self.pad_id = pad_id

  def map(self, element):
    inputs = np.concatenate((element[self.query_column], element[self.response_column]))
    targets = np.concatenate(
        (
            np.asarray([self.pad_id] * len(element[self.query_column])),
            element[self.response_column],
        )
    )
    return {
        "inputs": np.asarray(inputs[: self.max_target_length], dtype=np.int32),
        "targets": np.asarray(targets[: self.max_target_length], dtype=np.int32),
        "images": element["images"],
    }


@dataclasses.dataclass
class HFNormalizeFeatures(grain.MapTransform):
  """Normalize feature keys for HuggingFace input"""

  def __init__(self, column_name):
    self.column_name = column_name

  def map(self, element):
    return {
        "inputs": np.asarray(element[self.column_name], dtype=np.int32),
        "targets": np.asarray(element[self.column_name], dtype=np.int32),
    }


class HFDataSource(grain.RandomAccessDataSource):
  """A class that makes HuggingFace IterableDataset a grain datasource without random access support"""

  def __init__(
      self,
      dataset: "datasets.IterableDataset",
      dataloading_host_index: int,
      dataloading_host_count: int,
      num_threads: int,
      max_target_length: int,
      data_column_names: list[str],
  ):
    from datasets.distributed import split_dataset_by_node  # pylint: disable=import-outside-toplevel

    self._split_dataset_by_node = split_dataset_by_node
    self.dataset = dataset
    self.num_threads = num_threads
    self.dataloading_host_count = dataloading_host_count
    self.dataloading_host_index = dataloading_host_index
    self.max_target_lenth = max_target_length
    self.data_column_names = data_column_names
    if hasattr(dataset, "n_shards"):
      self.n_shards = dataset.n_shards
    else:
      self.n_shards = 1
    self._check_shard_count()
    self.dataset_shards = [dataloading_host_index * self.num_threads + i for i in range(self.num_threads)]
    self.datasets = [self._split_dataset_by_node(dataset, world_size=self.n_shards, rank=x) for x in self.dataset_shards]
    self.data_iters = []

  def _check_shard_count(self):
    if self.n_shards < (self.dataloading_host_count * self.num_threads):
      warnings.warn(
          f"WARNING: Inefficient dataloading. Your train or eval dataset contains {self.n_shards} shards, "
          "smaller than number of host loading data. This is known to lead to inefficient dataloading. See"
          "github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md#multihost-dataloading-best-practice"
      )
      self.n_shards = self.dataloading_host_count * self.num_threads

  def _update_shard(self, idx):
    """update shard"""
    new_shard = self.dataset_shards[idx] + self.dataloading_host_count * self.num_threads
    if new_shard < self.n_shards:
      max_logging.log(
          f"Updating host {self.dataloading_host_index} dataset {idx}, was on shard {self.dataset_shards[idx]}"
      )
      max_logging.log(f"New shard is {new_shard}")
      self.dataset_shards[idx] = new_shard
      self.datasets[idx] = self._split_dataset_by_node(
          self.dataset, world_size=self.n_shards, rank=self.dataset_shards[idx]
      )
      self.data_iters[idx] = iter(self.datasets[idx])
    else:
      raise StopIteration(f"Run out of shards on host {self.dataloading_host_index}, shard {new_shard} is not available")

  def __len__(self):
    """Return length of the HF dataset. Since HuggingFace IterableDataset does not have length,
    a fake length bigger than the dataset is returned"""
    return 10_000_000_000

  def __getitem__(self, index):
    """Since HuggingFace IterableDataset does not support random access by index.
    The next item in the iterator is returned."""
    if not self.data_iters:
      self.data_iters = [iter(x) for x in self.datasets]
    idx = int(current_thread().name.split("_")[1])

    while True:
      try:
        data = next(self.data_iters[idx])
        return data
      except StopIteration:
        self._update_shard(idx)


########## Functions used by Grain pipeline


class _GCSTFRecordReader(_TFRecordReader):
  """Extends Grain's _TFRecordReader to open TFRecord files from GCS via streaming BlobReader."""

  def __init__(self, path: str):
    # Skip parent __init__ (which calls open(path, "rb")) and open via GCS BlobReader instead.
    bucket_name, blob_name = gcs_utils.parse_gcs_bucket_and_prefix(path)
    self._reader = gcs_utils.storage.Client().bucket(bucket_name).blob(blob_name).open("rb")


class _GCSTFRecordDatasetIterator(_TFRecordDatasetIterator):
  """Extends Grain's _TFRecordDatasetIterator to use _GCSTFRecordReader for GCS paths."""

  def __init__(self, path: str):
    # Skip parent __init__ (which creates _TFRecordReader); use GCS-aware reader instead.
    grain.DatasetIterator.__init__(self)
    self._reader = _GCSTFRecordReader(path)


class GCSTFRecordIterDataset(TFRecordIterDataset):
  """Extends Grain's TFRecordIterDataset to support GCS paths."""

  def __iter__(self) -> grain.DatasetIterator:  # pylint: disable=non-iterator-returned
    return _GCSTFRecordDatasetIterator(self._path)


def make_tfrecord_iter_dataset(path: str):
  """Returns the appropriate TFRecordIterDataset for local or GCS paths."""
  if path.startswith("gs://"):
    return GCSTFRecordIterDataset(path)
  return TFRecordIterDataset(path)


def make_parquet_iter_dataset(path: str, hf_access_token: str | None = None):
  """Returns the appropriate ParquetIterDataset for local or HF paths."""
  if path.startswith("hf://"):
    from huggingface_hub import HfFileSystem  # pylint: disable=import-outside-toplevel

    return grain.experimental.ParquetIterDataset(path, filesystem=HfFileSystem(token=hf_access_token))
  return grain.experimental.ParquetIterDataset(path)


def compute_file_sharding(file_count, host_index, host_count):
  """Compute per-host file slicing and optional row-shard parameters.

  When file_count >= host_count, each host reads a disjoint subset of files via the
  standard `[host_index::host_count]` slice. When file_count < host_count, every file
  is replicated across `ceil(host_count/file_count)` hosts (or `floor` for files past
  the remainder), and those hosts shard records by index within the file. This bounds
  concurrent readers per file at `ceil(host_count/file_count)`

  Returns:
    file_slice (slice): file slice this host should keep — also the (start, step)
      pair `(host_index, host_count)` to feed `tf.distribute.InputContext` when applicable.
    files_per_host (int): files this host's slice contains per epoch.
    row_shard (tuple|None): `(row_shard_index, row_shard_count)` when host_count > file_count
      and the file's group has >1 reader; otherwise None.
  """
  if file_count >= host_count:
    return (
        slice(host_index, None, host_count),
        max(file_count // host_count, 1),
        None,
    )
  file_idx = host_index % file_count
  row_shard_idx = host_index // file_count
  row_shard_count = (host_count // file_count) + (1 if file_idx < (host_count % file_count) else 0)
  row_shard = (row_shard_idx, row_shard_count) if row_shard_count > 1 else None
  return slice(file_idx, None, file_count), 1, row_shard


class _IndexShardDatasetIterator(grain.DatasetIterator):
  """Iterator that yields every nth element of its parent (round-robin by index)."""

  def __init__(self, parent: grain.DatasetIterator, host_index: int, host_count: int):
    super().__init__(parent)
    self._host_index = host_index
    self._host_count = host_count
    self._next_index = 0

  def __next__(self):
    while True:
      value = next(self._parent)
      current = self._next_index
      self._next_index += 1
      if current % self._host_count == self._host_index:
        return value

  def get_state(self):
    return {
        "next_index": self._next_index,
        "parent": self._parent.get_state(),
    }

  def set_state(self, state):
    self._next_index = state["next_index"]
    self._parent.set_state(state["parent"])


class IndexShardIterDataset(grain.IterDataset):
  """Shards an IterDataset across hosts by element index (host i keeps records where idx % N == i).

  Use when the upstream `IterDataset` order is deterministic and identical on every host;
  this guarantees disjoint, balanced slices without per-file sharding.
  """

  def __init__(self, parent: grain.IterDataset, host_index: int, host_count: int):
    super().__init__(parent)
    self._host_index = host_index
    self._host_count = host_count

  def __iter__(self) -> _IndexShardDatasetIterator:
    return _IndexShardDatasetIterator(
        self._parent.__iter__(),
        host_index=self._host_index,
        host_count=self._host_count,
    )


@dataclasses.dataclass
class ParseFeatures(grain.MapTransform):
  """Parse serialized tf.train.Example protos for arrayrecord/tfrecord datasets.

  Also validates that the stored field type matches `tokenize`: raises
  ValueError if `tokenize=True` but the column contains integers (pre-tokenized)
  or if `tokenize=False` but the column contains bytes (raw text).
  """

  def __init__(self, data_columns, tokenize):
    self.data_columns = list(data_columns)
    self.tokenize = tokenize

  def map(self, element):
    """Parse a serialized tf.train.Example proto and extract features."""
    example = example_pb2.Example()
    example.ParseFromString(element)
    features = example.features.feature

    parsed = {}
    for col in self.data_columns:
      target_col = col
      if col not in features:
        # Fallback alias: support bidirectional mapping between 'text' and 'messages'
        if col == "text" and "messages" in features:
          target_col = "messages"
        elif col == "messages" and "text" in features:
          target_col = "text"
        else:
          raise ValueError(
              f"Column '{col}' not found in dataset. Available columns: {sorted(features.keys())}. "
              "Please set train_data_columns or eval_data_columns accordingly."
          )

      f = features[target_col]

      # Dynamically check proto field type instead of relying on the tokenize flag
      if len(f.float_list.value) > 0:
        parsed[col] = np.array(f.float_list.value, dtype=np.float32)
      elif len(f.int64_list.value) > 0:
        parsed[col] = np.array(f.int64_list.value, dtype=np.int32)
      elif len(f.bytes_list.value) > 0:
        parsed[col] = np.array(f.bytes_list.value, dtype=object)
      else:
        parsed[col] = np.array([])

    # Reshape the flattened arrays back to 2D [seq_len, top_k]
    seq_len = len(parsed.get("inputs", []))
    if seq_len > 0:
      if "top_k_logits" in parsed and len(parsed["top_k_logits"]) > 0:
        parsed["top_k_logits"] = parsed["top_k_logits"].reshape(seq_len, -1)
      if "top_k_indices" in parsed and len(parsed["top_k_indices"]) > 0:
        parsed["top_k_indices"] = parsed["top_k_indices"].reshape(seq_len, -1)

    return parsed


@dataclasses.dataclass
class NormalizeFeatures(grain.MapTransform):
  """Normalize text feature keys."""

  def __init__(self, column_names, tokenize):
    self.column_names = column_names
    self.tokenize = tokenize

  def map(self, element):
    if self.tokenize:
      return {col: element[col][0].decode() for col in self.column_names}
    else:
      return {col: element[col] for col in self.column_names}


@dataclasses.dataclass
class KeepFeatures(grain.MapTransform):
  """Filter dataset elements to specified features for parquet and other non-proto formats.

  Retains only the keys present in `feature_names`. Validates the stored value
  type against `tokenize`: raises ValueError if `tokenize=True` but a column
  contains integer data (pre-tokenized), or if `tokenize=False` but a column
  contains string/bytes data (raw text).
  """

  def __init__(self, feature_names: list[str], tokenize: bool = True):
    self.feature_names = feature_names
    self.tokenize = tokenize

  def map(self, element: dict[str, Any]) -> dict[str, Any]:
    """Applies the feature filtering to the input element."""
    missing = [n for n in self.feature_names if n not in element]
    if missing:
      raise ValueError(
          f"Column {missing} not found in dataset. Available columns: {sorted(element.keys())}. "
          "Please set train_data_columns or eval_data_columns accordingly."
      )
    filtered = {k: v for k, v in element.items() if k in self.feature_names}
    for col, val in filtered.items():
      if self.tokenize:
        if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.integer):
          raise ValueError(
              f"tokenize_data=True but column '{col}' contains integer (pre-tokenized) data. "
              "Set tokenize_train_data or tokenize_eval_data to False if your dataset is already tokenized."
          )
        if isinstance(val, (list, tuple)) and val and isinstance(val[0], (int, np.integer)):
          raise ValueError(
              f"tokenize_data=True but column '{col}' contains integer (pre-tokenized) data. "
              "Set tokenize_train_data or tokenize_eval_data to False if your dataset is already tokenized."
          )
      else:
        if isinstance(val, (str, bytes)):
          raise ValueError(
              f"tokenize_data=False but column '{col}' contains text data. "
              "Set tokenize_train_data or tokenize_eval_data to True if your dataset needs tokenization."
          )
    return filtered


@dataclasses.dataclass
class Rekey(grain.MapTransform):
  """Rename keys according to a mapping dict"""

  def __init__(self, mapping_dict, keep_old_keys=False):
    self.mapping_dict = mapping_dict
    self.keep_old_keys = keep_old_keys

  def map(self, element):
    old_keys = set()
    for new_key, old_key in self.mapping_dict.items():
      element[new_key] = element[old_key]
      old_keys.add(old_key)
    if not self.keep_old_keys:
      for key in old_keys:
        del element[key]
    return element


@dataclasses.dataclass
class ReformatPacking(grain.MapTransform):
  """Reformat packing outputs."""

  def __init__(self, column_names):
    self.column_names = column_names

  def map(self, element):
    ret = {}
    for col in self.column_names:
      ret[f"{col}"] = element[0][col]
      ret[f"{col}_segmentation"] = element[1][col]
      ret[f"{col}_position"] = element[2][col]
    return ret


@dataclasses.dataclass
class ExpandToLoadedRows(grain.MapTransform):
  """Place a batch of the trained rows at their positions in the loaded batch; the other rows are zeros.

  Zero rows have zero segmentation, so they carry no loss weight; the loss slices them off anyway.
  """

  def __init__(self, keep):
    self.keep = np.asarray(keep, dtype=bool)

  def map(self, element):
    out = {}
    for k, v in element.items():
      v = np.asarray(v)
      assert v.shape[0] == self.keep.sum(), (k, v.shape, int(self.keep.sum()))
      full = np.zeros((len(self.keep),) + v.shape[1:], dtype=v.dtype)
      full[self.keep] = v
      out[k] = full
    return out


@dataclasses.dataclass
class PadOrTrimToMaxLength(grain.MapTransform):
  """Pads or trims each input to the specified length.
  And optionally add true length for the input."""

  def __init__(
      self,
      max_length: int,
      pad_id: int = 0,
      config=None,
      add_true_length: bool = False,
      max_num_images_per_example: int = -1,
  ):
    self.max_length = max_length
    self.pad_id = pad_id
    self.config = config
    self.add_true_length = add_true_length
    self.max_num_images_per_example = max_num_images_per_example

  def _pad_text(self, x: np.ndarray, max_length: int, pad_id: int) -> np.ndarray:
    pad_amount = max(max_length - x.shape[0], 0)
    pad_amount = [(0, pad_amount)] + [(0, 0)] * (len(x.shape) - 1)
    return np.pad(x, pad_amount, constant_values=pad_id)[: self.max_length]

  def _pad_image_and_mask(self, preprocessed_image: mm_utils.PreprocessorOutput) -> mm_utils.PreprocessorOutput:
    """Pads the input tensors (image and mask) of a PreprocessorOutput to a maximum number of items.

    This function unifies padding logic for image tensors (standard or tiled) and
    mask tensors. It determines the tensor type based on its dimensions and applies
    the appropriate padding along the first axis.

    The maximum number of items is calculated based on model constraints or a
    user-defined limit, ensuring that sequence length limits are respected while
    reserving space for at least one text token. If the input tensor has fewer
    items than this maximum, it is padded with zeros.

    Args:
        preprocessed_image (multimodal_utils.PreprocessorOutput): The input numpy arrays to pad.
            - For masks, the expected shape is (num_masks, num_tiles).
            - For standard images, the shape is (num_images, H, W, C).
            - For tiled images, the shape is (num_images, num_tiles, H, W, C).

    Returns:
        np.ndarray: The tensor, padded with zeros up to the maximum number of
        items along the first axis.

    Raises:
        ValueError: If the input tensor's dimension is not 2, 4, or 5.
        ValueError: If the number of items in the input tensor exceeds the
        allowed maximum.

    Notes:
      - The computation of maximum images ensures that space is reserved in the sequence
        for at least one text token.
      - The dummy images used for padding are based on the image shape for initialization
        of this model (ignoring batch size).
    """
    if not isinstance(preprocessed_image, mm_utils.PreprocessorOutput):
      raise TypeError(f"Input must be multimodal_utils.PreprocessorOutput, but got {type(preprocessed_image)}")

    if getattr(preprocessed_image, "video_values", None) is not None:
      return preprocessed_image

    if preprocessed_image.pixel_values is None:
      raise ValueError("Input preprocessed_image must have pixel_values to pad images.")

    vision_block = mm_processor._get_vision_block(self.config)  # pylint: disable=protected-access
    model_name = getattr(self.config, "model_name", None)
    if (
        vision_block is None
        or not getattr(self.config, "use_multimodal", False)
        or vision_block in ["qwen3_omni", "qwen3_vl", "qwen3_5"]
    ):
      return preprocessed_image
    elif model_name and model_name.startswith("qwen"):
      raise ValueError(
          f"Qwen multimodal model '{model_name}' with vision_block '{vision_block}' was not "
          f"registered in `PadOrTrimToMaxLength`. Qwen models use native dynamic grids and must be "
          f"registered to bypass image padding."
      )

    # Determine the maximum number of images/masks allowed.
    image_offsets = mm_processor.get_image_offsets(self.config, preprocessed_image)
    single_image_offset = image_offsets // preprocessed_image.pixel_values.shape[0]

    # Reserve space for at least one text token.
    max_num_items = (self.max_length - 1) // single_image_offset
    if self.max_num_images_per_example > 0:
      max_num_items = min(self.max_num_images_per_example, max_num_items)

    image_tensor = preprocessed_image.pixel_values
    mask_tensor = preprocessed_image.pixel_mask

    def _pad(tensor: np.ndarray) -> np.ndarray:
      # Validate tensor dimensions.
      if tensor.ndim in (4, 5):  # Standard or Tiled Image
        tensor_type = "images"
      elif tensor.ndim == 2:  # Mask
        tensor_type = "masks"
      else:
        raise ValueError(
            "Input tensor must be 2D (mask), 4D (image), or 5D (tiled image), " f"but got {tensor.ndim} dimensions."
        )

      # Assert that the input tensor does not exceed the maximum size.
      if tensor.shape[0] > max_num_items:
        raise ValueError(f"Number of {tensor_type} ({tensor.shape[0]}) exceeds the maximum allowed ({max_num_items}).")

      # Apply padding if the tensor is smaller than the maximum size.
      if tensor.shape[0] < max_num_items:
        pad_size = max_num_items - tensor.shape[0]
        pad_shape_suffix = tensor.shape[1:]
        pad_shape = (pad_size,) + pad_shape_suffix
        pad_tensor = np.zeros(pad_shape, dtype=tensor.dtype)

        if tensor.size > 0:
          tensor = np.concatenate([tensor, pad_tensor], axis=0)
        else:
          # If the input tensor is empty, the result is just the padding.
          tensor = pad_tensor

      return tensor

    preprocessed_image.pixel_values = _pad(image_tensor)

    if mask_tensor is not None:
      preprocessed_image.pixel_mask = _pad(mask_tensor)

    return preprocessed_image

  def map(
      self, element: dict[str, np.ndarray | mm_utils.PreprocessorOutput]
  ) -> dict[str, np.ndarray | mm_utils.PreprocessorOutput]:
    """map to each element"""
    data_columns = list(element.keys())
    preserve_pad_valued_tokens = (
        self.config is not None and getattr(self.config, "training_objective", "causal_lm") == "block_diffusion"
    )
    for data_column in data_columns:
      if data_column != "images":
        if isinstance(element[data_column], mm_utils.PreprocessorOutput):
          raise TypeError("Only 'images' column can be of type PreprocessorOutput.")

        if preserve_pad_valued_tokens:
          element[f"{data_column}_segmentation"] = np.ones(
              element[data_column].shape[0], dtype=np.int32  # pyrefly: ignore[missing-attribute]
          )
        else:
          element[f"{data_column}_segmentation"] = (
              element[data_column] != self.pad_id  # pyrefly: ignore[unsupported-operation]
          )  # pyrefly: ignore[unsupported-operation]
        # pyrefly: ignore[missing-attribute]
        element[f"{data_column}_segmentation"] = element[
            f"{data_column}_segmentation"
        ].astype(  # pyrefly: ignore[missing-attribute]
            np.int32
        )
        element[f"{data_column}_position"] = np.arange(
            element[data_column].shape[0],
            dtype=np.int32,  # pyrefly: ignore[missing-attribute]
        )  # pyrefly: ignore[missing-attribute]
        if self.add_true_length:
          element[f"{data_column}_true_length"] = np.array(
              [element[data_column].shape[0]],
              dtype=np.int32,  # pyrefly: ignore[missing-attribute]
          )  # pyrefly: ignore[missing-attribute]

    for key, _ in element.items():
      if key == "images":
        if self.config.model_name is None:  # pyrefly: ignore[missing-attribute]
          raise ValueError("model_name must be provided when padding images")

        element["images"] = self._pad_image_and_mask(element["images"])  # pyrefly: ignore[bad-argument-type]

      elif preserve_pad_valued_tokens and key.endswith(("_segmentation", "_position")):
        element[key] = self._pad_text(element[key], self.max_length, 0)  # pyrefly: ignore[bad-argument-type]
      elif "true_length" not in key:
        element[key] = self._pad_text(element[key], self.max_length, self.pad_id)  # pyrefly: ignore[bad-argument-type]
    return element


@dataclasses.dataclass
class ExtractImagesAndMasks(grain.MapTransform):
  """Extracts images and masks from a PreprocessorOutput object.

  This transform is used in multi-modal data pipelines to extract the image
  tensors and their corresponding masks from a PreprocessorOutput object.
  The extracted images and masks are then added to the data element under
  the keys 'images' and 'image_masks', respectively.

  If the 'images' key is not present in the input element, the transform
  returns the element unchanged.
  """

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Applies the extraction transformation to the 'images' field if present."""
    preprocessed_image = element.get("images")
    if preprocessed_image is None:
      return element

    if not isinstance(preprocessed_image, mm_utils.PreprocessorOutput):
      raise TypeError(f"'images' must be of type PreprocessorOutput, but got {type(preprocessed_image)}")

    output = element.copy()
    if getattr(preprocessed_image, "video_values", None) is not None:
      output["images"] = preprocessed_image.video_values
      if getattr(preprocessed_image, "video_mask", None) is not None:
        output["image_masks"] = preprocessed_image.video_mask
      if getattr(preprocessed_image, "video_grid_thw", None) is not None:
        output["video_grid_thw"] = preprocessed_image.video_grid_thw
      if getattr(preprocessed_image, "video_second_per_grid", None) is not None:
        output["second_per_grids"] = preprocessed_image.video_second_per_grid
      return output

    output["images"] = preprocessed_image.pixel_values  # pyrefly: ignore[unsupported-operation]
    if preprocessed_image.pixel_mask is not None:
      output["image_masks"] = preprocessed_image.pixel_mask
    # Qwen MRoPE needs per-image (t, h, w) grids to build 3D positions.
    pixel_grid_thw = getattr(preprocessed_image, "pixel_grid_thw", None)
    if pixel_grid_thw is not None:
      output["image_grid_thw"] = pixel_grid_thw

    return output


@dataclasses.dataclass
class FoldImagesIntoBatch(grain.MapTransform):
  """Folds the 'image' dimension into the batch dimension.

  This transform is used in multi-modal data pipelines where each data example
  might have multiple associated images. For model processing, it's often
  efficient to treat each image as a separate item in a larger batch.

  This operation reshapes the 'images' tensor from a shape like
  (B, N, T, H, W, C) to (B * N, T, H, W, C), where B is the batch size, N is
  the number of images per example, and T is the number of image tiles.

  The transformation is triggered only if the input 'images' tensor has more
  dimensions than the expected batched image tensor.
  """

  model_name: str | None = None

  def __post_init__(self):
    """Initializes the target shape after the dataclass is created."""
    self.target_shape = mm_processor.get_dummy_image_shape_for_init(self.model_name)

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Applies the folding transformation to the 'images' field if present."""
    images = element.get("images")
    if images is None:
      return element

    video_grid_thw = element.get("video_grid_thw")
    if video_grid_thw is not None and images.ndim > len(self.target_shape):
      element["images"] = images.reshape(-1, *images.shape[2:])
      image_masks = element.get("image_masks")
      if image_masks is not None:
        element["image_masks"] = image_masks.reshape(-1, *image_masks.shape[2:])
      element["video_grid_thw"] = video_grid_thw.reshape(-1, video_grid_thw.shape[-1])
      second_per_grids = element.get("second_per_grids")
      if second_per_grids is not None:
        element["second_per_grids"] = second_per_grids.reshape(-1)
      return element

    # If ndim is greater than the expected ndim for a batched image tensor,
    # it implies an extra dimension (e.g., number of images per example)
    # that needs to be folded into the batch dimension.
    if images.ndim > len(self.target_shape):
      # Compute the new shape by merging the batch and image count dimensions.
      trailing_dims = self.target_shape[1:]

      # Reshape merges the leading dimensions (B, N) into one (-1) and
      # appends the correct trailing dimensions.
      element["images"] = images.reshape(-1, *trailing_dims)

    return element


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  slices = [
      slice(None),
  ] * len(x.shape)
  slices[axis] = slice(0, -1)
  padded = np.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(0))
  return padded[tuple(slices)]


def shift_left(x, pad_id, axis=1):
  """Shift to the left and pad."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (0, 1)
  slices = [
      slice(None),
  ] * len(x.shape)
  slices[axis] = slice(1, None)
  padded = np.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(pad_id))
  return padded[tuple(slices)]


def shift_and_refine(x, ignored_ids, axis=1):
  """Left-shift targets and mask labels equal to one of ``ignored_ids``."""
  x["targets"] = shift_left(x["targets"], ignored_ids[0], axis=axis)
  x["targets_segmentation"] = shift_left(x["targets_segmentation"], 0, axis=axis)
  for ignore_id in ignored_ids:
    x["targets_segmentation"] = np.where(x["targets"] != ignore_id, x["targets_segmentation"], 0)

  return x


@dataclasses.dataclass
class ShiftData(grain.MapTransform):
  """Convert copied token sequences into next-token targets after batching.

  ``targets`` and ``targets_segmentation`` are shifted one position to the
  left along ``axis``. The final target is padded with ``ignored_ids[0]`` and
  receives segmentation value zero. Any shifted target equal to an ignored ID
  also receives segmentation value zero, excluding it from loss computation.

  This transform intentionally leaves positions unchanged: input and target
  positions refer to the same prediction positions after the left shift.
  """

  def __init__(self, ignored_ids, axis=1):
    self.ignored_ids = ignored_ids
    self.axis = axis

  def map(self, element):
    return shift_and_refine(element, ignored_ids=self.ignored_ids, axis=self.axis)


def megatron_min_segment_length(config) -> int:
  """Return the Megatron-compatible short-segment merge threshold.

  The threshold is ``max_target_length // packing_max_segments_per_sample``.
  It is meaningful only when attention resets at document boundaries; a
  non-positive divisor disables merging.
  """
  if not config.reset_attention_mask:
    return 0
  divisor = config.packing_max_segments_per_sample
  if divisor <= 0:
    return 0
  return config.max_target_length // divisor


def _merge_short_segments_np(
    segmentation: np.ndarray,
    position: np.ndarray,
    min_seg_len: int,
) -> None:
  """Greedily merge short EOD-derived segments in place.

  This follows Megatron's packed-sequence boundary rule. A candidate boundary
  is retained only when it is strictly more than ``min_seg_len`` tokens after
  the previous retained boundary. Dropped boundaries merge their tokens into
  the preceding retained segment, with segment IDs and position IDs rebuilt
  as a single contiguous sequence.

  Args:
    segmentation: One-dimensional, one-indexed segment IDs.
    position: One-dimensional position IDs corresponding to ``segmentation``.
    min_seg_len: Boundary-merging threshold. Values of zero or one are no-ops.
  """
  seq_len = len(segmentation)
  if min_seg_len <= 1 or seq_len == 0:
    return

  boundaries = np.flatnonzero(np.diff(segmentation)) + 1
  seg_starts = [0, *boundaries.tolist()]

  if len(seg_starts) <= 1:
    return

  kept = [0]
  for start in seg_starts[1:]:
    if start - kept[-1] > min_seg_len:
      kept.append(start)

  if len(kept) == len(seg_starts):
    return

  kept.append(seq_len)
  for seg_idx in range(len(kept) - 1):
    s, e = kept[seg_idx], kept[seg_idx + 1]
    segmentation[s:e] = seg_idx + 1
    position[s:e] = np.arange(e - s, dtype=position.dtype)


@dataclasses.dataclass
class GenerateDocSegmentIds(grain.MapTransform):
  """Generate EOD-aware segmentation and position arrays for ``mmap`` samples.

  The input data must already contain document-ending EOD tokens, normally
  from Megatron preprocessing with ``--append-eod``. ``MMapSampleIndexDataSource``
  concatenates and windows those tokens; it does not insert EOD tokens itself.
  For every input token field, this transform adds ``<field>_segmentation``
  and ``<field>_position``.

  Args:
    eod_id: Token ID that marks the end of a document.
    reset_attention_mask: Controls document-boundary attention and positions.

      * ``True`` (default) -- attention resets at every document boundary.
        EOD belongs to the preceding document (same segment ID), and a new
        segment starts after EOD.  Positions continue through EOD and reset
        after EOD.

        ::

            tokens:        [tok tok tok EOD tok tok EOD tok tok tok tok tok]
            segmentation:  [ 1   1   1   1   2   2   2   3   3   3   3   3]
            positions:     [ 0   1   2   3   0   1   2   0   1   2   3   4]

      * ``False`` -- cross-document attention is allowed. Positions are a
        continuous ``arange`` over the full sample, and all tokens use
        segment ID ``1`` except masked EOD positions.

    eod_mask_loss: When ``reset_attention_mask`` is False, controls whether
      EOD tokens receive segmentation value zero and are excluded from loss.
      In the ``mmap`` pipeline, ``ShiftData`` subsequently masks shifted EOD
      labels because EOD is also the batch padding sentinel.
    min_segment_length: Optional threshold used to merge adjacent short
      EOD-derived segments when attention resets are enabled.
  """

  def __init__(
      self, eod_id: int, reset_attention_mask: bool = True, eod_mask_loss: bool = False, min_segment_length: int = 0
  ):
    self.eod_id = eod_id
    self.reset_attention_mask = reset_attention_mask
    self.eod_mask_loss = eod_mask_loss
    self.min_segment_length = min_segment_length

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply EOD-based segmentation and loss masking to each column."""
    for col, tokens in list(element.items()):
      seq_len = tokens.shape[0]
      is_eod = tokens == self.eod_id

      if self.reset_attention_mask:
        # EOD belongs to the preceding document: keep current seg_id and
        # continue position counter.  New segment starts AFTER EOD.
        segmentation = np.zeros(seq_len, dtype=np.int32)
        position = np.zeros(seq_len, dtype=np.int32)
        seg_id = 1
        pos_in_doc = 0
        for i in range(seq_len):
          if is_eod[i]:
            # EOD keeps the preceding document's seg_id and position
            segmentation[i] = seg_id
            position[i] = pos_in_doc
            pos_in_doc += 1
            # New segment starts after EOD
            seg_id += 1
            pos_in_doc = 0
          else:
            segmentation[i] = seg_id
            position[i] = pos_in_doc
            pos_in_doc += 1
        if self.min_segment_length > 0:
          _merge_short_segments_np(segmentation, position, self.min_segment_length)
      else:
        if self.eod_mask_loss:
          segmentation = np.where(is_eod, np.int32(0), np.int32(1))
        else:
          segmentation = np.ones(seq_len, dtype=np.int32)
        position = np.arange(seq_len, dtype=np.int32)

      element[f"{col}_segmentation"] = segmentation
      element[f"{col}_position"] = position
    return element


@dataclasses.dataclass
class MegatronSplitInputsTargets(grain.MapTransform):
  """Build Megatron-compatible pretraining fields from an ``L + 1`` sample.

  Input is ``{"text": tokens}``, where ``len(tokens) == L + 1``. The output
  contains length-``L`` fields with ``inputs = tokens[:-1]`` and
  ``targets = tokens[1:]``. Unlike a padded left-shift, this preserves the
  final real target and its loss contribution.

  This gives valid prediction targets at every position without padding,
  unlike the ShiftData approach which wastes the last position.

  Attention, loss, and position annotations are constructed independently to
  match Megatron's separate attention-mask, loss-mask, and position-ID rules:

  - ``inputs_segmentation`` (attention): controlled by ``reset_attention_mask``.
    When True, EOD belongs to its preceding document (same segment ID) and
    a new segment starts after EOD; when False, all tokens share segment 1.
  - ``targets_segmentation`` (loss): controlled by ``eod_mask_loss`` only.
    When True, positions where ``inputs == eod_id`` get segmentation=0
    (excluded from loss); when False, all positions get segmentation=1.
  - ``inputs_position`` (RoPE): controlled by ``reset_attention_mask``.
    When True, EOD continues the preceding document's position counter and
    resets to 0 after EOD; when False, positions are sequential.

  Args:
    eod_id: Token ID marking the end of a document.
    reset_attention_mask: When True, EOD ends the current attention segment
      and positions restart at zero for the following token.
    eod_mask_loss: When True, positions whose input token is EOD have loss
      mask zero.
    no_attnmask_dataset_ids: Optional dataset IDs for samples that disable
      attention-mask reset regardless of the global setting.
    min_segment_length: Optional threshold for merging short EOD-derived
      attention segments.
    emit_dataset_id: Whether to emit a per-token ``dataset_id`` field when
      the source sample provides one.
  """

  def __init__(
      self,
      eod_id: int,
      reset_attention_mask: bool = True,
      eod_mask_loss: bool = False,
      no_attnmask_dataset_ids: set[int] | None = None,
      min_segment_length: int = 0,
      emit_dataset_id: bool = False,
  ):
    self.eod_id = eod_id
    self.reset_attention_mask = reset_attention_mask
    self.eod_mask_loss = eod_mask_loss
    self.no_attnmask_dataset_ids = no_attnmask_dataset_ids or set()
    self._has_no_attnmask = bool(self.no_attnmask_dataset_ids)
    self.min_segment_length = min_segment_length
    self.emit_dataset_id = emit_dataset_id

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Split tokens into input/target pairs with EOD-based segmentation."""
    tokens = element["text"]
    inputs = tokens[:-1]
    targets = tokens[1:]

    seq_len = inputs.shape[0]
    is_eod = inputs == self.eod_id

    # Per-sample override: datasets in no_attnmask_dataset_ids bypass
    # attention mask reset (antllm no_attnmask_data behavior).
    effective_reset = self.reset_attention_mask
    if self._has_no_attnmask:
      dataset_id = element.get("dataset_id", None)
      if dataset_id is not None and int(dataset_id) in self.no_attnmask_dataset_ids:
        effective_reset = False

    # --- inputs_segmentation (attention mask) ---
    if effective_reset:
      input_segmentation = np.zeros(seq_len, dtype=np.int32)
      position = np.zeros(seq_len, dtype=np.int32)
      seg_id = 1
      pos_in_doc = 0
      for i in range(seq_len):
        if is_eod[i]:
          # EOD belongs to the preceding document: keep current seg_id,
          # continue position counter.  New segment starts AFTER EOD.
          input_segmentation[i] = seg_id
          position[i] = pos_in_doc
          pos_in_doc = 0
          seg_id += 1
        else:
          input_segmentation[i] = seg_id
          position[i] = pos_in_doc
          pos_in_doc += 1
      if self.min_segment_length > 0:
        _merge_short_segments_np(input_segmentation, position, self.min_segment_length)
    else:
      input_segmentation = np.ones(seq_len, dtype=np.int32)
      position = np.arange(seq_len, dtype=np.int32)

    # --- targets_segmentation (loss mask) ---
    # Independent of reset_attention_mask, matching Megatron's separate
    # loss_mask which only depends on eod_mask_loss.
    if self.eod_mask_loss:
      target_segmentation = np.where(is_eod, np.int32(0), np.int32(1))
    else:
      target_segmentation = np.ones(seq_len, dtype=np.int32)

    result = {
        "inputs": inputs,
        "targets": targets,
        "inputs_segmentation": input_segmentation,
        "targets_segmentation": target_segmentation,
        "inputs_position": position,
        "targets_position": position,
    }
    if self.emit_dataset_id and "dataset_id" in element:
      result["dataset_id"] = np.full(seq_len, int(element["dataset_id"]), dtype=np.int32)
    return result


@dataclasses.dataclass
class BlockDiffusionCorruption(grain.RandomMapTransform):
  """Adapts block-diffusion corruption to the Grain batch contract."""

  block_size: int
  mask_id: int
  min_noise: float = 1.0e-3
  logit_alignment: str = "same_position"
  canvas_policy: str = "all_masked"
  axis: int = 1

  def random_map(self, element, rng: np.random.Generator):
    """Corrupts inputs while preserving clean targets and input metadata."""
    inputs = np.asarray(element["inputs"])
    targets = np.asarray(element["targets"])
    targets_segmentation = np.asarray(element["targets_segmentation"])
    if inputs.shape != targets.shape or inputs.shape != targets_segmentation.shape:
      raise ValueError(
          "inputs, targets, and targets_segmentation must have identical shapes, got "
          f"{inputs.shape}, {targets.shape}, and {targets_segmentation.shape}"
      )
    result = block_diffusion_corruption.corrupt_tokens(
        inputs,
        targets_segmentation != 0,
        rng,
        block_size=self.block_size,
        mask_id=self.mask_id,
        min_noise=self.min_noise,
        logit_alignment=self.logit_alignment,
        canvas_policy=self.canvas_policy,
        axis=self.axis,
    )
    output = dict(element)
    output["inputs"] = result.inputs
    output["targets"] = targets
    output["corruption_mask"] = result.corruption_mask.astype(targets_segmentation.dtype)
    output["targets_loss_mask"] = result.targets_loss_mask.astype(targets_segmentation.dtype)
    return output


@dataclasses.dataclass
class ComputeQwen3OmniPositions(grain.MapTransform):
  """Computes 3D position IDs for Qwen3-Omni multimodal sequences.

  This transform replaces the standard 1D sequential positions with 3D
  positions (temporal, height, width) for multimodal models like Qwen3-Omni.

  For text-only sequences, all 3 dimensions receive the same sequential values.
  For multimodal sequences with vision/audio, vision tokens get true 3D positions
  and text tokens continue sequentially from max(vision_pos) + 1.

  The actual position computation is delegated to multimodal_utils.get_rope_index(),
  which can be tested and modified independently.
  """

  def __init__(
      self,
      data_column: str = "inputs",
      spatial_merge_size: int = 2,
      position_id_per_seconds: int = 25,
      use_audio_in_video: bool = False,
      config=None,
      keep_aux_fields: bool = False,
  ):
    """Initialize the Qwen3-Omni position computation transform.

    Args:
      data_column: Name of the data column to compute positions for (default: "inputs").
      spatial_merge_size: Number of patches merged spatially (e.g., 2 for 2x2→1).
      position_id_per_seconds: Temporal granularity (tokens per second, typically 25).
      use_audio_in_video: If True, audio tokens are interleaved with video tokens.
      config: Optional model config for model-specific special token IDs.
      keep_aux_fields: If True, keep auxiliary (aux) fields — mrope deltas / grid
        metadata — on the element for inference. Training should leave this False
        so the batch matches get_shaped_batch.
    """
    self.data_column = data_column
    self.spatial_merge_size = spatial_merge_size
    self.position_id_per_seconds = position_id_per_seconds
    self.use_audio_in_video = use_audio_in_video
    self.config = config
    self.keep_aux_fields = keep_aux_fields

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute 3D position IDs for the batch element.

    Args:
      element: Dictionary containing:
        - {data_column}: Token IDs with shape (batch, seq_len)
        - {data_column}_segmentation: Attention mask (1=real, 0=padding)
        - image_grid_thw: Optional (num_images, 3) or (batch, num_images, 3) array
        - video_grid_thw: Optional (num_videos, 3) array
        - audio_lengths: Optional (num_audios,) array
        - second_per_grids: Optional (num_videos,) array

    Returns:
      element with {data_column}_position updated to shape (batch, seq_len, 3)
      for 3D positions (always 3D, even for text-only sequences).
    """

    # Extract inputs and metadata
    input_ids = element[self.data_column]
    attention_mask = element.get(f"{self.data_column}_segmentation")

    # Extract multimodal metadata (if present)
    image_grid_thw = element.get("image_grid_thw")
    video_grid_thw = element.get("video_grid_thw")
    audio_lengths = element.get("audio_lengths")
    second_per_grids = element.get("second_per_grids")

    # grain.Batch stacks per-example (N, 3) grids to (B, N, 3). get_rope_index
    # resets image_idx per sequence against a shared (N, 3) table, which is
    # correct when training force-resizes all images to the same grid.
    if image_grid_thw is not None and image_grid_thw.ndim == 3:
      image_grid_thw = image_grid_thw[0]

    # Call the standalone get_rope_index function from multimodal_utils
    from maxtext.multimodal import processor_qwen3_omni  # pylint: disable=import-outside-toplevel

    # TODO(jfacevedo/hengtaoguo): Now get_rope_index is Qwen3-Omni specific. We should generalize it for other models
    position_ids, mrope_position_deltas = processor_qwen3_omni.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        use_audio_in_video=self.use_audio_in_video,
        audio_lengths=audio_lengths,
        second_per_grids=second_per_grids,
        spatial_merge_size=self.spatial_merge_size,
        position_id_per_seconds=self.position_id_per_seconds,
        config=self.config,
    )

    # Update element with 3D positions
    # Shape: (batch, seq_len, 3) for multimodal, or (batch, seq_len) for text-only
    element[f"{self.data_column}_position"] = position_ids.astype(np.int32)
    if self.keep_aux_fields:
      element[f"{self.data_column}_mrope_deltas"] = mrope_position_deltas
    else:
      # Drop metadata that is not part of the training shaped batch.
      element.pop(f"{self.data_column}_mrope_deltas", None)
      element.pop("image_grid_thw", None)
      if getattr(self.config, "video_max_grid_t", None) is None:
        element.pop("video_grid_thw", None)
      element.pop("audio_lengths", None)
      element.pop("second_per_grids", None)

    return element
