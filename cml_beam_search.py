# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import logging

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from utils.file_utils import ModelOutput

from bart.utils.generation_beam_search import BeamScorer, BeamSearchScorer
from bart.utils.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)
from bart.model.outputs import (
    BaseModelOutput
)

import sys
import time

import coremltools as ct

encoder = ct.models.MLModel("encoder.mlpackage")
decoder0 = ct.models.MLModel("decoder0.mlpackage")
decoder = ct.models.MLModel("decoder.mlpackage")

device = torch.device("cpu")

logger = logging.getLogger(__name__)

@dataclass
class BeamSearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using beam search.

    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        sequences_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_return_sequences)`, `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Final beam scores of the generated ``sequences``.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam
            . :obj:`(max_length,)`-shaped tuple of :obj:`torch.FloatTensor` with each tensor of shape
            :obj:`(batch_size*num_beams*num_return_sequences, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams, num_heads, generated_length,
            sequence_length)`.
        hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams*num_return_sequences, generated_length,
            hidden_size)`.
    """

    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class BeamSearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using beam search. Hidden states and attention weights
    of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
    attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)

    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        sequences_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_return_sequences)`, `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Final beam scores of the generated ``sequences``.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam
            . :obj:`(max_length,)`-shaped tuple of :obj:`torch.FloatTensor` with each tensor of shape
            :obj:`(batch_size*num_beams, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer of the decoder) of shape :obj:`(batch_size,
            num_heads, sequence_length, sequence_length)`.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams*num_return_sequences, num_heads,
            generated_length, sequence_length)`.
        decoder_hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams*num_return_sequences, generated_length,
            hidden_size)`.
    """

    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]

def _get_logits_processor(
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    encoder_no_repeat_ngram_size: int,
    encoder_input_ids: torch.LongTensor,
    bad_words_ids: List[List[int]],
    min_length: int,
    eos_token_id: int,
    prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
    num_beams: int,
    diversity_penalty: float,
) -> LogitsProcessorList:
    """
    This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
    :obj:`~transformers.LogitsProcessor` instances used to modify the scores of the language model head.
    """

    # init warp parameters
    repetition_penalty = repetition_penalty if repetition_penalty is not None else None
    no_repeat_ngram_size = (
        no_repeat_ngram_size if no_repeat_ngram_size is not None else None
    )
    encoder_no_repeat_ngram_size = (
        encoder_no_repeat_ngram_size
        if encoder_no_repeat_ngram_size is not None
        else None
    )
    bad_words_ids = bad_words_ids if bad_words_ids is not None else None
    min_length = min_length if min_length is not None else None
    eos_token_id = eos_token_id if eos_token_id is not None else None
    diversity_penalty = diversity_penalty if diversity_penalty is not None else None
    # instantiate processors list
    processors = LogitsProcessorList()

    if repetition_penalty is not None and repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if encoder_no_repeat_ngram_size is not None and encoder_no_repeat_ngram_size > 0:
        processors.append(EncoderNoRepeatNGramLogitsProcessor(encoder_no_repeat_ngram_size, encoder_input_ids))
    if bad_words_ids is not None:
        processors.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
    if min_length is not None and eos_token_id is not None and min_length > -1:
        processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
    if prefix_allowed_tokens_fn is not None:
        processors.append(PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams))
    return processors



@torch.no_grad()
def generate(
    input_ids: Optional[torch.LongTensor] = None,
    max_length: Optional[int] = None, # must
    min_length: Optional[int] = None,
    early_stopping: Optional[bool] = None,
    num_beams: Optional[int] = None, # must
    repetition_penalty: Optional[float] = None,
    bad_words_ids: Optional[Iterable[int]] = None,
    bos_token_id: Optional[int] = None, # must
    pad_token_id: Optional[int] = None, # must
    eos_token_id: Optional[int] = None, # must
    length_penalty: Optional[float] = None,
    no_repeat_ngram_size: Optional[int] = None,
    encoder_no_repeat_ngram_size: Optional[int] = None,
    num_return_sequences: Optional[int] = None,
    decoder_start_token_id: Optional[int] = None,
    diversity_penalty: Optional[float] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    **model_kwargs,
) -> Union[BeamSearchOutput, torch.LongTensor]:

    # set init values
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else None
    )

    output_scores = output_scores if output_scores is not None else False
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else False
    )

    # init `attention_mask`
    attention_mask = input_ids.new_ones(input_ids.shape)

    # Storing encoder_input_ids for logits_processor that could use them
    encoder_input_ids = input_ids

    # kick encoder, and get last_hidden_state for the requested input_ids
    encoder_outputs = encoder.predict({
        "input_ids": input_ids.to(torch.int32).numpy()
    })
    last_hidden_state = torch.from_numpy(encoder_outputs["hidden_states"])
    print("last_hidden_state", last_hidden_state)
    print("last_hidden_state", last_hidden_state.shape)

    # set input_ids as decoder's input_ids
    input_ids = (
        torch.ones((input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device)
        * eos_token_id
    )

    if input_ids.shape[-1] >= max_length:
        logger.warning(
            f"Length of input_ids is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}."
            "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
        )

    # get distribution pre_processing samplers
    logits_processor = _get_logits_processor(
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
        encoder_input_ids=encoder_input_ids,
        bad_words_ids=bad_words_ids,
        min_length=min_length,
        eos_token_id=eos_token_id,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=num_beams,
        diversity_penalty=diversity_penalty,
    )

    batch_size = input_ids.shape[0]

    length_penalty = length_penalty if length_penalty is not None else None
    early_stopping = early_stopping if early_stopping is not None else False

    if num_return_sequences > num_beams:
        raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

    beam_scorer = BeamSearchScorer(
        batch_size=batch_size,
        max_length=max_length,
        num_beams=num_beams,
        device=device,
        length_penalty=length_penalty,
        do_early_stopping=early_stopping,
        num_beam_hyps_to_keep=num_return_sequences,
    )

    # interleave with `num_beams`
    expanded_return_idx = (
        torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, num_beams).view(-1).to(input_ids.device)
    )
    input_ids = input_ids.index_select(0, expanded_return_idx)

    attention_mask = attention_mask.index_select(0, expanded_return_idx)
    last_hidden_state = last_hidden_state.index_select(
        0, expanded_return_idx.to(last_hidden_state.device)
    )

    # ********** start beam search **********

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None

#    batch_size = len(beam_scorer._beam_hyps)
#    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    assert (
        num_beams * batch_size == batch_beam_size
    ), "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    past_keys = None
    past_values = None

    cnt = 0

    while cur_len < max_length:
        decoder_input_ids = input_ids[:, -1:]

        if cnt == 0:
            decoder_outputs = decoder0.predict({
                "input_ids": decoder_input_ids.to(torch.int32).numpy(), # [bsz, 1] = [bsz, <last token of current prediction>]
                "attention_mask": attention_mask.to(torch.int32).numpy(), # always [batch_size, seq_len]
                "last_hidden_state": last_hidden_state, # [bsz, seq_len, hidden_size]
            })
        else:
            # decoder_outputs = lm_logits, hidden_states, present_keys, present_values
            decoder_outputs = decoder.predict({
                "input_ids": decoder_input_ids.to(torch.int32).numpy(), # [bsz, 1] = [bsz, <last token of current prediction>]
                "attention_mask": attention_mask.to(torch.int32).numpy(), # always [batch_size, seq_len]
                "last_hidden_state": last_hidden_state, # [bsz, seq_len, hidden_size]
                "past_keys_1": past_keys[0].to(torch.int32).numpy() if past_keys is not None else None,
                "past_keys_2": past_keys[1].to(torch.int32).numpy() if past_keys is not None else None,
                "past_keys_3": past_keys[2].to(torch.int32).numpy() if past_keys is not None else None,
                "past_values_1": past_values[0].to(torch.int32).numpy() if past_values is not None else None,
                "past_values_2": past_values[1].to(torch.int32).numpy() if past_values is not None else None,
                "past_values_3": past_values[2].to(torch.int32).numpy() if past_values is not None else None,
            })
        
        #print("decoder_outputs", decoder_outputs)
        lm_logits = torch.from_numpy(decoder_outputs["lm_logits"])
        #print("lm_logits", lm_logits.shape)
        hidden_states = torch.from_numpy(decoder_outputs["hidden_states"])
        # get tensor
        present_keys = (
            torch.from_numpy(decoder_outputs["present_keys_1"]), 
            torch.from_numpy(decoder_outputs["present_keys_2"]), 
            torch.from_numpy(decoder_outputs["present_keys_3"])
        )
        present_values = (
            torch.from_numpy(decoder_outputs["present_values_1"]), 
            torch.from_numpy(decoder_outputs["present_values_2"]), 
            torch.from_numpy(decoder_outputs["present_values_3"])
        )

        next_token_logits = lm_logits[:, -1, :]

        next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
        next_token_scores = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )

        next_indices = next_tokens // vocab_size
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        cur_len = cur_len + 1

        # update past
        reordered_present_keys = tuple(present_key.index_select(0, beam_idx) for present_key in present_keys)
        reordered_present_values = tuple(present_value.index_select(0, beam_idx) for present_value in present_values)
        past_keys = reordered_present_keys
        past_values = reordered_present_values

        cnt += 1

        if beam_scorer.is_done:
            break

    sequence_outputs = beam_scorer.finalize(
        input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None
        return BeamSearchEncoderDecoderOutput(
            sequences=sequence_outputs["sequences"],
            sequences_scores=sequence_outputs["sequence_scores"],
            scores=scores,
        )
    else:
        return sequence_outputs["sequences"]


"""
# rquired
# pad_token_id 1
# bos_token_id 0
# eos_token_id 2
# decoder_start_token_id 2

python cml_beam_search.py ../SampleTexts/news.txt

"""

from bart.tokenizer.tokenization import BartTokenizer
tokenizer = BartTokenizer.from_pretrained("../HuggingFaceModels/distilbart-xsum-12-3")
source_file = sys.argv[1]

# generator parameters
max_length = 142
min_length = 11
early_stopping = True
num_beams = 5
repetition_penalty = None
bad_words_ids = None
bos_token_id = 0
pad_token_id = 1
eos_token_id = 2
length_penalty = 1.0
no_repeat_ngram_size = 3
encoder_no_repeat_ngram_size = 0
num_return_sequences = 1
decoder_start_token_id = 2
diversity_penalty = 0.0
prefix_allowed_tokens_fn = None
output_scores = False
return_dict_in_generate = False

# summarizer parameters
#max_tokens = 960
max_tokens = 512

with open(source_file, 'r') as f:
    text = f.read().replace('\n', '')

def summarize(text):
    # split into sentences
    sentences = text.split('.')
    # trim heading and trailing whitespace
    sentences = [s.strip() for s in sentences]
    # join into "max max_tokens" tokens (using the tokenizer)
    texts = []
    for sentence in sentences:
        if len(texts) == 0:
            texts.append(sentence)
        else:
            if len(tokenizer(texts[-1] + sentence).input_ids) < max_tokens:
                texts[-1] += sentence
            else:
                texts.append(sentence)

    # summarize each text
    preds = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        #print(f"inputs: {inputs}")
        #print(f"input_ids: {inputs['input_ids']}")

        summary_ids = generate(
            input_ids=inputs["input_ids"],
            max_length = max_length,
            min_length = min_length,
            early_stopping = early_stopping,
            num_beams = num_beams,
            repetition_penalty = repetition_penalty,
            bad_words_ids = bad_words_ids,
            bos_token_id = bos_token_id,
            pad_token_id = pad_token_id,
            eos_token_id = eos_token_id,
            length_penalty = length_penalty,
            no_repeat_ngram_size = no_repeat_ngram_size,
            encoder_no_repeat_ngram_size = encoder_no_repeat_ngram_size,
            num_return_sequences = num_return_sequences,
            decoder_start_token_id = decoder_start_token_id,
            diversity_penalty = diversity_penalty,
            prefix_allowed_tokens_fn = prefix_allowed_tokens_fn,
            output_scores = output_scores,
            return_dict_in_generate = return_dict_in_generate,
        )
        print(f"summary_ids: {summary_ids}")
        pred = tokenizer.batch_decode(
            summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        preds.append(pred)

    # join the summaries with whitespace
    pred = ' '.join(preds)

    # if the summary is longer than "max_tokens" tokens, recursively call summarize
    summarized_tokens = len(tokenizer(pred).input_ids)
    print(f"summarized_tokens: {summarized_tokens}")
    if summarized_tokens > max_tokens:
        print("recursively calling summarize ...")
        pred = summarize(pred)

    return pred


start_time = time.time()

pred = summarize(text)

end_time = time.time()

print(f"elapsed time: {end_time - start_time}")
print(f"pred: {pred}")

