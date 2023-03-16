# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
# ==============================================================================

"""
python run_eval.py ../HuggingFaceModels/distilbart-xsum-12-3 <path>/<to/<file>
"""

import json
from pathlib import Path
from typing import Dict, List
import sys
import time

from bart.config.config import BartConfig
from bart.tokenizer.tokenization import BartTokenizer
from bart.model.modeling import BartForConditionalGeneration

model_path = sys.argv[1]
source_file = sys.argv[2]
max_tokens = 512

config = BartConfig(**json.load(open(f"{model_path}/config.json", "r")))
model = BartForConditionalGeneration.from_pretrained(model_path, config=config)
tokenizer = BartTokenizer.from_pretrained(model_path)

model.eval()

print(f"model_path: {model_path}")
print(f"source_file: {source_file}")

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
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
#            attention_mask=inputs["attention_mask"],
            use_cache=True, # past_key_values
            num_return_sequences=1,
            num_beams=5,
            max_length=142,
            output_scores=False,
            return_dict_in_generate=False,
            encoder_no_repeat_ngram_size=0,
            diversity_penalty=0.0
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
