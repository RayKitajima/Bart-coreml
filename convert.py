
"""
python convert.py ../HuggingFaceModels/distilbart-xsum-12-3
"""

import json
import time
import sys
import torch
import numpy as np
import coremltools as ct

import collections

from bart.config.config import BartConfig
from bart.tokenizer.tokenization import BartTokenizer
from bart.model.modeling import BartForConditionalGeneration

model_path = sys.argv[1]
model_name = model_path.split('/')[-1]

config = BartConfig(**json.load(open(f"{model_path}/config.json", "r")))
model = BartForConditionalGeneration.from_pretrained(model_path, config=config)
tokenizer = BartTokenizer.from_pretrained(model_path)

model.eval()

print(f"model_path: {model_path}")

# dummy inputs

batch_size = 5
max_length = 1022
pad_token_id = 1
bos_token_id = 0
eos_token_id = 2
dummy_token_id = 5
num_heads = 16
embed_size_per_head = 64 # hidden_size / num_heads
hidden_size = 1024 # model.config.d_model

# input_ids [batch_size, max_length]
# max length is 1024 = 1022 + 2 for bos and eos
input_ids = torch.tensor([[dummy_token_id] * max_length])
print(f"input_ids.shape: {input_ids.shape}")

# attention_mask [batch_size, max_length]
attention_mask = torch.ones([batch_size, max_length])
print(f"attention_mask.shape: {attention_mask.shape}")

# decoder_input_ids [batch_size, 1]
decoder_input_ids = torch.tensor([[bos_token_id]] * batch_size)
print(f"decoder_input_ids.shape: {decoder_input_ids.shape}")

# past_keys [batch_size, num_heads, encoder_sequence_length, embed_size_per_head] * num decoder layers
past_keys_1 = torch.zeros([batch_size, num_heads, max_length, embed_size_per_head])
past_keys_2 = torch.zeros([batch_size, num_heads, max_length, embed_size_per_head])
past_keys_3 = torch.zeros([batch_size, num_heads, max_length, embed_size_per_head])
print(f"past_keys_1.shape: {past_keys_1.shape}")
print(f"past_keys_2.shape: {past_keys_2.shape}")
print(f"past_keys_3.shape: {past_keys_3.shape}")

# past_values [batch_size, num_heads, encoder_sequence_length, embed_size_per_head] * num decoder layers
past_values_1 = torch.zeros([batch_size, num_heads, max_length, embed_size_per_head])
past_values_2 = torch.zeros([batch_size, num_heads, max_length, embed_size_per_head])
past_values_3 = torch.zeros([batch_size, num_heads, max_length, embed_size_per_head])
print(f"past_values_1.shape: {past_values_1.shape}")
print(f"past_values_2.shape: {past_values_2.shape}")
print(f"past_values_3.shape: {past_values_3.shape}")

# last_hidden_state [batch_size, max_length, hidden_size]
last_hidden_state = torch.zeros([batch_size, max_length, hidden_size])
print(f"last_hidden_state.shape: {last_hidden_state.shape}")


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
        )
        return encoder_outputs[0] # last_hidden_state


class DecoderWrapperFirst(torch.nn.Module):
    def __init__(self, decoder, lm_head, final_logits_bias):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.final_logits_bias = final_logits_bias

    def forward(self, decoder_input_ids, attention_mask, last_hidden_state):
        # propagate last_hidden_state(=encoder_outputs[0]) across the decoder's cross-attention layers to prepare key/value states
        self.decoder.set_encoder_key_value_states(last_hidden_state)

        # decoder_outputs = (hidden_states, present_keys, present_values)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=attention_mask
        )
        lm_logits = self.lm_head(decoder_outputs[0]) + self.final_logits_bias

        return (lm_logits, decoder_outputs)


class DecoderWrapper(torch.nn.Module):
    def __init__(self, decoder, lm_head, final_logits_bias):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.final_logits_bias = final_logits_bias

    def forward(self, decoder_input_ids, attention_mask, last_hidden_state, past_keys_1, past_keys_2, past_keys_3, past_values_1, past_values_2, past_values_3):
        past_keys = (past_keys_1, past_keys_2, past_keys_3)
        past_values = (past_values_1, past_values_2, past_values_3)

        # propagate last_hidden_state(=encoder_outputs[0]) across the decoder's cross-attention layers to prepare key/value states
        self.decoder.set_encoder_key_value_states(last_hidden_state)

        # decoder_outputs = (hidden_states, present_keys, present_values)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            past_keys=past_keys,
            past_values=past_values,
        )
        lm_logits = self.lm_head(decoder_outputs[0]) + self.final_logits_bias

        return (lm_logits, decoder_outputs)


def convert_encoder():
    # get wrapped encoder
    encoder = EncoderWrapper(model.get_encoder())
    encoder.eval()

    # jit trace
    print("# tracing encoder ... ")
    traced_model = torch.jit.trace(encoder, input_ids)
    print("# traced.")

    print("# converting encoder ...")
    converted_model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(shape=ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=max_length, default=max_length))), dtype=np.float32, name="input_ids"),
            #ct.TensorType(shape=ct.Shape(shape=(1, 1024)), dtype=np.float32, name="input_ids"),
        ],
        minimum_deployment_target=ct.target.iOS16,
    )
    converted_model.save("encoder.mlpackage")
    print("# converted.")
    # print("# compressing decoder ...")
    # compressed_model = ct.compression_utils.affine_quantize_weights(converted_model)
    # compressed_model.save("encoder_compressed.mlpackage")
    # print("# compressed.")


def convert_decoder_first():
    # get wrapped decoder
    decoder = DecoderWrapperFirst(model.get_decoder(), model.lm_head, model.final_logits_bias)
    decoder.eval()
    
    # jit trace
    print("# tracing decoder ... ")
    traced_model = torch.jit.trace(decoder, (decoder_input_ids, attention_mask, last_hidden_state))
    print("# traced.")

    print("# converting decoder ...")
    converted_model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(shape=(batch_size, 1), dtype=np.float32, name="input_ids"),
            ct.TensorType(shape=(batch_size, ct.RangeDim(lower_bound=1, upper_bound=max_length, default=max_length)), dtype=np.float32, name="attention_mask"),
            ct.TensorType(shape=(batch_size, ct.RangeDim(lower_bound=1, upper_bound=max_length, default=max_length), hidden_size), dtype=np.float32, name="last_hidden_state")
        ],
        # lm_logits, hidden_states, present_keys, present_values
        outputs=[
            ct.TensorType(name="lm_logits"),
            ct.TensorType(name="hidden_states"),
            ct.TensorType(name="present_keys_1"),
            ct.TensorType(name="present_keys_2"),
            ct.TensorType(name="present_keys_3"),
            ct.TensorType(name="present_values_1"),
            ct.TensorType(name="present_values_2"),
            ct.TensorType(name="present_values_3"),
        ],
        minimum_deployment_target=ct.target.iOS16,
    )
    converted_model.save("decoder0.mlpackage")
    print("# converted.")
    # print("# compressing decoder ...")
    # compressed_model = ct.compression_utils.affine_quantize_weights(converted_model)
    # compressed_model.save("decoder0_compressed.mlpackage")
    # print("# compressed.")


def convert_decoder():
    # get wrapped decoder
    decoder = DecoderWrapper(model.get_decoder(), model.lm_head, model.final_logits_bias)
    decoder.eval()
    
    # jit trace
    print("# tracing decoder ... ")
    traced_model = torch.jit.trace(decoder, (decoder_input_ids, attention_mask, last_hidden_state, past_keys_1, past_keys_2, past_keys_3, past_values_1, past_values_2, past_values_3))
    print("# traced.")

    print("# converting decoder ...")
    converted_model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(shape=(batch_size, 1), dtype=np.float32, name="input_ids"),
            ct.TensorType(shape=(batch_size, ct.RangeDim(lower_bound=1, upper_bound=max_length, default=max_length)), dtype=np.float32, name="attention_mask"),
            ct.TensorType(shape=(batch_size, ct.RangeDim(lower_bound=1, upper_bound=max_length, default=max_length), hidden_size), dtype=np.float32, name="last_hidden_state"),
            ct.TensorType(shape=(batch_size, num_heads, ct.RangeDim(lower_bound=1, upper_bound=max_length, default=max_length), embed_size_per_head), dtype=np.float32, name="past_keys_1"),
            ct.TensorType(shape=(batch_size, num_heads, ct.RangeDim(lower_bound=1, upper_bound=max_length, default=max_length), embed_size_per_head), dtype=np.float32, name="past_keys_2"),
            ct.TensorType(shape=(batch_size, num_heads, ct.RangeDim(lower_bound=1, upper_bound=max_length, default=max_length), embed_size_per_head), dtype=np.float32, name="past_keys_3"),
            ct.TensorType(shape=(batch_size, num_heads, ct.RangeDim(lower_bound=1, upper_bound=max_length, default=max_length), embed_size_per_head), dtype=np.float32, name="past_values_1"),
            ct.TensorType(shape=(batch_size, num_heads, ct.RangeDim(lower_bound=1, upper_bound=max_length, default=max_length), embed_size_per_head), dtype=np.float32, name="past_values_2"),
            ct.TensorType(shape=(batch_size, num_heads, ct.RangeDim(lower_bound=1, upper_bound=max_length, default=max_length), embed_size_per_head), dtype=np.float32, name="past_values_3"),
        ],
        # lm_logits, hidden_states, present_keys, present_values
        outputs=[
            ct.TensorType(name="lm_logits"),
            ct.TensorType(name="hidden_states"),
            ct.TensorType(name="present_keys_1"),
            ct.TensorType(name="present_keys_2"),
            ct.TensorType(name="present_keys_3"),
            ct.TensorType(name="present_values_1"),
            ct.TensorType(name="present_values_2"),
            ct.TensorType(name="present_values_3"),
        ],
        minimum_deployment_target=ct.target.iOS16,
    )
    converted_model.save("decoder.mlpackage")
    print("# converted.")
    # print("# compressing decoder ...")
    # compressed_model = ct.compression_utils.affine_quantize_weights(converted_model)
    # compressed_model.save("decoder_compressed.mlpackage")
    # print("# compressed.")


convert_encoder()
convert_decoder_first()
convert_decoder()
