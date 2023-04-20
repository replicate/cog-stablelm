import os
import subprocess
import time
from collections import OrderedDict
from typing import Any, List, Optional

import torch
from cog import BasePredictor, Input, Path
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
from transformers import (
    AutoConfig,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
    GPTNeoXForCausalLM,
)


DEFAULT_MODEL = "stabilityai/stablelm-tuned-alpha-7b"
CACHE_DIR = "pretrained_weights"
TOKENIZER_PATH = "./tokenizer"

TENSORIZER_WEIGHTS_PATH = "gs://replicate-weights/stablelm-tuned-alpha-7b.tensors"

SYSTEM_PROMPT = """# StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""


class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def maybe_download(path):
    if path.startswith("gs://"):
        output_path = "/tmp/weights.tensors"
        subprocess.check_call(["gcloud", "storage", "cp", path, output_path])
        return output_path
    return path


class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        if weights is not None and weights.name == "weights":
            weights = None
        if weights is None and TENSORIZER_WEIGHTS_PATH:
            print("Loading tensorized weights from public path")
            self.model = self.load_tensorizer(
                weights=maybe_download(TENSORIZER_WEIGHTS_PATH)
            )

        elif (hasattr(weights, "filename") and "tensors" in weights.filename) or str(
            weights
        ).endswith(".tensors"):
            self.model = self.load_tensorizer(weights)
        else:
            self.model = self.load_huggingface_model(weights=weights)

        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        self.stop = StopOnTokens()
        self.generator = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer, device=0
        )

    def load_huggingface_model(self, weights=None):
        st = time.time()
        print(f"loading weights from {weights} w/o tensorizer")
        model = GPTNeoXForCausalLM.from_pretrained(weights, cache_dir=CACHE_DIR).to(
            "cuda:0"
        )
        print(f"weights loaded in {time.time() - st}")
        return model

    def load_tensorizer(self, weights):
        st = time.time()
        print(f"deserializing weights from {weights}")
        config = AutoConfig.from_pretrained(DEFAULT_MODEL)

        model = no_init_or_tensor(
            lambda: GPTNeoXForCausalLM.from_pretrained(
                None, config=config, state_dict=OrderedDict()
            )
        )
        des = TensorDeserializer(weights, plaid_mode=True)
        des.load_into_module(model)
        print(f"weights loaded in {time.time() - st}")
        return model

    def predict(
        self,
        prompt: str = Input(
            description=f"Input Prompt.", default="What's your mood today?"
        ),
        instructions: str = Input(
            description=f"Instructions Prompt.", default=SYSTEM_PROMPT
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=100,
        ),
        top_p: float = Input(
            description="Valid if you choose top_p decoding. When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1.2,
        ),
    ) -> str:

        prompt_text = f"<|SYSTEM|>{instructions}<|USER|>{prompt}<|ASSISTANT|>"

        result = self.generator(
            prompt_text,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            num_beams=1,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stopping_criteria=StoppingCriteriaList([self.stop]),
        )
        return result[0]["generated_text"].replace(prompt_text, "")
