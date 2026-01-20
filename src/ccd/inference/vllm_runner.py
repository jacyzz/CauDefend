from __future__ import annotations

import gc
import os
import threading
from typing import List, Optional, Tuple, Union

import torch

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

from ccd.inference.engine import (
    GenerationConfig,
    ModelConfig,
    PromptConfig,
    build_prompt,
    extract_response,
    load_system_prompt,
)


class VLLMRunner:
    _instance: Optional[VLLMRunner] = None
    _lock = threading.Lock()

    def __init__(self):
        self.llm: Optional[LLM] = None
        self.model_name: Optional[str] = None

    @classmethod
    def get_instance(cls) -> VLLMRunner:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = VLLMRunner()
        return cls._instance

    def load_model(self, cfg: ModelConfig):
        if LLM is None:
            raise ImportError("vllm is not installed. Please install it with: pip install vllm")

        # If already loaded same model, skip
        if self.llm is not None and self.model_name == cfg.model:
            return

        # Unload previous if exists (though vllm doesn't support easy unloading...)
        # We can try to clear it but usually vllm requires process restart or heavy GC.
        # For now, we'll try basic cleanup.
        if self.llm is not None:
             # This is tricky with vLLM, it allocates Ray actors or heavy GPU buffers.
             # We will try to just set it to None and GC, but it might not be enough.
            del self.llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.model_name = cfg.model
        
        # Determine dtype
        dtype = "auto"
        if cfg.dtype == "float16":
            dtype = "float16"
        elif cfg.dtype == "bfloat16":
            dtype = "bfloat16"

        # Load LLM
        # Note: tensor_parallel_size=1 by default for this simple integration
        self.llm = LLM(
            model=cfg.model,
            dtype=dtype,
            trust_remote_code=cfg.trust_remote_code,
            # If we need to support LoRA (peft_adapter), vLLM has enable_lora=True.
            # For now, let's assume base model or merged model.
            # If peft_adapter is present, we might warn or try enable_lora.
        )

    def generate(
        self,
        input_texts: Union[str, List[str]],
        prompt_cfg: PromptConfig,
        gen_cfg: GenerationConfig,
    ) -> Tuple[Union[List[str], List[List[str]]], Optional[List[float]], Union[List[str], List[List[str]]]]:
        """
        Returns:
            - responses: List of extracted responses.
              If input_texts is str: List[str] (length n).
              If input_texts is List[str]: List[List[str]] (length len(input_texts), each inner list length n).
            - scores: None
            - decoded: same structure as responses but raw text.
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        is_single = isinstance(input_texts, str)
        if is_single:
            input_texts = [input_texts]

        system_prompt = load_system_prompt(prompt_cfg)
        prompts = [build_prompt(t, system_prompt) for t in input_texts]

        # Map GenerationConfig to SamplingParams
        # vLLM SamplingParams:
        # n: int
        # best_of: int
        # presence_penalty: float
        # frequency_penalty: float
        # temperature: float
        # top_p: float
        # top_k: int
        # use_beam_search: bool
        # length_penalty: float
        # early_stopping: Union[bool, str]
        # stop: Union[None, str, List[str]]
        # ignore_eos: bool
        # max_tokens: int
        # logprobs: int
        # prompt_logprobs: int
        
        # Heuristics for beam search
        use_beam_search = False
        if gen_cfg.num_beams > 1:
            use_beam_search = True
        
        best_of = gen_cfg.num_beams if use_beam_search else gen_cfg.num_return_sequences

        sampling_params = SamplingParams(
            n=gen_cfg.num_return_sequences,
            best_of=best_of,
            temperature=gen_cfg.temperature if not use_beam_search else 0.0,
            top_p=gen_cfg.top_p if not use_beam_search else 1.0,
            # use_beam_search was removed in vLLM 0.2.2+; inferred from best_of > 1 and temperature ~ 0
            # use_beam_search=use_beam_search,
            max_tokens=gen_cfg.max_new_tokens,
            # seed=gen_cfg.seed, # vLLM might not support seed per request easily in older versions, check version
        )

        outputs = self.llm.generate(prompts, sampling_params)

        grouped_responses = []
        grouped_decoded = []
        
        for output in outputs:
            curr_resps = []
            curr_decs = []
            for comp in output.outputs:
                 text = comp.text
                 curr_decs.append(text)
                 curr_resps.append(extract_response(text))
            grouped_responses.append(curr_resps)
            grouped_decoded.append(curr_decs)
            
        if is_single:
            return grouped_responses[0], None, grouped_decoded[0]
        
        return grouped_responses, None, grouped_decoded

    def unload(self):
        # Best effort unload
        if self.llm:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            del self.llm
            self.llm = None
            self.model_name = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

