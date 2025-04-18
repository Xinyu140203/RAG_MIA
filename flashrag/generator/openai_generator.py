import os
from typing import List
from copy import deepcopy
import warnings
from tqdm import tqdm
import numpy as np
import asyncio
import asyncio
from openai import AsyncOpenAI, AsyncAzureOpenAI
import tiktoken
import openai
import torch

class OpenaiGenerator:
    """Class for api-based openai models"""

    def __init__(self, config):
        self.model_name = config["generator_model"]
        self.batch_size = config["generator_batch_size"]
        self.generation_params = config["generation_params"]

        self.openai_setting = config["openai_setting"]
        if self.openai_setting["api_key"] is None:
            self.openai_setting["api_key"] = os.getenv("OPENAI_API_KEY")
        if "api_type" in self.openai_setting and self.openai_setting["api_type"] == "azure":
            del self.openai_setting["api_type"]
            self.client = AsyncAzureOpenAI(**self.openai_setting)
        else:
            self.client = AsyncOpenAI(**self.openai_setting)
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)

    async def get_response(self, input: List, **params):
        try:

            response = await asyncio.wait_for(
                self.client.chat.completions.create(model=self.model_name, messages=input, **params),
                timeout=300
            )
            return response.choices[0]
        except asyncio.TimeoutError:
            print("Request timed out")
            raise
        except Exception as e:
            print(f"Error while connecting to GPT: {e}")
            raise


    async def get_batch_response(self, input_list: List[List], batch_size, **params):
        total_input = [self.get_response(input, **params) for input in input_list]
        all_result = []
        for idx in tqdm(range(0, len(input_list), batch_size), desc="Generation process: "):
            batch_input = total_input[idx : idx + batch_size]
            batch_result = await asyncio.gather(*batch_input)
            all_result.extend(batch_result)

        return all_result



    def generate(self, input_list: List[List], batch_size=None, return_scores=False,return_sure=False, **params) -> List[str]:
        # deal with single input
        if len(input_list) == 1 and isinstance(input_list[0], dict):
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        # deal with generation params
        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        if "do_sample" in generation_params:
            generation_params.pop("do_sample")

        max_tokens = params.pop("max_tokens", None) or params.pop("max_new_tokens", None)
        if max_tokens is not None:
            generation_params["max_tokens"] = max_tokens
        else:
            generation_params["max_tokens"] = generation_params.get(
                "max_tokens", generation_params.pop("max_new_tokens", None)
            )
        generation_params.pop("max_new_tokens", None)

        if return_scores or return_sure:
            if generation_params.get("logprobs") is not None:
                generation_params["logprobs"] = True
                warnings.warn("Set logprobs to True to get generation scores.")
            else:
                generation_params["logprobs"] = True

        if generation_params.get("n") is not None:
            generation_params["n"] = 1
            warnings.warn("Set n to 1. It can minimize costs.")
        else:
            generation_params["n"] = 1
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.get_batch_response(input_list, batch_size, **generation_params))

        # parse result into response text and logprob
        scores = []
        response_text = []
        a_text = []
        b_text = []
        buffer = []
        in_a = False
        in_b = False
        a_yes_prob = 0
        a_no_prob =0
        b_yes_prob = 0
        b_no_prob =0

        a_yes_found = False
        a_no_found = False
        b_yes_found = False
        b_no_found = False
        yes_probs_tensor=[]
        no_probs_tensor=[]
        perplexity=[]
        for res in result:
            response_text.append(res.message.content)
            if return_sure :
                token_score_mapping = list(map(lambda x: (x.token, np.exp(x.logprob)), res.logprobs.content))
                score = np.exp(list(map(lambda x: x.logprob, res.logprobs.content)))
                scores.append(score)
                for token, prob in token_score_mapping:
                    print(f"Token: {token}, Probability: {prob}")
                    buffer.append(token.strip())
                    buffer_str = ''.join(buffer)
                    if "(a)" in buffer_str or "(a" in buffer_str:
                        in_a = True
                        in_b = False
                        buffer.clear()
                        continue
                    if "(b)" in buffer_str or "(b" in buffer_str:
                        in_a = False
                        in_b = True
                        buffer.clear()
                        continue

                    if in_a:
                        a_text.append(token)
                        if token.strip() == 'Yes' and not a_yes_found:
                            a_yes_prob = prob
                            a_yes_found = True
                        elif token.strip() == 'No' and not a_no_found:
                            a_no_prob = prob
                            a_no_found = True

                    elif in_b:
                        b_text.append(token)
                        if token.strip() == 'Yes' and not b_yes_found:
                            b_yes_prob = prob
                            b_yes_found = True
                        elif token.strip() == 'No' and not b_no_found:
                            b_no_prob = prob
                            b_no_found = True
                a_text_str = ''.join(a_text).strip()
                b_text_str = ''.join(b_text).strip()
            yes_prob=0
            no_prob=0
            if return_scores :
                if res.logprobs is not None and res.logprobs.content is not None:

                    token_score_mapping = list(map(lambda x: (x.token, np.exp(x.logprob)), res.logprobs.content))
                    score = np.exp(list(map(lambda x: x.logprob, res.logprobs.content)))
                else:
                    tokens = [x.token for x in res.choices] if hasattr(res, 'choices') else []
                    token_score_mapping = list(map(lambda x: (x, 0), tokens))
                    score = 0

                log_probs = torch.log(torch.tensor(score))
                avg_log_prob = log_probs.mean(dim=0)
                perplexity_tensor = torch.exp(-avg_log_prob)
                perplexity.append(perplexity_tensor.item())
                found = False
                for token, prob in token_score_mapping:
                    print(f"Token: {token}, Probability: {prob}")
                    if token.strip() == 'Yes':
                        yes_prob=prob
                        no_prob =1-prob
                        yes_probs_tensor .append(yes_prob )
                        no_probs_tensor .append(no_prob)
                        found = True
                        break
                    if token.strip() =='No':
                        no_prob =prob
                        yes_prob =1-prob
                        yes_probs_tensor .append(yes_prob )
                        no_probs_tensor .append(no_prob)
                        found = True
                        break
                if not found:
                    yes_probs_tensor.append(yes_prob)
                    no_probs_tensor.append(no_prob)
                    print("Neither 'Yes' nor 'No' was found in the generated tokens. Appending default probabilities (0, 0).")
        print(perplexity)
        if return_scores:
            return response_text, yes_probs_tensor ,no_probs_tensor
        elif return_sure :
            return response_text,a_yes_prob,a_no_prob,b_yes_prob,b_no_prob,scores
        else:
            return response_text
