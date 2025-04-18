import re
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from flashrag.utils import get_retriever, get_generator, selfask_pred_parse, ircot_pred_parse
from flashrag.pipeline import BasicPipeline
from flashrag.dataset import get_batch_dataset, merge_batch_dataset
from flashrag.prompt import PromptTemplate
import torch
import numpy as np

class IRCOTPipeline(BasicPipeline):
    IRCOT_INSTRUCTION = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'
    IRCOT_EXAMPLE = "Wikipedia Title: Kurram Garhi\nKurram Garhi is a small village located near the city of Bannu, which is the part of Khyber Pakhtunkhwa province of Pakistan. Its population is approximately 35000. Barren hills are near this village. This village is on the border of Kurram Agency. Other nearby villages are Peppal, Surwangi and Amandi Kala.\n\nWikipedia Title: 2001–02 UEFA Champions League second group stage\nEight winners and eight runners- up from the first group stage were drawn into four groups of four teams, each containing two group winners and two runners- up. Teams from the same country or from the same first round group could not be drawn together. The top two teams in each group advanced to the quarter- finals.\n\nWikipedia Title: Satellite tournament\nA satellite tournament is either a minor tournament or event on a competitive sporting tour or one of a group of such tournaments that form a series played in the same country or region.\n\nWikipedia Title: Trojkrsti\nTrojkrsti is a village in Municipality of Prilep, Republic of Macedonia.\n\nWikipedia Title: Telephone numbers in Ascension Island\nCountry Code:+ 247< br> International Call Prefix: 00 Ascension Island does not share the same country code( +290) with the rest of St Helena.\n\nQuestion: Are both Kurram Garhi and Trojkrsti located in the same country?\nThought: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is: no.\n\n"

    def __init__(self, config, prompt_template=None, retriever=None, generator=None, max_iter=2):
        # if not provide prompt template, use default template provided by IRCOT
        if prompt_template is None:
            prompt_template = PromptTemplate(
                config=config,
                system_prompt=f"{self.IRCOT_INSTRUCTION}\n\n{self.IRCOT_EXAMPLE}",
                user_prompt="{reference}Question: {question}\nThought:",
                reference_template="Wikipedia Title: {title}\n{text}\n\n",
                enable_chat=True,
            )

        super().__init__(config, prompt_template)
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever

        self.max_iter = max_iter

    def run_item(self, item):

        question = item.question
        retrieval_result, scores = self.retriever.search(question, return_score=True)
        doc2score = {doc_item["id"]: score for doc_item, score in zip(retrieval_result, scores)}
        id2doc = {doc_item["id"]: doc_item for doc_item in retrieval_result}
        thoughts = []
        iter_num = 0
        found_first_yes_prob = False
        found_first_no_prob = False
        max_yes_probs_tensor=0
        max_no_probs_tensor=0
        while iter_num < self.max_iter:
            input_prompt = self.prompt_template.get_string(
                question=question, retrieval_result=retrieval_result, previous_gen=" ".join(thoughts)
            )
            new_thought,yes_probs_tensor,no_probs_tensor,perplexity = self.generator.generate([input_prompt],return_perplexity=True)
            new_thought=new_thought [0]
            if not isinstance(yes_probs_tensor, torch.Tensor) or yes_probs_tensor.dim() != 2 or yes_probs_tensor.shape != (1, 1):
                yes_probs_tensor = torch.tensor([[float(yes_probs_tensor[0])]])
            if not isinstance(no_probs_tensor, torch.Tensor) or no_probs_tensor.dim() != 2 or no_probs_tensor.shape != (1, 1):
                no_probs_tensor = torch.tensor([[float(no_probs_tensor[0])]])
            yes_probs_tensor=yes_probs_tensor [0].item()
            no_probs_tensor=no_probs_tensor [0].item()
            perplexity=perplexity[0]
            if not found_first_yes_prob and yes_probs_tensor != 0:
                max_yes_probs_tensor = yes_probs_tensor
                found_first_yes_prob = True
            if not found_first_no_prob and no_probs_tensor != 0:
                max_no_probs_tensor = no_probs_tensor
                found_first_no_prob = True
            thoughts.append(new_thought)
            iter_num += 1
            if "So the answer is:" in new_thought:
                item.update_output(
                    f"intermediate_output_iter{iter_num}",
                    {
                        "input_prompt": input_prompt,
                        "new_thought": new_thought,
                    },
                )
                break

            # retrieve new docs and merge
            if new_thought :
                new_retrieval_result, new_scores = self.retriever.search(new_thought, return_score=True)
                for doc_item, score in zip(new_retrieval_result, new_scores):
                    id2doc[doc_item["id"]] = doc_item
                    doc_id = doc_item["id"]
                    if doc_id in doc2score:
                        doc2score[doc_id] = max(doc2score[doc_id], score)
                    else:
                        doc2score[doc_id] = score
                sorted_doc_score = sorted(doc2score.items(), key=lambda x: x[1], reverse=False)
                sorted_doc_id = [t[0] for t in sorted_doc_score]
                retrieval_result = [id2doc[id] for id in sorted_doc_id]

            item.update_output(
                f"intermediate_output_iter{iter_num}",
                {
                    "input_prompt": input_prompt,
                    "new_thought": new_thought,
                    "new_retreival_result": new_retrieval_result,
                },
            )

        item.update_output("retrieval_result", retrieval_result)
        item.update_output("yes_probs_tensor",max_yes_probs_tensor)
        item.update_output("no_probs_tensor",max_no_probs_tensor)
        item.update_output("pred", " ".join(thoughts))
        item.update_output("perplexity",perplexity)
        return item

    def run(self, dataset, do_eval=True, pred_process_fun=ircot_pred_parse):
        pred_list=[]
        yes_list=[]
        no_list=[]
        perplexity=[]
        for item in tqdm(dataset, desc="Inference: "):
            self.run_item(item)
            pred_list .append(item.output ["pred"])
            yes_list.append(item.output ["yes_probs_tensor"])
            no_list .append(item.output ["no_probs_tensor"])
            perplexity.append(item.output ["perplexity"])

        torch.set_printoptions(precision=4, sci_mode=False, threshold=2000)
        yes_list_tensor = torch.tensor(yes_list)
        no_list_tensor = torch.tensor(no_list)
        perplexity_tensor = torch.tensor(perplexity)
        print(pred_list)
        print(yes_list_tensor)
        print(no_list_tensor )
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset    
