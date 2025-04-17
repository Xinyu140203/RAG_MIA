from flashrag.config import Config
from flashrag.utils import get_dataset
import argparse


def naive(args):
    save_note = "naive"
    config_dict = {"save_note": save_note, "gpu_id": args.gpu_id, "dataset_name": args.dataset_name}

    from flashrag.pipeline import SequentialPipeline

    # preparation
    config = Config("my_config.yaml", config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    pred_process_fun = lambda x: x.split("\n")[0]
    pipeline = SequentialPipeline(config)

    result = pipeline.run(test_data)


def llmlingua(args):
    """
    Reference:
        Huiqiang Jiang et al. "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models"
        in EMNLP 2023
        Huiqiang Jiang et al. "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression"
        in ICLR MEFoMo 2024.
        Official repo: https://github.com/microsoft/LLMLingua
    """
    refiner_name = "longllmlingua"  #
    refiner_model_path = "Llama-2-7b-hf"

    config_dict = {
        "refiner_name": refiner_name,
        "refiner_model_path": refiner_model_path,
        "llmlingua_config": {
            "rate": 0.55,
            "condition_in_question": "after_condition",
            "reorder_context": "sort",
            "dynamic_context_compression_ratio": 0.3,
            "condition_compare": True,
            "context_budget": "+100",
            "rank_method": "longllmlingua",
        },
        "refiner_input_prompt_flag": False,
        "save_note": "longllmlingua",
        "gpu_id": args.gpu_id,
        "dataset_name": args.dataset_name,
    }

    # preparation
    config = Config("my_config.yaml", config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline

    pipeline = SequentialPipeline(config)
    result = pipeline.run(test_data)


def sc(args):
    """
    Reference:
        Yucheng Li et al. "Compressing Context to Enhance Inference Efficiency of Large Language Models"
        in EMNLP 2023.
        Official repo: https://github.com/liyucheng09/Selective_Context

    Note:
        Need to install spacy:
            ```python -m spacy download en_core_web_sm```
        or
            ```
            wget https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0.tar.gz
            pip install en_core_web_sm-3.6.0.tar.gz
            ```
    """
    refiner_name = "selective-context"
    refiner_model_path = "gpt2"

    config_dict = {
        "refiner_name": refiner_name,
        "refiner_model_path": refiner_model_path,
        "sc_config": {"reduce_ratio": 0.2},
        "save_note": "selective-context",
        "gpu_id": args.gpu_id,
        "dataset_name": args.dataset_name,
    }

    # preparation
    config = Config("my_config.yaml", config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline

    pipeline = SequentialPipeline(config)
    result = pipeline.run(test_data)


def ircot(args):
    """
    Reference:
        Harsh Trivedi et al. "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions"
        in ACL 2023
    """
    save_note = "ircot"
    config_dict = {"save_note": save_note, "gpu_id": args.gpu_id, "dataset_name": args.dataset_name}

    from flashrag.pipeline import IRCOTPipeline

    # preparation
    config = Config("my_config.yaml", config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    print(config["generator_model_path"])
    pipeline = IRCOTPipeline(config)

    result = pipeline.run(test_data)


def spring(args):
    """
    Reference:
        Yutao Zhu et al. "One Token Can Help! Learning Scalable and Pluggable Virtual Tokens for Retrieval-Augmented Large Language Models"
    """

    save_note = "spring"
    config_dict = {
        "save_note": save_note,
        "gpu_id": args.gpu_id,
        "dataset_name": args.dataset_name,
        "framework": "hf",
    }
    config = Config("my_config.yaml", config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    # download token embedding from: https://huggingface.co/yutaozhu94/SPRING
    token_embedding_path = "llama2.7b.chat.added_token_embeddings.pt"

    from flashrag.prompt import PromptTemplate
    from flashrag.pipeline import SequentialPipeline
    from flashrag.utils import get_generator, get_retriever

    # prepare prompt and generator for Spring method
    system_prompt = (
        "context:{reference}\n"
        "Answer the question based on the provided context."

    )
    added_tokens = [f" [ref{i}]" for i in range(1, 51)]
    added_tokens = "".join(added_tokens)
    user_prompt = added_tokens + "Question: {question}\nAnswer:"
    prompt_template = PromptTemplate(config, system_prompt, user_prompt, enable_chat=False)

    generator = get_generator(config)
    generator.add_new_tokens(token_embedding_path, token_name_func=lambda idx: f"[ref{idx+1}]")

    pipeline = SequentialPipeline(
        config=config, prompt_template=prompt_template, generator=generator
    )
    result = pipeline.run(test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running exp")
    parser.add_argument("--method_name", type=str,default= "sure")
    parser.add_argument("--split", type=str,default= "test")
    parser.add_argument("--dataset_name", type=str,default="nq")
    #parser.add_argument("--gpu_id", type=str)
    parser.add_argument("--gpu_id", type=str, default="-1")

    func_dict = {
        "Standard-RAG": naive,
        "llmlingua": llmlingua,
        "SC-RAG": sc,
        "ircot": ircot,
        'spring' :spring
    }

    args = parser.parse_args()

    func = func_dict[args.method_name]
    func(args)
