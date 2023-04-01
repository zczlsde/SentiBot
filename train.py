import random 
import time
from datetime import datetime

import torch
from tqdm import tqdm
tqdm.pandas()
from evaluate import load
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from iTrainingLogger import iSummaryWriter
from utils import load_prompts, load_perc_prompts


def train(
    model_type, # {"gpt2", "gpt2_bert"}
    perc, # whether to use perc dataset
    prompt_in_loop, # true / false, whether to load the prompt each loop
    output_length,
    bert_beta = 0 , # only valid when 'gpt2_bert'
):
    reference_score = False
    # Logging 
    if model_type == "gpt2":
        EXPERIMENT_NAME = "POEM_GPT2"
    elif model_type == "gpt2_bert":
        EXPERIMENT_NAME = "POEM_GPT2_BERT"
        reference_score = True
    else:
        raise Exception("Unknown Model Type")
    
    DATASET_NAME = 'PERC' if perc else "ND"
    PROMPT_NAME = 'IN' if prompt_in_loop else 'OUT'
    Beta = bert_beta if reference_score else '0'
    LOG_PATH = f'./logs/{EXPERIMENT_NAME}_output{output_length}_{DATASET_NAME}_{PROMPT_NAME}_{Beta}'
    writer = iSummaryWriter(log_path=LOG_PATH, log_name='SentiBot')
    print("LOG PATH: " + LOG_PATH)
    
    # Model Config
    config = PPOConfig(
        model_name="ismaelfaro/gpt2-poems.en",
        learning_rate=1.41e-5,
        batch_size=64,
        forward_batch_size=16, # could be increased
    )
    if perc:
        dataset = load_dataset('csv', data_files={'train': './Baseline/Data/PERC.csv'})
        load_p = load_perc_prompts
    else:
        dataset = load_dataset(
            "Ozziey/poems_dataset", 
            data_files="final_df_emotions(remove-bias).csv"
        )['train']
        load_p = load_prompts

    if not prompt_in_loop:
        prompts = load_p(5, dataset)[:5]

    set_seed(config.seed)

    # Initialise Model
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    if reference_score:
        bertscore = load("bertscore")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialise Trainer
    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)
    if reference_score:
        ppo_trainer_ref = PPOTrainer(config, ref_model, ref_model, tokenizer)
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    sentiment_pipe = pipeline(
        "sentiment-analysis", 
        model="nickwong64/bert-base-uncased-poems-sentiment", 
        device=device
    )

    # Output Config
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    output_min_length = 4
    output_max_length = output_length # Increase the max length
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    # Training
    for epoch in tqdm(range(500)):
        logs, timing = dict(), dict()
        start_time = time.time()
        if prompt_in_loop:
            prompts = load_p(5, dataset)[:5]

        batch = {
            'tokens': [],
            'query': []
        }
        for _ in range(config.batch_size):
            random_prompt = random.choice(prompts)                                  # 随机选择一个prompt
            tokens = tokenizer.encode(random_prompt)
            batch['tokens'].append(tokens)
            batch['query'].append(random_prompt)
        query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]
        
        # Generate response
        response_tensors = []
        if reference_score:
            ref_response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
            if reference_score:
                ref_response = ppo_trainer_ref.generate(query, **generation_kwargs)
                ref_response_tensors.append(ref_response.squeeze()[-gen_len:])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        if reference_score:
            batch["ref_response"] = [tokenizer.decode(r.squeeze()) for r in ref_response_tensors]
        
        # Sentiment
        t = time.time()
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts)

        rewards = []
        # Compute Bert Reference Score
        if reference_score:
            predictions = batch["response"]
            references = batch["ref_response"]
            results = bertscore.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased")
            results_bert = results['precision']

            for idx, output in enumerate(pipe_outputs):
                if output['label'] == 'positive':
                    rewards.append(torch.tensor(
                        (1-bert_beta)*output['score'] + 
                        bert_beta * results_bert[idx]
                    ).to(device))
                elif output['label'] == 'no_impact' or output['label'] == 'mixed':
                    rewards.append(torch.tensor(
                        bert_beta * results_bert[idx]
                    ).to(device))
                elif output['label'] == 'negative':
                    rewards.append(torch.tensor(
                        (1-bert_beta)* (-output['score']) + 
                        bert_beta * results_bert[idx]
                    ).to(device))
                else:
                    raise ValueError(f"Wrong {output['label']}.")
        else:          
            for output in pipe_outputs:
                if output['label'] == 'positive':
                    rewards.append(torch.tensor(output['score']).to(device))
                elif output['label'] == 'no_impact' or output['label'] == 'mixed':
                    rewards.append(torch.tensor(0.0).to(device))
                elif output['label'] == 'negative':
                    rewards.append(torch.tensor(-output['score']).to(device))
                else:
                    raise ValueError(f"Wrong {output['label']}.")

        timing['time/get_sentiment_preds'] = time.time() - t

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        timing['time/optimization'] = time.time() - t

        timing['time/epoch'] = time.time() - start_time # logging
        logs.update(timing)
        logs.update(stats)
        logs['env/reward_mean'] = (sum(rewards) / len(rewards)).cpu()
        print(f"epoch {epoch} mean-reward: {logs['env/reward_mean']}")
        
        if epoch != 0 and epoch % 100 == 0:
            save_path = f'{LOG_PATH}/EPOCH{epoch}'
            model.save_pretrained(save_path, push_to_hub=False)
            tokenizer.save_pretrained(save_path, push_to_hub=False)

        writer.add_scalar('train/reward', logs['env/reward_mean'], epoch)
        for k, v in timing.items():
            writer.add_scalar(k, v, epoch)
        writer.add_scalar("objective/entropy", stats["objective/entropy"], epoch)
        writer.add_scalar("objective/kl", stats["objective/kl"], epoch)
        writer.record()


if __name__ == "__main__":

    for model in {'gpt2',}: #{'gpt2_bert', 'gpt2'}:
        for perc in {True, False}:
            for prompt_in in {True, False}:
                for output_length in {128, 32}:
                    if model == 'gpt2_bert':
                        for bert_beta in {0.1, 0.3, 0.5, 0.7, 0.9}:
                            train(
                                model,
                                perc,
                                prompt_in,
                                output_length, 
                                bert_beta
                            )
                    else:
                        train(
                            model,
                            perc,
                            prompt_in,
                            output_length
                        )
