# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
import random
import time 
from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from iTrainingLogger import iSummaryWriter

tqdm.pandas()

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a T5 model on the IMDB dataset using PPO
# (proximal policy optimization).
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config` then run the script with
# `accelerate launch ppo-sentiment-t5-small.py`
#
########################################################################

# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
writer = iSummaryWriter(log_path='./logs', log_name='PPO-Sentiment-t5')
config = PPOConfig(model_name="lvwerra/t5-imdb", learning_rate=5e-5, batch_size=16)
# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

# prompt池
prompts = [
    'This movie ',
    'That movie '
]
# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)


generation_kwargs = {"top_k": 0.0, "top_p": 1.0, "do_sample": True, "eos_token_id": -1}


# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)


device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
sentiment_pipe = pipeline("sentiment-analysis", "lvwerra/distilbert-imdb", device=device)


output_min_length = 16
output_max_length = 32
output_length_sampler = LengthSampler(output_min_length, output_max_length)



for epoch in tqdm(range(50)):
    logs, timing = dict(), dict()
    t0 = time.time()
    batch = {
        'tokens': [],
        'query': []
    }
    for _ in range(config.batch_size):
        random_prompt = random.choice(prompts)                                  # 随机选择一个prompt
        tokens = tokenizer.encode(random_prompt) + [tokenizer.eos_token_id]
        batch['tokens'].append(tokens)
        batch['query'].append(random_prompt)
    query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]
    
    t = time.time()
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze())
    batch["response"] = [tokenizer.decode(r[1:].squeeze()) for r in response_tensors]
    
    t = time.time()
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[1]["score"]).to(device) for output in pipe_outputs]
    print(rewards)
    timing['time/get_sentiment_preds'] = time.time() - t
    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    # ppo_trainer.log_stats(stats, batch, rewards)
    timing['time/optimization'] = time.time() - t

    timing['time/epoch'] = time.time() - t0                                     # logging
    logs.update(timing)
    logs.update(stats)
    
    logs['env/reward_mean'] = sum(rewards) / len(rewards)
    # logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    # logs['env/reward_dist'] = rewards.cpu().numpy()
    print(f"epoch {epoch} mean-reward: {logs['env/reward_mean']}")
    
    print('Random Sample 5 text(s) of model output:')
    for i in range(5):                                                           # 随机打5个生成的结果
        print(f'{i+1}. {random.choice(texts)}')
    
    writer.add_scalar('train/reward', logs['env/reward_mean'], epoch)
    for k, v in timing.items():
        writer.add_scalar(k, v, epoch)
    writer.add_scalar("objective/entropy", stats["objective/entropy"], epoch)
    writer.add_scalar("objective/kl", stats["objective/kl"], epoch)
    # writer.add_scalar("objective/logprobs", stats["objective/logprobs"], epoch)
    # writer.add_scalar("ppo/mean_non_score_reward", stats["ppo/mean_non_score_reward"], epoch)
    writer.record()
# for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
#     query_tensors = batch["input_ids"]

#     # Get response from gpt2
#     response_tensors = []
#     for query in query_tensors:
#         gen_len = output_length_sampler()
#         generation_kwargs["max_new_tokens"] = gen_len
#         response = ppo_trainer.generate(query, **generation_kwargs)
#         response_tensors.append(response.squeeze())
#     batch["response"] = [tokenizer.decode(r[1:].squeeze()) for r in response_tensors]

#     # Compute sentiment score
#     texts = [q + r for q, r in zip(batch["query"], batch["response"])]
#     pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
#     rewards = [torch.tensor(output[1]["score"]).to(device) for output in pipe_outputs]

#     # Run PPO step
#     stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
#     ppo_trainer.log_stats(stats, batch, rewards)