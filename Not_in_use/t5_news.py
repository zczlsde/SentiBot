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
writer = iSummaryWriter(log_path='./logs', log_name='PPO-Sentiment-Zh')
# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
config = PPOConfig(model_name="mrm8488/t5-base-finetuned-summarize-news", learning_rate=5e-5, batch_size=256)
# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

def build_imdb_dataset(tokenizer, input_min_text_length=2, input_max_text_length=500):
    # load imdb with datasets
    ds = load_dataset("argilla/news-summary", split="train")
    ds = ds.rename_columns({"text": "news"})
    # ds = ds.filter(lambda x: len(x["news"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["news"][:input_size()]) + [tokenizer.eos_token_id]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collater(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# senti_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# sentiment_pipe = pipeline('sentiment-analysis', model=senti_model, tokenizer=senti_tokenizer, device=pipe_device)
# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_imdb_dataset(tokenizer)

query = tokenizer("I really liked this movie because", return_tensors="pt")["input_ids"]

generation_kwargs = {"top_k": 0.0, "top_p": 1.0, "do_sample": True, "eos_token_id": -1}


# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collater)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
sentiment_pipe = pipeline("sentiment-analysis", "lvwerra/distilbert-imdb", device=device)

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
output_min_length = 16
output_max_length = 100
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    logs, timing = dict(), dict()
    query_tensors = batch["input_ids"]

    # Get response from gpt2
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze())
    batch["response"] = [tokenizer.decode(r[1:].squeeze()) for r in response_tensors]

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    for i in range(5):                                                           # 随机打5个生成的结果
        print(f'{i+1}. {random.choice(texts)}')

    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[1]["score"]).to(device) for output in pipe_outputs]

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
    
    logs.update(timing)
    logs.update(stats)
    logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
    logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    logs['env/reward_dist'] = rewards.cpu().numpy()
    print(f"epoch {epoch} mean-reward: {logs['env/reward_mean']}")
    
    writer.add_scalar('train/reward', logs['env/reward_mean'], epoch)
    for k, v in timing.items():
        writer.add_scalar(k, v, epoch)
    writer.add_scalar('ppo/loss/policy', stats['ppo/loss/policy'], epoch)
    writer.add_scalar('ppo/loss/value', stats['ppo/loss/value'], epoch)
    writer.add_scalar('ppo/policy/entropy', stats['ppo/policy/entropy'], epoch)
    writer.add_scalar('ppo/policy/policykl', stats['ppo/policy/policykl'], epoch)
    writer.record()
    
    