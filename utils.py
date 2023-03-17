from datasets import load_dataset


def load_perc_prompts(num):
    dataset = load_dataset('csv', data_files={'train': './Baseline/Data/PERC.csv'})
    joy_dataset = dataset.filter(lambda row: row['Emotion'] == 'surprise' 
                                 or row['Emotion'] == 'joy' 
                                 or row['Emotion'] == 'love' 
                                 or row['Emotion'] == 'peace' 
                                 or row['Emotion'] == 'courage')
        
    prompts = joy_dataset.map(lambda poem: 
        {'prompt': ' '.join(poem['Poem'].split()[:num])}
    )['prompt']
    return prompts

def load_prompts(num):
    dataset = load_dataset(
        "Ozziey/poems_dataset", 
        data_files="final_df_emotions(remove-bias).csv"
    )['train']
    joy_dataset = dataset.filter(lambda row: row['label'] == 'joy')
    
    prompts = joy_dataset.map(lambda poem: 
        {'prompt': ' '.join(poem['poem content'].split()[:num])}
    )['prompt']

    return prompts