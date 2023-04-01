import os
import Baseline.testing as t
import matplotlib.pyplot as plt

def evaluate(generated_path, save_path):
    original_sentence = "To My Fairy Fancies NAY, no longer I may hold you, In my spirit's soft caresses, Nor like lotus-leaves enfold you In the tangles of my tresses. Fairy fancies, fly away To the white cloud-wildernesses, Fly away! Nay, no longer ye may linger With your laughter-lighted faces, Now I am a thought-worn singer In life's high and lonely places. Fairy fancies, fly away, To bright wind-inwoven spaces, Fly"

    _, fluency_data = t.fluency(generated_path)
    plt.figure()
    plt.hist(fluency_data, bins=50)
    plt.title("Fluency Distribution")
    plt.xlabel("Fluency Score")
    plt.savefig(save_path + '/fluency_diagram.png')
    plt.close()

    diversity_data = t.diversity(generated_path, size=100)
    plt.figure()
    plt.hist(diversity_data, bins=50)
    plt.title("Diversity Distribution")
    plt.xlabel("Diversity Score")
    plt.savefig(save_path + '/diversity_diagram.png')
    plt.close()

    novelty_data = t.novelty(training_phrase=original_sentence, path=generated_path, size=5000, start=0)
    plt.figure()
    plt.hist(novelty_data, bins=50)
    plt.title("Novelty Distribution")
    plt.xlabel("Novelty Score")
    plt.savefig(save_path + '/novelty_diagram.png')
    plt.close()

def main():
    all_model_path = r'./logs'
    for model_path in os.listdir(all_model_path):
        print(model_path)
        if 'POEM' in model_path:
            model_log_dir = all_model_path + '/' + model_path  
            evaluate(model_log_dir + '/generate.txt', model_log_dir)

if __name__ == '__main__':
    main()