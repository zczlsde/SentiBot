from itertools import compress
import language_tool_python


def fluency(verbose = 0):
    """
    Unfinished
    Give a score on the average fluency and fluency distribution

    Returns:
            error rate: single float
            error rates: array of float
    """
    with open("./Data/Generations/perc_generations_500.txt", encoding="utf8") as f:
        lines = f.readlines()

    phrases = []
    phrase = ""
    for x in lines:
        if x.startswith("==="):
            phrases.append(phrase)
            phrase = ""
            continue
        phrase = phrase + x
    print(len(phrases))

    # mask = [not x.startswith("===") for x in lines]
    # sentences = list(compress(lines, mask))   
    tool = language_tool_python.LanguageTool("en-US")
    total_length = 0
    total_errors = 0
    error_rate = 0
    error_rates = []
    counter = 0
    for phrase in phrases:
        if verbose >= 1:
            counter += 1
            print(counter)
        length = len(phrase.split())
        total_length += length
        matches = tool.check(phrase)
        naive_errors = len(matches)
        total_errors += naive_errors
        if length > 0:
            error_rates.append(1-naive_errors/length)
    tool.close()
    error_rate = 1-total_errors/total_length
    return error_rate, error_rates

if __name__ == "__main__":
    rate, rates = fluency(verbose=1)
    print(rate, rates)
