from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

sid = SentimentIntensityAnalyzer()


def review_rating(text):
    scores = sid.polarity_scores(text)
    if scores['compound'] == 0:
        return 'neutral'
    elif scores['compound'] > 0:
        return 'bullish'
    else:
        return 'bearish'


def create_array_of_fixed_length(array_words_file, n):
    result = []
    array_of_words_file = []
    for i in range(0, int(len(array_words_file)) + 1, n):
        temp = array_words_file[i:i + n]
        array_of_words_file.append(temp)
    for array_words in array_of_words_file:
        result.append(" ".join(str(word) for word in array_words))
    return result


def create_labeling_to_each_pargraph(folder_path):
    result = []
    for file in os.listdir(folder_path):
        result_of_each_document = {}
        file = open(folder_path + "/" + file, "r").read()
        array_of_words_file = file.split(" ")
        array_of_fixed_length = create_array_of_fixed_length(array_of_words_file, 20)
        for array in array_of_fixed_length:
            result_of_each_document[array] = review_rating(array)
        result.append(result_of_each_document)
    return result


if __name__ == '__main__':
    text_files = create_labeling_to_each_pargraph("transcriptsFolder")
    for file in text_files:
        print(file)

