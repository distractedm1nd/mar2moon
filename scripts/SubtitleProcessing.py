import re

BTC_FILTER = ["bitcoin", "btc"]
ETH_FILTER = ["ethereum", " eth "]
DOGE_FILTER = ["doge", "dogecoin"]
CRYPTO_SPACE_FILTER = ["Crypto",
                       "Cryptocurrency",
                       "Blockchain",
                       "Miner",
                       "Mining",
                       "Wallet",
                       "Ether",
                       "Defi",
                       "Exchange",
                       "Value",
                       "Decentralized",
                       "Decentralization"]

DEFAULT_TEXT_CHUNK_LABELS = {
    "DOGE": DOGE_FILTER,
    "ETH": ETH_FILTER,
    "BTC": BTC_FILTER,
    "crypto_space": CRYPTO_SPACE_FILTER
}


def generate_text_chunks(subtitle_file, chunk_size, min_chunk_size):
    """Generates text chunks with start and end timestamps from a subtitle file.

    Parameters
    ----------
    subtitle_file : str
        Paths of the subtitle file to be used.
    chunk_size : int
        Word count of the chunks.
    min_chunk_size : int
        Chunks with less words will be discarded.

    Returns
    -------
    List of text chunks which are lists of words, Lists of corresponding start and end timestamps for each chunk.
    """

    text_chunks = list()
    chunk_start_times = list()
    chunk_end_times = list()

    if subtitle_file[-4:] == ".vtt":
        # Subtitle file is in vtt format.

        words, word_end_times = get_words_with_end_times(subtitle_file)

        if words is None:
            print("Could not generate text chunks for file: " + subtitle_file)
            return None, None, None

        # Generate text chunks of desired size
        text_chunks, chunk_start_times, chunk_end_times = generate_text_chunks_from_word_list(words, word_end_times,
                                                                                              chunk_size)
    elif subtitle_file[-4:] == ".txt":
        # Subtitle file is a plain text.
        # Possibly approximate timestamps?
        pass

    # Discard last chunk if too small
    if len(text_chunks[-1]) < min_chunk_size:
        text_chunks.pop()
        chunk_start_times.pop()
        chunk_end_times.pop()

    return text_chunks, chunk_start_times, chunk_end_times


def auto_label_text_chunk_default_labels(text):
    return auto_label_text_chunk(text, DEFAULT_TEXT_CHUNK_LABELS)


def auto_label_text_chunk(text, labels):
    """Tries to label a text chunk by finding a keyword in the text

    Parameters
    ----------
    text : str
        The text chunk
    labels : dict
        A dict of possible labels (keys in dict). The values of the dict are keywords for the labels.
        If a keyword is found in the text the text chunk will be labelled with the corresponding label.

    Returns
    -------
    Returns a label for the text. "None" if no suitable label was found.
    """

    for label in labels:
        regex_str = "|".join(labels[label])
        r = re.compile(regex_str, flags=re.I)
        if re.search(r, text) is not None:
            return label

    return "None"


def get_words_with_end_times(subtitle_file_path):
    """Get all words from a subtitle file (vtt format) with their corresponding end timestamps

    """

    with open(subtitle_file_path) as subtitle_file:

        # Remove first 4 lines (containing meta information)
        for j in range(0, 4):
            subtitle_file.readline()

        text = subtitle_file.read()

        # Check if the subtitle file supports individual word times
        if text.find("<c>") == -1:
            print("Individual word times are not supported for file: " + subtitle_file_path)
            return None, None

        chunks = text.split(" \n\n")  # split into chunks for easier data processing

        words = list()
        word_end_times = list()

        for chunk in chunks:
            chunk_lines = chunk.split("\n")
            words_line = chunk_lines[2]

            words_in_chunk = []
            word_end_times_in_chunk = []

            first_word_end_index = words_line.find("<")
            if first_word_end_index != -1:
                first_word = words_line[
                             0:first_word_end_index]  # get the first word (can't be found using method below)

                words_in_chunk = re.findall("<c> [\S]*</c>", words_line)  # get all words
                words_in_chunk = [w[4:-4] for w in words_in_chunk]  # strip <c> and <c/>

                word_end_times_in_chunk = re.findall("<\d\d:\d\d:\d\d.\d\d\d>", words_line)  # get all word end times
                word_end_times_in_chunk = [t[1:-1] for t in word_end_times_in_chunk]  # strip < and >
            else:
                # Only one word
                first_word = words_line

            last_time = chunk_lines[4][17:29]  # end time for the last word

            words_in_chunk.insert(0, first_word)
            word_end_times_in_chunk.append(last_time)

            words.extend(words_in_chunk)
            word_end_times.extend(word_end_times_in_chunk)

        # For the last chunk we have to get the word end time from somewhere else
        first_line_in_last_chunk = chunks[-1].split("\n")[0]
        last_time = first_line_in_last_chunk[17:29]
        word_end_times.pop()
        word_end_times.append(last_time)

        if len(words) != len(word_end_times):
            print("Warning: word count does not match times count")

        return words, word_end_times


def generate_text_chunks_from_word_list(words, word_end_times, chunk_size):
    """Generates text chunks with a start and end timestamp from a list of words.

    """

    text_chunks = list()
    chunk_start_times = list()
    chunk_end_times = list()

    for i in range(0, len(words), chunk_size):
        text_chunks.append(words[i:i + chunk_size])

        # Save start time
        if len(chunk_end_times) == 0:
            chunk_start_times.append("00:00:00.000")
        else:
            chunk_start_times.append(chunk_end_times[-1])

        # Save end time
        chunk_end_times.append(word_end_times[min((i + chunk_size)-1, len(word_end_times) - 1)])

    return text_chunks, chunk_start_times, chunk_end_times
