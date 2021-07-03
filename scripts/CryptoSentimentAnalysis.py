import numpy as np
import pandas as pd

DEFAULT_AUDIO_FILES_FOLDER = "data\\downloaded_audio_files"
DEFAULT_CLIP_FOLDER = "data\\extracted_clips"

DEFAULT_CLIP_LENGTH = 12  # in seconds


def get_sentiments(video_urls=[], playlist_urls=[], coins=["BTC", "ETH", "DOGE"],
                   wav2vec_model=None, sentiment_model=None,
                   audio_files_folder=DEFAULT_AUDIO_FILES_FOLDER,
                   clip_folder=DEFAULT_CLIP_FOLDER,
                   clip_length=DEFAULT_CLIP_LENGTH):

    """
    Gets sentiments for specified coins from audio/video files.

    :param video_urls:
    :param playlist_urls:
    :param coins:
    :param wav2vec_model:
    :param sentiment_model:
    :param audio_files_folder:
    :param clip_folder:
    :param clip_length:
    :return: Returns a data frame with the following structure: (Date, Author, Title, Coin, Sentiment)
    """

    # Download audio
    if video_urls is not None or playlist_urls is not None:
        download_audio_files(video_urls, playlist_urls, audio_files_folder)

    # Collect Audio Files (from audio_files_folder)

    # Extract audio clips
    clip_audio_files(source_audio_files_folder=audio_files_folder, clip_folder=clip_folder, clip_length=clip_length)

    # Speech to text

    # Label coin

    # Filter for coins?

    # Extract audio features

    # Label sentiment



def download_audio_files(video_urls=[], playlist_urls=[], save_audio_files_folder=DEFAULT_AUDIO_FILES_FOLDER):

    # Download

    # Ensure proper naming

    pass


def clip_audio_files(source_audio_files_folder=DEFAULT_AUDIO_FILES_FOLDER,
                     clip_folder=DEFAULT_CLIP_FOLDER,
                     clip_length=DEFAULT_CLIP_LENGTH):
    pass
