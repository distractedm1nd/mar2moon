import numpy as np
import pandas as pd
from mar2moon.scripts import VideoDownloader

DEFAULT_AUDIO_FILES_FOLDER = "data\\downloaded_audio_files"
DEFAULT_CLIP_FOLDER = "data\\extracted_clips"

DEFAULT_CLIP_LENGTH = 12  # in seconds


def get_sentiments(video_urls=[],
                   playlist_urls=[],
                   start_date=None,
                   end_date=None,
                   coins=["BTC", "ETH", "DOGE"],
                   audio_files_folder=DEFAULT_AUDIO_FILES_FOLDER,
                   clip_folder=DEFAULT_CLIP_FOLDER,
                   clip_length=DEFAULT_CLIP_LENGTH,
                   wav2vec_model=None,
                   sentiment_model=None,
                   use_audio_features=True):

    """
    Gets sentiments for specified coins from audio/video files.


    :param video_urls: List of video/audio URLs to do download.
    :param playlist_urls: List of playlist URLs to download.
    :param start_date: Do not use videos/audios before this date.
    :param end_date: Do not use videos/audios after this date.
    :param coins: List of coins to find the sentiments for. Only BTC, ETH and DOGE are supported.
    :param audio_files_folder: The folder where downloaded files are stored and the source folder of all audio files for
     the analysis.
    :param clip_folder: Extracted audio clips are stored in this folder.
    :param clip_length: The length of each audio clip.
    :param wav2vec_model: Trained speech to text model.
    :param sentiment_model: Trained sentiment analysis model.
    :param use_audio_features: Use audio features for sentiment labelling.
    :return: Returns a data frame with the following structure: (Date, Author, Title, Coin, Sentiment)
    """

    # Download audio
    if video_urls is not None or playlist_urls is not None:
        download_audio_files(video_urls=video_urls,
                             playlist_urls=playlist_urls,
                             start_date=start_date,
                             end_date=end_date,
                             save_audio_files_folder=audio_files_folder)

    # Collect Audio Files (from audio_files_folder)

    # Extract audio clips
    clip_audio_files(source_audio_files_folder=audio_files_folder, clip_folder=clip_folder, clip_length=clip_length)

    # Speech to text

    # Label coin

    # Filter for coins?

    # Extract audio features

    # Label sentiment


def download_audio_files(video_urls=[],
                         playlist_urls=[],
                         start_date=None,
                         end_date=None,
                         save_audio_files_folder=DEFAULT_AUDIO_FILES_FOLDER):

    # Download

    ensure_correct_naming(save_audio_files_folder)

    pass


def ensure_correct_naming(folder_path):
    """Renames (downloaded) files to avoid problematic characters.

    This function is executed after downloading videos to ensure correct naming of the files.
    It can also be used to convert names of previously downloaded files.
    """

    VideoDownloader.ensure_correct_naming(folder_path)


def clip_audio_files(source_audio_files_folder=DEFAULT_AUDIO_FILES_FOLDER,
                     clip_folder=DEFAULT_CLIP_FOLDER,
                     clip_length=DEFAULT_CLIP_LENGTH):
    pass
