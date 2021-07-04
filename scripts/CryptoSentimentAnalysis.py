import subprocess
import numpy as np
import pandas as pd
from os import listdir
import VideoDownloader
import AudioFeatureExtraction

DEFAULT_AUDIO_FILES_FOLDER = "data\\downloaded_audio_files"
DEFAULT_CLIP_FOLDER = "data\\extracted_clips"

DEFAULT_CLIP_LENGTH = 12  # in seconds

FILE_NAME_SEPARATOR = "-sep-"


def get_sentiments(video_urls=[],
                   playlist_urls=[],
                   start_date=None,
                   end_date=None,
                   coins=["BTC", "ETH", "DOGE"],
                   audio_files_folder=DEFAULT_AUDIO_FILES_FOLDER,
                   clips_folder=DEFAULT_CLIP_FOLDER,
                   clip_length=DEFAULT_CLIP_LENGTH,
                   praat_path="praat.exe",
                   praat_script="praat\\GetAudioFeatures.praat",
                   wav2vec_model=None,
                   sentiment_model=None,
                   use_audio_features=True):
    """
    Gets sentiments for specified coins from audio/video files.


    :param video_urls: List of video/audio URLs to do download.
    :param playlist_urls: List of playlist URLs to download.
    :param start_date: Do not use videos/audios before this date. Format: YYYYMMDD.
    :param end_date: Do not use videos/audios after this date. Format: YYYYMMDD.
    :param coins: List of coins to find the sentiments for. Only BTC, ETH and DOGE are supported.
    :param audio_files_folder: The folder where downloaded files are stored and the source folder of all audio files for
     the analysis.
    :param clips_folder: Extracted audio clips are stored in this folder.
    :param clip_length: The length of each audio clip.
    :param praat_script: Path to the Praat installation.
    :param praat_path: The Praat script used for the audio feature extraction.
    :param wav2vec_model: Trained speech to text model.
    :param sentiment_model: Trained sentiment analysis model.
    :param use_audio_features: Use audio features for sentiment labelling.
    :return: Returns a data frame with the following structure: (Date, Author, Title, Coin, Sentiment)
    """

    # Download audio.
    if len(video_urls) > 0 or len(playlist_urls) > 0:
        download_audio_files(video_urls=video_urls,
                             playlist_urls=playlist_urls,
                             start_date=start_date,
                             end_date=end_date,
                             save_audio_files_folder=audio_files_folder)

        print("Download finished")

    # Collect audio files in a data frame.

    # This data frame contains information about the video files to be used in further analysis.
    all_file_names = listdir(audio_files_folder)

    video_files_info = []
    for file_name in all_file_names:
        if file_name[-4:] == ".wav":
            video_info = file_name[:-4].split(FILE_NAME_SEPARATOR)
            video_files_info.append(video_info)

    # Save video info in a data frame.
    df_video_files_info = pd.DataFrame(video_files_info, columns=["Author", "Date", "Title"])
    # Reorder data frame
    df_video_files_info = df_video_files_info[["Date", "Author", "Title"]]
    df_video_files_info["Date"] = df_video_files_info["Date"].astype("int")
    # print(df_video_files_info)

    # Only keep entries in the user specified date range.
    if start_date is not None:
        start_date_int = int(start_date)
        df_video_files_info = df_video_files_info[df_video_files_info["Date"] >= start_date_int]
    if end_date is not None:
        end_date_int = int(end_date)
        df_video_files_info = df_video_files_info[df_video_files_info["Date"] <= end_date_int]
    # print(df_video_files_info)

    # Extract audio clips

    extract_clips_from_audio_files(df_video_info=df_video_files_info,
                                   source_audio_files_folder=audio_files_folder,
                                   clip_folder=clips_folder,
                                   clip_length=clip_length)

    print("Clips extracted")

    # Start building the final data set

    all_file_names = listdir(clips_folder)

    clip_files_info = []
    for file_name in all_file_names:
        if file_name[-4:] == ".wav":
            clip_info = file_name[:-4].split(FILE_NAME_SEPARATOR)
            clip_info.append(file_name)
            clip_files_info.append(clip_info)

    # Save clip info in a data frame.
    df = pd.DataFrame(clip_files_info, columns=["Author", "Date", "Title", "Clip_Id", "File_Name"])
    # Reorder data frame
    df = df[["Date", "Author", "Title", "Clip_Id", "File_Name"]]

    # print(df)

    # Speech to text
    df["Text"] = "wav2vec output"  # Todo

    # Label coin

    # Filter for coins (do not extract audio features for text without a coin, because takes to long)

    # Extract audio features

    # Get audio features for each clip.
    audio_features = df.apply(
        lambda x: AudioFeatureExtraction.get_audio_features(clips_folder + "\\" + x["File_Name"],
                                                            praat_path,
                                                            praat_script), axis=1)

    # Make sure audio_features has the same in every entry.
    none_list = [None, None, None, None, None, None, None, None, None, None, None]
    audio_features = audio_features.apply(lambda x: none_list if len(x) == 1 else x)

    # Create a data frame with audio features.
    df_audio_features = pd.DataFrame(np.vstack(audio_features), columns=["Pitch_Min",
                                                                         "Pitch_Max",
                                                                         "Pitch_05_Quantile",
                                                                         "Pitch_95_Quantile",
                                                                         "Pitch_Range",
                                                                         "pitch stdev",
                                                                         "Pitch_Mean",
                                                                         "Pitch_Median",
                                                                         "Jitter",
                                                                         "Shimmer",
                                                                         "Hammarberg_Index"])

    df = pd.concat([df, df_audio_features], axis=1)

    print("Audio features extracted")

    # print(df)

    # Label sentiment

    # Return subset of the data frame


def download_audio_files(video_urls=[],
                         playlist_urls=[],
                         start_date=None,
                         end_date=None,
                         save_audio_files_folder=DEFAULT_AUDIO_FILES_FOLDER):
    # Download

    ensure_correct_naming(save_audio_files_folder)


def ensure_correct_naming(folder_path):
    """Renames (downloaded) files to avoid problematic characters.

    This function is executed after downloading videos to ensure correct naming of the files.
    It can also be used to convert names of previously downloaded files.
    """

    VideoDownloader.ensure_correct_naming(folder_path)


def extract_clips_from_audio_files(df_video_info=None,
                                   source_audio_files_folder=DEFAULT_AUDIO_FILES_FOLDER,
                                   clip_folder=DEFAULT_CLIP_FOLDER,
                                   clip_length=DEFAULT_CLIP_LENGTH):
    """
    Extracts clips of equal length from audio files specified in a data frame.

    :param df_video_info: Data frame containing the videos to extract clips from.
    :param source_audio_files_folder: Folder containing the source audio files.
    :param clip_folder: Folder to save the extracted clips
    :param clip_length: Length of the clips.
    :return:
    """

    df_video_info.apply(lambda x: extract_clip_for_video_info_data_frame_row(x,
                                                                             source_audio_files_folder,
                                                                             clip_folder,
                                                                             clip_length), axis=1)


def extract_clip_for_video_info_data_frame_row(row, source_audio_files_folder, clip_folder, clip_length):
    audio_file_name = get_file_name_from_video_info_data_frame_row(row)
    extract_audio_clips_equal_length(source_audio_files_folder + "\\" + audio_file_name,
                                     audio_file_name[:-4],
                                     clip_folder,
                                     clip_length)


def extract_audio_clips_equal_length(audio_file, output_clip_base_name, save_clips_folder, clip_length):
    """
    Cuts an audio file in clips of equal length.

    :param audio_file: The source audio file.
    :param output_clip_base_name: Base name of the extracted audio clips.
    :param save_clips_folder: Folder to save the extracted clips.
    :param clip_length: Length of the clips.
    :return:
    """

    output_file = save_clips_folder + "\\" + output_clip_base_name + FILE_NAME_SEPARATOR + "%04d.wav"

    audio_file = audio_file.replace("/", "\\")
    output_file = output_file.replace("/", "\\")

    subprocess.call([
        "ffmpeg",
        "-i", audio_file,
        "-ar", "16000",  # downsample to 16Khz
        "-ac", "1",  # stereo -> mono
        "-f", "segment",
        "-segment_time", str(clip_length),
        "-c", "copy",
        output_file])


def get_file_name_from_video_info_data_frame_row(row):
    """
    Reconstructs the file name from a data row in the video info data frame.

    :param row: Row in the video info data frame.
    :return: The file name of the video of the corresponding row.
    """

    return row["Author"] + FILE_NAME_SEPARATOR + str(row["Date"]) + FILE_NAME_SEPARATOR + row["Title"] + ".wav"
