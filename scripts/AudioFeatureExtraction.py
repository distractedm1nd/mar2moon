import pandas as pd
import numpy as np
import re
import subprocess


def extract_audio_clip(audio_file, output_clip_name, output_folder, start_time, end_time):
    """Extracts an audio clip from audio file starting at start_time and ending at end_time using ffmpeg.

    """

    output_file = output_folder + "/" + output_clip_name

    audio_file = audio_file.replace("/", "\\")
    output_file = output_file.replace("/", "\\")

    subprocess.call([
        "ffmpeg",
        "-ss", start_time,
        "-to", end_time,
        "-i", audio_file,
        "-ar", "16000",  # downsample to 16Khz
        "-ac", "1",  # stereo -> mono
        output_file])

    print("save: " + output_file)


def extract_audio_clip_by_sentiment_labelled_data_row(row, audio_files_folder, output_folder, overwrite_podcast_name="",
                                                      correct_file_extension=False):
    """Extracts an audio clip based on a row in the manually labelled sentiment data.

    """

    podcast_title = row["Podcast_Title"]
    if len(overwrite_podcast_name) > 0:
        podcast_title = overwrite_podcast_name
    if correct_file_extension:
        podcast_title = podcast_title[:-7]

    audio_file = audio_files_folder + "/" + podcast_title + ".wav"
    start_string = row["Start_Time"].replace(":", ".")
    end_string = row["End_Time"].replace(":", ".")
    output_clip_name = podcast_title + "_" + start_string + "_" + end_string + ".wav"

    extract_audio_clip(audio_file, output_clip_name, output_folder, row["Start_Time"], row["End_Time"])
