import subprocess
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from os import listdir
import VideoDownloader
import AudioFeatureExtraction


class SentimentAnalysisPipeline:
    DEFAULT_AUDIO_FILES_FOLDER = "data\\downloaded_audio_files"
    DEFAULT_CLIP_FOLDER = "data\\extracted_clips"
    DEFAULT_WAV2VEC_REPOSITORY = "distractedm1nd/wav2vec-en-finetuned-on-cryptocurrency"
    DEFAULT_CLIP_LENGTH = 15  # in seconds
    DEFAULT_COINS = ["BTC", "ETH", "DOGE"]
    DEFAULT_PRAAT_PATH = "praat.exe",
    DEFAULT_PRAAT_SCRIPT = "praat\\GetAudioFeatures.praat"
    DEFAULT_FILE_NAME_SEPARATOR = "-sep-"

    def __init__(self,
                 coins=DEFAULT_COINS,
                 audio_files_folder=DEFAULT_AUDIO_FILES_FOLDER,
                 clips_folder=DEFAULT_CLIP_FOLDER,
                 clip_length=DEFAULT_CLIP_LENGTH,
                 praat_path=DEFAULT_PRAAT_PATH,
                 praat_script=DEFAULT_PRAAT_SCRIPT,
                 separator=DEFAULT_FILE_NAME_SEPARATOR,
                 wav2vec_model=None,
                 wav2vec_processor=None,
                 sentiment_model=None,
                 use_audio_features=True):

        """
        Initializes the crypto sentiment analysis pipeline

        :param coins: List of coins to find the sentiments for. Only BTC, ETH and DOGE are supported.
        :param audio_files_folder: The folder where downloaded files are stored and the source folder of all audio files for
         the analysis.
        :param clips_folder: Extracted audio clips are stored in this folder.
        :param clip_length: The length of each audio clip.
        :param praat_script: Path to the Praat installation.
        :param praat_path: The Praat script used for the audio feature extraction.
        :param separator: Separator used in filenames
        :param wav2vec_model: Trained speech to text model.
        :param wav2vec_processor: Trained wav2vec processor.
        :param sentiment_model: Trained sentiment analysis model.
        :param use_audio_features: Use audio features for sentiment labelling.
        :return: CryptoSentimentAnalysis Pipeline instance
        """

        self.coins = coins
        self.audio_files_folder = audio_files_folder
        self.clips_folder = clips_folder
        self.clip_length = clip_length
        self.praat_path = praat_path
        self.praat_script = praat_script
        self.wav2vec_model = wav2vec_model
        self.separator = separator
        self.wav2vec_processor = wav2vec_processor
        self.sentiment_model = sentiment_model
        self.use_audio_features = use_audio_features

        if wav2vec_model is None or wav2vec_processor is None:
            print("Downloading standard Wav2Vec model and processor")
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(self.DEFAULT_WAV2VEC_REPOSITORY)
            self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained(self.DEFAULT_WAV2VEC_REPOSITORY)

    def get_sentiments(self, video_urls=[], playlist_urls=[], start_date=None, end_date=None):
        """
        Gets sentiments for specified coins from audio/video files.


        :param video_urls: List of video/audio URLs to do download.
        :param playlist_urls: List of playlist URLs to download.
        :param start_date: Do not use videos/audios before this date. Format: YYYYMMDD.
        :param end_date: Do not use videos/audios after this date. Format: YYYYMMDD.
        :return: Returns a data frame with the following structure: (Date, Author, Title, Coin, Sentiment)
        """

        # Download audio.
        if len(video_urls) > 0 or len(playlist_urls) > 0:
            self.download_audio_files(video_urls=video_urls,
                                      playlist_urls=playlist_urls,
                                      start_date=start_date,
                                      end_date=end_date)

            print("Download finished")

        # Collect audio files in a data frame.

        # This data frame contains information about the video files to be used in further analysis.
        all_file_names = listdir(self.audio_files_folder)

        video_files_info = []
        for file_name in all_file_names:
            if file_name[-4:] == ".wav":
                video_info = file_name[:-4].split(self.separator)
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

        self.extract_clips_from_audio_files(df_video_info=df_video_files_info)

        print("Clips extracted")

        # Start building the final data set

        all_file_names = listdir(self.clips_folder)

        clip_files_info = []
        for file_name in all_file_names:
            if file_name[-4:] == ".wav":
                clip_info = file_name[:-4].split(self.separator)
                clip_info.append(file_name)
                clip_files_info.append(clip_info)

        # Save clip info in a data frame.
        df = pd.DataFrame(clip_files_info, columns=["Author", "Date", "Title", "Clip_Id", "File_Name"])
        # Reorder data frame
        df = df[["Date", "Author", "Title", "Clip_Id", "File_Name"]]

        # print(df)

        # Speech to text
        df["Text"] = df["File_Name"].apply(lambda x: self.get_wav2vec_output(x))

        print("Text extracted")

        # Label coin

        # Filter for coins (do not extract audio features for text without a coin, because takes to long)

        # Extract audio features

        # Get audio features for each clip.
        df_audio_features = self.get_audio_features_df(df)

        df = pd.concat([df, df_audio_features], axis=1)

        print("Audio features extracted")

        print(df)

        # Label sentiment

        # Return subset of the data frame

    def get_wav2vec_output(self, filename):
        #TODO: Make parameter use_cuda + batchsize for speedup, but it requires that all audio files are already loaded into the df

        # Reads audio file
        file = self.clips_folder + "\\" + filename
        file = file.replace("\\", "/")
        audio, sampling_rate = sf.read(file)
        assert(sampling_rate == 16_000, "Sampling rate was not 16k.")

        # Batch size 1
        input_values = self.wav2vec_processor(audio, return_tensors="pt", padding="longest", sampling_rate=sampling_rate).input_values

        # retrieve logits
        logits = self.wav2vec_model(input_values).logits

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.wav2vec_processor.batch_decode(predicted_ids)[0].lower()

    def download_audio_files(self, video_urls=[], playlist_urls=[], start_date=None, end_date=None):
        # Download

        VideoDownloader.ensure_correct_naming(self.audio_files_folder)

    def extract_clips_from_audio_files(self, df_video_info=None):
        """
        Extracts clips of equal length from audio files specified in a data frame.

        :param df_video_info: Data frame containing the videos to extract clips from.
        :return:
        """

        df_video_info.apply(lambda x: self.extract_clip_for_video_info_data_frame_row(x), axis=1)

    def extract_clip_for_video_info_data_frame_row(self, row):
        audio_file_name = self.reconstruct_filename_from_metadata(row)
        self.extract_audio_clips_equal_length(self.audio_files_folder + "\\" + audio_file_name, audio_file_name[:-4])

    def extract_audio_clips_equal_length(self, audio_file, output_clip_base_name):
        """
        Cuts an audio file in clips of equal length.

        :param audio_file: The source audio file.
        :param output_clip_base_name: Base name of the extracted audio clips.
        :return:
        """
        # TODO: use os.path.join() when building filenames so it also works on linux+mac
        output_file = self.clips_folder + "\\" + output_clip_base_name + self.separator + "%04d.wav"

        audio_file = audio_file.replace("/", "\\")
        output_file = output_file.replace("/", "\\")

        subprocess.call([
            "ffmpeg",
            "-i", audio_file,
            "-ar", "16000",  # downsample to 16Khz
            "-ac", "1",  # stereo -> mono
            "-f", "segment",
            "-segment_time", str(self.clip_length),
            #"-c", "copy",
            output_file])

    def reconstruct_filename_from_metadata(self, row):
        """
        Reconstructs the file name from a data row in the video info data frame.

        :param row: Row in the video info data frame.
        :return: The file name of the video of the corresponding row.
        """

        return row["Author"] + self.separator + str(row["Date"]) + self.separator + row["Title"] + ".wav"

    def get_audio_features_df(self, df):
        """
        Generates a data frame with audio features for input df.

        :param df: Input data.
        :return: Data frame containing audio features.
        """

        audio_features = df["File_Name"].apply(
            lambda x: AudioFeatureExtraction.get_audio_features(self.clips_folder + "\\" + x,
                                                                self.praat_path,
                                                                self.praat_script))

        # Make sure audio_features has the same length in every entry.
        none_list = [None] * 11
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

        return df_audio_features

    def get_audio_features_df_parallel(self, df):
        """
        Generates a data frame with audio features for input df. Runs parallel for speed

        :param df:
        :return:
        """

        # Todo parallel library

        return self.get_audio_features_df(df)