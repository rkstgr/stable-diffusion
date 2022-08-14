from pathlib import Path
from typing import TypedDict, List, Dict

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

TrackId = int


class TrackInfo(TypedDict):
    artist_id: int
    album_id: int
    durationInSec: float
    genres: List[str]
    instruments: List[str]
    moods: List[str]
    chunk_nr: int


def _load_track_info(track_file) -> Dict[TrackId, TrackInfo]:
    """
    Load tracks from tsv files and return dictionary with track_id as key and TrackInfo as value.
    """
    df = pd.read_csv(track_file, sep="\t")
    df.genres = df.genres.apply(eval)
    df.instruments = df.instruments.apply(eval)
    df.moods = df.moods.apply(eval)
    return df.set_index("id").to_dict("index")


class MTGBase(Dataset):
    """
    Args:
        tsv_file: path to tsv file with track info
        audio_root: path to the audio files, will search recursively for all files with file_type
        file_type: file type of audio files, e.g. "opus" or "mp3"
    """

    def __init__(self,
                 tsv_file,
                 data_root,
                 file_type=".opus",
                 sampling_rate=48000,
                 ):
        self.data_root = data_root
        self.file_type = file_type
        self.sampling_rate = sampling_rate

        self.tracks: Dict[TrackId, TrackInfo] = _load_track_info(tsv_file)
        self.track_ids = list(self.tracks.keys())
        print("Loaded {} tracks from {}".format(len(self.tracks), tsv_file))

        # check that all audio tracks are present
        for track_id in self.track_ids:
            audio_file = Path(self.data_root).joinpath(str(track_id) + file_type)
            if not audio_file.exists():
                raise FileNotFoundError(f"{audio_file=} not found")

    def load_audio(self, audio_file) -> torch.Tensor:
        raise NotImplementedError

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, i):
        track_id = self.track_ids[i]
        track: TrackInfo = self.tracks[track_id]
        audio_file = Path(self.data_root).joinpath(str(track_id) + self.file_type)

        audio_repr = self.load_audio(audio_file)
        example = {
            "track_id": track_id,
            "artist_id": track["artist_id"],
            "album_id": track["album_id"],
            "genres": track["genres"],
            "audio_repr": audio_repr,
        }
        return example


class MTGFullAudio(MTGBase):
    """
    Loads full audio files resampled to sampling_rate and as mono channel.

    Note:
        This is a very slow operation, as it requires loading the entire audio file into memory.
    """

    def load_audio(self, audio_file):
        if getattr(self, "print_audio_metadata", True):
            metadata = torchaudio.info(audio_file)
            print(metadata)
            self.print_audio_metadata = False

        waveform, sr = torchaudio.load(audio_file)

        # resample if necessary
        if sr != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sampling_rate)(waveform)

        # for now, assume that all audio files are mono
        if waveform.shape[0] == 2:
            waveform = waveform.mean(dim=0)

        return waveform
