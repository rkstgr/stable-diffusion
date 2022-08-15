import os
import warnings
from pathlib import Path
from typing import TypedDict, List, Dict

import pandas as pd
import torch
import torchaudio
from einops import rearrange
from mdct import mdct, imdct
from torch.utils.data import Dataset, IterableDataset

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
                 split: str,
                 tsv_file=None,
                 data_root=None,
                 file_type=".opus",
                 sampling_rate=48000,
                 ):
        self.data_root = data_root if data_root else Path(os.environ["MTG_DATA_ROOT"])
        tsv_file = tsv_file if tsv_file else Path(self.data_root).joinpath(f"{split}.tsv")
        self.file_type = file_type
        self.sampling_rate = sampling_rate

        self.tracks: Dict[TrackId, TrackInfo] = _load_track_info(tsv_file)
        self.track_ids = list(self.tracks.keys())
        print("Loaded {} tracks from {}".format(len(self.tracks), tsv_file))

        # check that all audio tracks are present
        for track_id in self.track_ids:
            audio_path = self.get_audio_path(track_id)
            if not audio_path.exists():
                raise FileNotFoundError(f"{audio_path=} not found")

    def load_audio(self, audio_path: Path) -> torch.Tensor:
        raise NotImplementedError

    def get_audio_path(self, track_id: TrackId) -> Path:
        return Path(self.data_root) / "opus" / f"{track_id}{self.file_type}"

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, i):
        track_id = self.track_ids[i]
        track: TrackInfo = self.tracks[track_id]
        audio_path = self.get_audio_path(track_id)

        audio_repr = self.load_audio(audio_path)
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

    def load_audio(self, audio_path):
        if getattr(self, "print_audio_metadata", True):
            metadata = torchaudio.info(audio_path)
            print(metadata)
            self.print_audio_metadata = False

        waveform, sr = torchaudio.load(audio_path)

        # resample if necessary
        if sr != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sampling_rate)(waveform)

        # for now, assume that all audio files are mono
        if waveform.shape[0] == 2:
            waveform = waveform.mean(dim=0)

        return waveform


class MtgMdct(IterableDataset):
    def __init__(self,
                 split: str,
                 sampling_rate=22050,
                 size=256
                 ):
        super(MtgMdct).__init__()
        self.size = size
        self.n_fft = size * 2  # for MDCT
        self.sample_size = (size - 1) * size - 2 * size + self.n_fft
        self.sample_durationInSec = self.sample_size / sampling_rate
        print(f"sample_size: {self.sample_size} ({self.sample_size * 1000 / sampling_rate:.3f}ms)")

        self.mtg_base = MTGFullAudio(split=split, sampling_rate=sampling_rate)

    def __len__(self):
        """
        Note:
            This is only an approximation, use with caution.
        """
        warnings.warn("This is only an approximation of the actual dataset length, use with caution.")
        total_samples = 0
        for track_id in self.mtg_base.track_ids:
            track_duration = self.mtg_base.tracks[track_id]["durationInSec"]
            total_samples += track_duration // self.sample_durationInSec  # (sizedim - size) / step + 1
        return total_samples

    def __iter__(self):
        def iterator():
            for i, example in enumerate(self.mtg_base):
                audio_repr = example["audio_repr"]
                mdct_repr = torch.from_numpy(mdct(audio_repr, framelength=self.n_fft))
                mdct_repr_unfolded = rearrange(mdct_repr.unfold(1, self.size, self.size), "f u t -> u f t")
                for i, sample in enumerate(mdct_repr_unfolded):
                    yield {
                        **example,
                        "audio_sample_nr": i,
                        "audio_repr": sample[..., None],
                    }

        return iterator()

