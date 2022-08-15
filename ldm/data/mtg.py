import multiprocessing
import os
import warnings
from pathlib import Path
from typing import TypedDict, List, Dict

import pandas as pd
import torch
import torchaudio
from einops import rearrange
from mdct import mdct, imdct
from parallelbar import progress_map
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

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


class MtgMdct(Dataset):
    def __init__(self,
                 split: str,
                 sampling_rate=22050,
                 size=256
                 ):
        super(MtgMdct).__init__()
        self.size = size
        self.sampling_rate = sampling_rate
        self.n_fft = size * 2  # for MDCT
        self.sample_size = (size - 1) * size - 2 * size + self.n_fft
        self.sample_durationInSec = self.sample_size / sampling_rate
        print(f"sample_size: {self.sample_size} ({self.sample_size * 1000 / sampling_rate:.3f}ms)")
        self.mtg_base = MTGFullAudio(split=split, sampling_rate=sampling_rate)

        self.samples = self.preprocess()

    def get_mdct_path(self, track_id: int, sample_id: int) -> Path:
        return Path(
            self.mtg_base.data_root) / "mdct" / f"{self.size}-{self.sampling_rate}Hz" / f"{track_id}" / f"{sample_id}.pt"

    def save_mdct(self, track_id, sample_id, tensor):
        mdct_path = self.get_mdct_path(track_id, sample_id)
        mdct_path.parent.mkdir(parents=True, exist_ok=True)
        with mdct_path.open("wb") as f:
            torch.save(tensor, f)

    def load_mdct(self, track_id, sample_id):
        mdct_path = self.get_mdct_path(track_id, sample_id)
        if mdct_path.exists():
            return torch.load(mdct_path)
        else:
            raise FileNotFoundError(f"{mdct_path=} not found")

    def mdct_samples(self, track_id) -> List[Path]:
        track_mdct_path = self.get_mdct_path(track_id, 0).parent
        return [p for p in track_mdct_path.glob(f"*.pt")]

    def mdct_is_processed(self, track_id) -> bool:
        track_mdct_path = self.get_mdct_path(track_id, 0).parent
        return track_mdct_path.joinpath("processed").exists()

    def mdct_mark_processed(self, track_id):
        track_mdct_path = self.get_mdct_path(track_id, 0).parent
        track_mdct_path.joinpath("processed").touch()

    def preprocess_track(self, track_id):
        track_info = self.mtg_base.tracks[track_id]
        track_samples = []
        self.mtg_base.print_audio_metadata = False
        if self.mdct_is_processed(track_id):
            for sample_path in self.mdct_samples(track_id):
                sample = {
                    **track_info,
                    "sample_id": int(sample_path.stem)
                }
                track_samples.append(sample)
        else:
            audio_repr = self.mtg_base.load_audio(self.mtg_base.get_audio_path(track_id))
            mdct_repr = torch.from_numpy(mdct(audio_repr, framelength=self.n_fft))
            mdct_repr_unfolded = rearrange(mdct_repr.unfold(1, self.size, self.size), "f u t -> u f t")
            for sample_id, sample_tensor in enumerate(mdct_repr_unfolded):
                self.save_mdct(track_id, sample_id, sample_tensor)
                sample = {
                    **track_info,
                    "sample_id": sample_id
                }
                track_samples.append(sample)
            self.mdct_mark_processed(track_id)
        return track_samples

    def preprocess(self):
        print("Preprocessing mdcts...")
        samples = progress_map(self.preprocess_track, self.mtg_base.track_ids)
        return [item for sublist in samples for item in sublist]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# main
if __name__ == "__main__":
    sampling_rate = 22050
    size = 256

    dataset = MtgMdct(split="train_0",
                      sampling_rate=sampling_rate,
                      size=size
                      )
    a1 = dataset[0]
    print(a1["audio_repr"].shape)
    print("DONE")