import itertools
import math
import os
from pathlib import Path
from typing import Callable, TypedDict, List, Dict

import pandas as pd
import torch
import torchaudio
from einops import rearrange
from mdct import mdct
from torch.utils.data import Dataset, IterableDataset, DataLoader, default_collate
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

TrackId = int


class TrackInfo(TypedDict):
    artist_id: int
    album_id: int
    durationInSec: float
    genres: List[str]
    instruments: List[str]
    moods: List[str]
    chunk_nr: int


class Section(TrackInfo):
    track_id: int
    section_nr: int


def _load_track_info(track_file) -> Dict[TrackId, TrackInfo]:
    """
    Load tracks from tsv files and return dictionary with track_id as key and TrackInfo as value.
    """
    df = pd.read_csv(track_file, sep="\t")
    df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True)
    df.genres = df.genres.apply(eval)
    df.instruments = df.instruments.apply(eval)
    df.moods = df.moods.apply(eval)
    return df.set_index("id").to_dict("index")


class MTGBase(Dataset):
    """
    Args:
        split: "train" or "valid", determines the tsv file that will be loaded: data_root/split.tsv
        tsv_file: specify a different tsv file than the one specified by split
        data_root: specify a different data root than the one specified by os.environ["MTG_DATA_ROOT"]
        filter_predicate: a function that takes a TrackInfo and returns True if the track should be included in the dataset
        genres: a list of genres that should be included in the dataset (filter_predicate will overwrite this)
        file_type: the file type of the audio files (e.g. ".opus")
        sampling_rate: the sampling rate in which the audio files should be loaded
    """

    def __init__(self,
                 split: str,
                 tsv_file=None,
                 data_root=None,
                 filter_predicate: Callable[[TrackInfo], bool] = None,
                 genres: List[str] = None,
                 file_type=".opus",
                 sampling_rate=48000,
                 ):
        self.data_root = data_root if data_root else Path(os.environ["MTG_DATA_ROOT"])
        tsv_file = tsv_file if tsv_file else Path(self.data_root).joinpath(f"{split}.tsv")
        self.file_type = file_type
        self.sampling_rate = sampling_rate

        self.tracks: Dict[TrackId, TrackInfo] = _load_track_info(tsv_file)
        if not filter_predicate and genres:
            print("Filtering on genres: {}".format(genres))
            filter_predicate = lambda t: any([genre in t["genres"] for genre in genres])
        if filter_predicate:
            self.tracks = {k: v for k, v in self.tracks.items() if filter_predicate(v)}
        self.track_ids = list(self.tracks.keys())
        print("Loaded {} tracks from {}".format(len(self.tracks), tsv_file))
        total_duration = sum([t["durationInSec"] for t in self.tracks.values()])
        print(f"Total duration: {total_duration:.1f}s, {total_duration / 3600:.1f}h, {total_duration / 3600 / 24:.1f}d")

        print("Checking that all audio tracks are present")
        for track_id in self.track_ids:
            audio_path = self.get_audio_path(track_id)
            if not audio_path.exists():
                raise FileNotFoundError(f"{audio_path=} not found")

    def load_audio(self, audio_path: Path) -> torch.Tensor:
        raise NotImplementedError

    def get_audio_path(self, track_id: TrackId) -> Path:
        return Path(self.data_root) / "opus" / f"{track_id}{self.file_type}"

    def get_approx_audio_length(self, track_id: TrackId) -> int:
        return int(self.sampling_rate * self.tracks[track_id]["durationInSec"])

    def get_audio_length(self, track_id: TrackId) -> int:
        metadata = torchaudio.info(self.get_audio_path(track_id))
        frames = metadata.num_frames
        if metadata.sample_rate != self.sampling_rate:
            frames = math.ceil(frames * self.sampling_rate / metadata.sample_rate)
        return int(frames)

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
        waveform, sr = torchaudio.load(audio_path)

        # resample if necessary
        if sr != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sampling_rate)(waveform)

        # for now, assume that all audio files are mono
        if waveform.shape[0] == 2:
            waveform = waveform.mean(dim=0)

        return waveform


def mdct_length(audio_length, n_fft=512) -> int:
    return int(math.ceil(audio_length / n_fft) * n_fft / (n_fft // 2) + 1)


def generate_mdct_sections(audio_wave: torch.Tensor, size, step):
    """
    Returns a tensor with (n_sections, n_fft, n_frames)
    """
    audio_mdct = mdct(audio_wave, framelength=size * 2)
    audio_mdct = torch.from_numpy(audio_mdct).float()
    audio_mdcts = audio_mdct.unfold(1, size, step)
    audio_mdcts = rearrange(audio_mdcts, "f b t -> b f t")
    return audio_mdcts


class MtgMdctIterable(IterableDataset):
    def __init__(self, split: str, sampling_rate=22050, size=256, step=256, **kwargs):
        super().__init__()
        self.split = split
        self.size = size
        self.step = step
        self.sampling_rate = sampling_rate
        self.mtg_base = MTGFullAudio(split=split, sampling_rate=sampling_rate, **kwargs)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        track_ids = self.mtg_base.track_ids
        if worker_info is not None:
            num_workers = worker_info.num_workers
            track_ids = track_ids[worker_info.id::num_workers]
        for track_id in track_ids:
            audio_repr = self.mtg_base.load_audio(self.mtg_base.get_audio_path(track_id))
            mdct_sections = generate_mdct_sections(audio_repr, self.size, self.step)
            for i, mdct_view in enumerate(mdct_sections):
                yield {
                    "track_id": track_id,
                    "section_nr": i,
                    "mdct": mdct_view
                }

    def __len__(self):
        print("Returning approximate length")
        if not hasattr(self, "len_cached"):
            total_length = 0
            for track_id in self.mtg_base.track_ids:
                audio_length = self.mtg_base.get_approx_audio_length(track_id)
                n_sections = mdct_length(audio_length, self.size*2)
                total_length += n_sections
            self.len_cached = total_length
        return self.len_cached


def collate_unbatch(x):
    return {k: v[0] for k, v in default_collate(x).items()}


# main
if __name__ == "__main__":
    test_sampling_rate = 22050
    test_size = 256
    batch_size = 4

    dataset = MtgMdctIterable(split="train_dev",
                              # genres=["classical"],
                              size=test_size,
                              step=test_size,
                              sampling_rate=test_sampling_rate
                              )

    print("dataset length", len(dataset))

    # dataset = dp.iter.IterableWrapper(dataset).shuffle(buffer_size=batch_size * 10)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, collate_fn=collate_unbatch, prefetch_factor=1)
    shuffled = ShufflerIterDataPipe(dataloader, buffer_size=100)
    dataloader = DataLoader(shuffled,
                            batch_size=batch_size,
                            num_workers=0,
                            shuffle=True
                            )
    
    print("dataloader length", len(dataloader))

    for i, batch in enumerate(dataloader):
        print(list(zip(batch["track_id"].numpy(), batch["section_nr"].numpy())), end="\t")
        print(batch["mdct"].shape)
        if i > 5:
            break
