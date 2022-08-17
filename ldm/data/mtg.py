import math
import os
from pathlib import Path
from typing import Callable, TypedDict, List, Dict

import pandas as pd
import ray
import torch
import torchaudio
from einops import rearrange
from mdct import mdct
from torch.utils.data import Dataset

from ldm.data.ray_util import ProgressBar

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


def get_audio_length(audio_path, sampling_rate: int) -> int:
    metadata = torchaudio.info(audio_path)
    frames = metadata.num_frames
    if metadata.sample_rate != sampling_rate:
        frames = math.ceil(frames * sampling_rate / metadata.sample_rate)
    return int(frames)


@ray.remote
def get_number_sections(track_path: Path, sampling_rate: int, size: int, step: int, pba) -> int:
    audio_length = get_audio_length(track_path, sampling_rate)
    audio_mdct_length = mdct_length(audio_length, n_fft=int(size * 2))
    n_sections = int((audio_mdct_length - size) / step + 1)
    pba.update.remote(1)
    return n_sections


def aggregate_sections(dataset: MTGBase, size: int, step: int) -> List[Section]:
    pb = ProgressBar(total=len(dataset), description="Aggregating sections", min_interval=5)
    actor = pb.actor

    track_sections_refs = [
        get_number_sections.remote(
            dataset.get_audio_path(track_id),
            dataset.sampling_rate,
            size,
            step,
            actor
        )
        for track_id in dataset.track_ids]
    pb.print_until_done()
    track_sections = ray.get(track_sections_refs)

    sections = []
    for track_id, n_sections in zip(dataset.track_ids, track_sections):
        track_info = dataset.tracks[track_id]
        for i in range(n_sections):
            sections.append(Section(track_id=track_id, section_nr=i, **track_info))
    return sections


class MtgMdct(Dataset):
    def __init__(self,
                 split: str,
                 sampling_rate=22050,
                 size=256,
                 **kwargs
                 ):
        super(MtgMdct).__init__()
        self.split = split
        self.size = size
        self.step = size
        self.sampling_rate = sampling_rate
        self.mtg_base = MTGFullAudio(split=split, sampling_rate=sampling_rate, **kwargs)

        try:
            self.sections = self.load_sections()
        except FileNotFoundError:
            self.sections: List[Section] = aggregate_sections(self.mtg_base, self.size, self.step)
            self.save_sections(self.sections)

        print(f"Dataset contains {len(self.sections)} sections")

    def sections_path(self) -> Path:
        return self.mtg_base.data_root / "sections" / self.split / f"{self.size}-{self.sampling_rate}.feather"

    def load_sections(self) -> List[Section]:
        path = self.sections_path()
        if path.exists():
            print(f"Loading sections from {path}")
            df = pd.read_feather(path)
            return df.to_dict("records")
        else:
            raise FileNotFoundError(f"{path} not found")

    def save_sections(self, sections):
        path = self.sections_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(sections)
        print(f"Saving sections to {path}")
        df.to_feather(path)

    def get_mdct_path(self, track_id: int) -> Path:
        return Path(
            self.mtg_base.data_root) / "mdct" / f"{self.size}-{self.sampling_rate}Hz" / f"{track_id}.pt"

    def save_mdct(self, track_id, tensor):
        mdct_path = self.get_mdct_path(track_id)
        mdct_path.parent.mkdir(parents=True, exist_ok=True)
        with mdct_path.open("wb") as f:
            torch.save(tensor, f)

    def load_mdct(self, track_id, section_nr):
        mdct_path = self.get_mdct_path(track_id)
        if mdct_path.exists():
            return torch.load(mdct_path)[section_nr].clone()
        else:
            raise FileNotFoundError(f"{mdct_path=} not found")

    def generate_track_mdct(self, track_id):
        """
        Returns a list of tensor views of the mdcts for the given track_id.
        """
        audio = self.mtg_base.load_audio(self.mtg_base.get_audio_path(track_id))
        audio_mdct = mdct(audio, framelength=self.size * 2)
        audio_mdct = torch.from_numpy(audio_mdct).float()
        audio_mdcts = audio_mdct.unfold(1, self.size, self.size)
        audio_mdcts = rearrange(audio_mdcts, "f u t -> u f t")
        return audio_mdcts

    def __len__(self) -> int:
        return len(self.sections)

    def __getitem__(self, i):
        section = self.sections[i]
        track_id = section["track_id"]
        section_nr = section["section_nr"]

        try:
            section_mdct = self.load_mdct(track_id, section_nr)
        except (FileNotFoundError, EOFError):
            track_mdct = self.generate_track_mdct(track_id)
            self.save_mdct(track_id, track_mdct)
            section_mdct = track_mdct[section_nr]

        return {
            "track_id": track_id,
            "section_nr": section_nr,
            "mdct": section_mdct
        }


# main
if __name__ == "__main__":
    ray.init()
    test_sampling_rate = 22050
    test_size = 256

    dataset = MtgMdct(split="valid",
                      # genres=["classical"],
                      sampling_rate=test_sampling_rate,
                      size=test_size
                      )
