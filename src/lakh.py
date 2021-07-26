import os
import json
import h5py
import pickle as pk
from midi import get_midi_info
from utils import LazyDict
from tqdm import tqdm



# Utility functions for retrieving paths
def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

class LakhMIDIDataset(object):
    def __init__(self, data_dir, max_num_songs=None):
        self.data_dir = data_dir
        scores_path = os.path.join(self.data_dir, 'match_scores.json')
        with open(scores_path, 'r') as f:
            scores_dict = json.load(f)

        self.songs = {}
        num_songs = len(scores_dict)
        self.msd_id_to_song_name = {}
        for song_idx, (msd_id, song_score_dict) in tqdm(enumerate(scores_dict.items()), total=num_songs):
            if len(song_score_dict) == 0:
                continue

            if (max_num_songs is not None) and (song_idx >= max_num_songs):
                break

            precompute_path = self._get_precompute_path(msd_id)
            if not os.path.exists(precompute_path):
                print(f"* processing {msd_id}")
                print(f"    + loading metadata")
                h5_path = self.msd_id_to_h5(msd_id)
                with h5py.File(h5_path, 'r') as f:
                    genre = f['metadata']['songs'][()]['genre'][0].decode()
                    title = f['metadata']['songs'][()]['title'][0].decode()
                    artist = f['metadata']['songs'][()]['artist_name'][0].decode()

                # Just get best matching MIDI
                midi_md5 = max(song_score_dict.items(), key=lambda x: x[1])[0]
                midi_path = self.get_midi_path(msd_id, midi_md5)

                song_name = f"{artist} - {title}"
                song_item = {
                    'msd_id': msd_id,
                    'midi_md5': midi_md5,
                    'genre': genre,
                    'title': title,
                    'artist': artist,
                    'name': song_name
                }
                print(f"    + analyzing midi")
                song_item.update(get_midi_info(midi_path))
                os.makedirs(os.path.dirname(precompute_path), exist_ok=True)
                with open(precompute_path, 'wb') as f:
                    pk.dump(song_item, f)
            else:
                print(f"* loading {msd_id}")
                with open(precompute_path, 'rb') as f:
                    song_item = pk.load(f)
                    song_name = f"{song_item['artist']} - {song_item['title']}"
            self.msd_id_to_song_name[msd_id] = song_name
        self.song_name_to_msd_id = {v: k for k, v in self.msd_id_to_song_name.items()}

        self.song_ids = list(sorted(self.song_name_to_msd_id.keys()))
        self.songs = LazyDict(self.song_ids, self.get_song_item)

    def get_song_item(self, song_name):
        msd_id = self.song_name_to_msd_id[song_name]
        precompute_path = self._get_precompute_path(msd_id)
        with open(precompute_path, 'rb') as f:
            song_item = pk.load(f)
        return song_item

    def msd_id_to_h5(self, msd_id):
        """Given an MSD ID, return the path to the corresponding h5"""
        return os.path.join(self.data_dir, 'lmd_matched_h5',
                            msd_id_to_dirs(msd_id) + '.h5')

    def get_midi_path(self, msd_id, midi_md5):
        """Given an MSD ID and MIDI MD5, return path to a MIDI file.
        kind should be one of 'matched' or 'aligned'. """
        return os.path.join(self.data_dir, 'lmd_matched',
                            msd_id_to_dirs(msd_id), midi_md5 + '.mid')

    def _get_precompute_path(self, msd_id):
        """Given an MSD ID and MIDI MD5, return path to a MIDI file.
        kind should be one of 'matched' or 'aligned'. """
        return os.path.join(self.data_dir, 'lmd_urbanorchestra',
                            str(msd_id) + '.pkl')

    def _preprocess_dataset(self):
        notes = None



