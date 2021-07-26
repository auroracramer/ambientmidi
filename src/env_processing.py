import numpy as np
import librosa
import random
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from features import get_feature_dict


MIN_CLIP_DURATION = 0.125
MAX_CLIP_DURATION = 1.0 # seconds
HOP_LENGTH = 128
WIN_LENGTH = 512
MIN_NUM_INSTRS = 5
MAX_NUM_INSTRS = 20
TONALITY_THRESHOLD = 0.5


def extract_events(audio, sr):
    print(f"* detecting onsets")
    # Get onsets
    onset_idxs = librosa.onset.onset_detect(y=audio,
                                            sr=sr,
                                            hop_length=128,
                                            units='samples',
                                            backtrack=True,
                                            delta=0.07)

    audio_length = audio.shape[0]
    min_clip_length = int(MIN_CLIP_DURATION * sr)
    max_clip_length = int(MAX_CLIP_DURATION * sr)

    print(f"* computing detected event features for {len(onset_idxs)} events")
    clip_list = []
    for onset_idx in onset_idxs:
        end_idx = min(onset_idx + max_clip_length, audio_length)
        clip = audio[onset_idx:end_idx]
        rms = librosa.feature.rms(clip, frame_length=WIN_LENGTH, hop_length=HOP_LENGTH).flatten()
        thresh = rms.max() * 0.01
        silent_frame_idxs = [idx for idx in np.nonzero(rms <= thresh)[0]
                             if idx >= min_clip_length]
        # Try to get first offset
        if len(silent_frame_idxs) > 0:
            end_idx = librosa.frames_to_samples(silent_frame_idxs[0] + 1,
                                                hop_length=HOP_LENGTH,
                                                n_fft=WIN_LENGTH)
            clip = audio[onset_idx:end_idx]

        clip_dict = get_feature_dict(clip, sr)
        clip_list.append(clip_dict)
    return clip_list


def get_kmeans(event_clip_list, n_clusters, feature_key):
    chunk_size = 64
    num_clips = len(event_clip_list)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters)
    start_idx = 0
    while start_idx < num_clips:
        end_idx = min(start_idx + chunk_size, num_clips)
        nsynth_arr = np.array([x[feature_key]
                               for x in event_clip_list[start_idx:end_idx]])
        kmeans.partial_fit(X=nsynth_arr)
        start_idx += chunk_size
    return kmeans


def cluster_events(event_clip_list, n_clusters=None):
    feature_key = 'mfcc'
    X = np.array([x[feature_key] for x in event_clip_list])
    if n_clusters is None:
        best_cluster_ids = None
        best_n_clusters = None
        best_score = -1

        if len(event_clip_list) < MAX_NUM_INSTRS:
            n_cluster_list = range(1, len(event_clip_list))
        else:
            n_cluster_list = range(MIN_NUM_INSTRS, MAX_NUM_INSTRS + 1)

        for n_clusters in n_cluster_list:
            if not (n_clusters < (len(event_clip_list) - 1)):
                continue
            print(f"* trying {n_clusters} clusters")
            kmeans = get_kmeans(event_clip_list, n_clusters, feature_key)
            cluster_ids = kmeans.predict(X=X)
            try:
                score = silhouette_score(X, labels=cluster_ids)
            except ValueError:
                continue
            print(f"      + silhouette: {score}")
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_cluster_ids = cluster_ids
        n_clusters = best_n_clusters
        cluster_ids = best_cluster_ids
        print(f"* using {n_clusters} environmental clusters")
    else:
        print(f"* getting {n_clusters} environmental clusters")
        kmeans = get_kmeans(event_clip_list, n_clusters, feature_key)
        cluster_ids = kmeans.predict(X=X)
    cluster_to_events = {}
    for cluster_id, event in zip(cluster_ids, event_clip_list):
        if cluster_id not in cluster_to_events:
            cluster_to_events[cluster_id] = {'events': []}
        cluster_to_events[cluster_id]['events'].append(event)
    for cluster_id, cluster_item in cluster_to_events.items():
        print(f"* computing features for environmental cluster {cluster_id+1}/{n_clusters}")
        X_mfcc = np.array([x['mfcc'] for x in cluster_item['events']])
        X_tonality = np.array([x['tonality'] for x in cluster_item['events']])
        cluster_item['mean_mfcc'] = X_mfcc.mean()
        cluster_item['mean_tonality'] = X_tonality.mean()
        cluster_item['is_drum'] = cluster_item['mean_tonality'] <= TONALITY_THRESHOLD
        cluster_item['num_events'] = len(cluster_item['events'])
        #cluster_item['medoid_mfcc'] = medoid(X_mfcc)
        #cluster_item['medoid_tonality'] = medoid(X_tonality)
    return cluster_to_events


def medoid(X):
    return min(list(X), key=lambda x: ((X - x[np.newaxis, :]) ** 2).sum())


def extract_clusters_to_events(input_audio, input_sr):
    print("Extracting environmental events")
    event_clip_list = extract_events(input_audio, input_sr)
    print("Clustering environmental events")
    return cluster_events(event_clip_list, n_clusters=None)
