import numpy as np
import sklearn
from dppy.finite_dpps import FiniteDPP
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


def medoid(X):
    return min(list(X), key=lambda x: ((X - x[np.newaxis, :]) ** 2).sum())


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


def find_best_kmeans_by_silihoutte(
    X_feat, event_clip_list, n_clusters=None, cluster_feature_key="mfcc",
    min_num_instrs=5, max_num_instrs=20, verbose=False,
):
    best_cluster_ids = None
    best_n_clusters = None
    best_score = -1

    # If we have less clips than the maximum number of instruments,
    # just consider number of clusters up to number of clips
    if len(event_clip_list) < max_num_instrs:
        n_cluster_list = range(1, len(event_clip_list))
    else:
        n_cluster_list = range(min_num_instrs, max_num_instrs + 1)

    for n_clusters in n_cluster_list:
        if not (n_clusters < (len(event_clip_list) - 1)):
            continue
        if verbose:
            print(f"* trying {n_clusters} clusters")
        kmeans = get_kmeans(event_clip_list, n_clusters, cluster_feature_key)
        cluster_ids = kmeans.predict(X=X_feat)
        try:
            score = silhouette_score(X_feat, labels=cluster_ids)
        except ValueError:
            continue
        if verbose:
            print(f"      + silhouette: {score}")
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
            best_cluster_ids = cluster_ids
    return best_n_clusters, best_cluster_ids


def get_clip_clusters(
    event_clip_list, n_clusters=None, cluster_feature_key="mfcc", tonality_threshold=0.5, max_cluster_size=None,
):

    # Build feature matrix
    X = np.array([x[cluster_feature_key] for x in event_clip_list])

    if n_clusters is None:
        n_clusters, cluster_ids = find_best_kmeans_by_silihoutte()
        print(f"* using {n_cvlusters} environmental clusters")
    else:
        print(f"* getting {n_clusters} environmental clusters")
        kmeans = get_kmeans(event_clip_list, n_clusters, cluster_feature_key)
        cluster_ids = kmeans.predict(X=X)
        
    # Group events by cluster
    clusters_to_events = {}
    for cluster_id, event in zip(cluster_ids, event_clip_list):
        if cluster_id not in clusters_to_events:
            clusters_to_events[cluster_id] = {'events': []}
        clusters_to_events[cluster_id]['events'].append(event)
    cluster_ids = list(clusters_to_events.keys())

    # Compute cluster features
    for cluster_id, cluster_item in clusters_to_events.items():
        print(f"* computing features for environmental cluster {cluster_id+1}/{n_clusters}")
        X_mfcc = np.array([x['mfcc'] for x in cluster_item['events']])
        X_tonality = np.array([x['tonality'] for x in cluster_item['events']])
        cluster_item['mean_mfcc'] = X_mfcc.mean()
        cluster_item['mean_tonality'] = X_tonality.mean()
        cluster_item['is_drum'] = cluster_item['mean_tonality'] <= tonality_threshold
        cluster_item['num_events'] = len(cluster_item['events'])
        #cluster_item['medoid_mfcc'] = medoid(X_mfcc)
        #cluster_item['medoid_tonality'] = medoid(X_tonality)

    if max_cluster_size:
        reduce_large_clusters(clusters_to_events, max_cluster_size)


def reduce_large_clusters(env_clusters_to_events, max_events=50):
    # Within each cluster, reduce the set of examples to at most max_events.
    # If we have to pick a subset, use k-DPP sampling to get diverse set
    reduced_env_clusters_to_events = {}
    for cluster_id, cluster_item in env_clusters_to_events.items():
        if len(cluster_item['events']) >= max_events:
            X_mfcc = np.array([event['mfcc'] for event in cluster_item['events']])
            D_euc = sklearn.metrics.pairwise.euclidean_distances(X_mfcc)
            D_euc = D_euc / np.max(D_euc)
            L_euc = 1 - D_euc
            #K_euc = np.dot(L_euc, np.linalg.inv(np.eye(L_euc.shape[0]) - L_euc))
            dpp = FiniteDPP('likelihood', L=L_euc)
            subset_idxs = dpp.sample_exact_k_dpp(size=max_events)
            reduced_env_clusters_to_events[cluster_id] = dict(cluster_item)
            reduced_env_clusters_to_events[cluster_id]['events'] = [
                cluster_item['events'][event_idx] for event_idx in subset_idxs
            ]
        else:
            reduced_env_clusters_to_events[cluster_id] = dict(cluster_item)
    return reduced_env_clusters_to_events