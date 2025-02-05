"""
    iterates over the million playlist dataset and outputs info
    about what is in there.
    THIS IS A MODIFICATION OF THE SCRIPT stats.py ORIGINALLY PROVIDED

    Usage:

        python format_rta_input.py path-to-mpd-data output-path
"""
import sys
import json
import re
import collections
import os
import datetime
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import save_npz
import csv
import tqdm
import pickle
from collections import Counter, defaultdict
import argparse
from src.embeddings.model import MatrixFactorizationModel
from src.data_manager.data_manager import DataManager
from sklearn.model_selection import train_test_split
import jsonlines

import ijson


total_playlists = 0
total_tracks = 0
unique_track_count = 0
tracks = set()
artists = set()
albums = set()
titles = set()
total_descriptions = 0
ntitles = set()
n_playlists = 1000000
n_tracks = 2262292
playlist_track = lil_matrix((n_playlists, n_tracks), dtype=np.int32) # to build interaction matrix of binary value
tracks_info = {} # to keep base infos on tracks
title_histogram = collections.Counter()
artist_histogram = collections.Counter()
track_histogram = collections.Counter()
last_modified_histogram = collections.Counter()
num_edits_histogram = collections.Counter()
playlist_length_histogram = collections.Counter()
num_followers_histogram = collections.Counter()
playlists_list = []
quick = False
max_files_for_quick_processing = 2


class CollectiveConfig:

    def __init__(self, name, budget, targetsongs, strategy):
        self.name = name
        self.budget = budget
        self.targetsongs = targetsongs
        self.strategy = strategy
        self.collective_controlled_inds = []

    def __repr__(self):
        return f"Config(name={self.name})"


def load_configs_from_jsonl(file_path, check_valid_song = True, all_songs = None):
    configs = []
    with open(file_path, 'r') as file:
        for line in file:
            config_data = json.loads(line.strip())
            #Validate song info
            if check_valid_song:
                for track_uri in config_data["targetsongs"]:
                    if track_uri not in all_songs:
                        raise ValueError(f"Song {track_uri} not found in the song info. Either the config file has a song that doesn't exist or you need to load extra songs")

            config = CollectiveConfig(
                name=config_data["name"],
                budget=config_data["budget"],
                targetsongs=config_data["targetsongs"],
                strategy=config_data["strategy"]
            )
            configs.append(config)
    return configs


def get_playlist_and_song_from_mpd(raw_path, just_summary_info = False):
    print("processing MPD and gettsings songs")
    count = 0
    filenames = os.listdir(raw_path)
    all_playlists = []
    song_info = {}
    for filename in tqdm.tqdm(sorted(filenames, key=str)):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((raw_path, filename))
            with open(fullpath, 'r') as f:
                objects = ijson.items(f, 'playlists.item')
                
                for playlist in objects:
                    if just_summary_info:
                        temp_info = {"pid": playlist["pid"], 'tracks' : playlist['tracks'], "num_unique_tracks": len(set([t["track_uri"] for t in playlist["tracks"]]))}
                        # Get unique song info
                        for track in playlist["tracks"]:
                            track_uri = track["track_uri"]
                            if track_uri not in song_info:
                                # Copy and del the position info
                                track_meta = dict(track)
                                del track_meta["pos"]
                                song_info[track_uri] = track_meta
                        all_playlists.append(temp_info)
                    else:
                        all_playlists.append(playlist)
                count += 1
        if quick and count > max_files_for_quick_processing:
            break

    # for filename in tqdm.tqdm(sorted(filenames, key=str)):
    #     if filename.startswith("mpd.slice.") and filename.endswith(".json"):
    #         fullpath = os.sep.join((raw_path, filename))
    #         f = open(fullpath)
    #         js = f.read()
    #         f.close()
    #         mpd_slice = json.loads(js)
    #         if just_summary_info:
    #             for playlist in mpd_slice["playlists"]:
    #                 temp_info = {"pid": playlist["pid"], 'tracks' : playlist['tracks'], "num_unique_tracks": len(set([t["track_uri"] for t in playlist["tracks"]]))}
    #                 # Get unique song info
    #                 for track in playlist["tracks"]:
    #                     track_uri = track["track_uri"]
    #                     if track_uri not in song_info:
    #                         # Copy and del the position info
    #                         track_meta = dict(track)
    #                         del track_meta["pos"]
    #                         song_info[track_uri] = track_meta
    #                 all_playlists.append(temp_info)
    #         else:
    #             all_playlists.extend(mpd_slice['playlists'])
    #         count += 1
    #     if quick and count > max_files_for_quick_processing:
    #         break
    return all_playlists, song_info

def process_mpd2(raw_path, out_path, config_path, seed = 0, test_size = 10000, val_size = 10000, extra_song_file = None, modified_playlist_file = 'modified_playlist.json'):
    print("processing MPD")
    # Get the playlist info
    all_playlists_summary, song_info = get_playlist_and_song_from_mpd(raw_path, just_summary_info=True)
    # IF extra song file is provided, load the song info from that file and append to the song info
    new_song_info = {}
    if extra_song_file:
        global n_tracks
        global playlist_track
        with jsonlines.open(extra_song_file, 'r') as file:
            for extra_song_info in file:
                new_song_info[extra_song_info['track_uri']] = extra_song_info
                # song_info[extra_song_info['track_uri']] = extra_song_info
        song_info.update(new_song_info)
        # Update the number of tracks and the playlist track matrix globally
        # n_tracks += len(new_song_info)
        n_tracks = len(song_info)
        playlist_track = lil_matrix((n_playlists, n_tracks), dtype=np.int32) 
    # Get the playlist ids that have more than 20 unique tracks
    candidate_pids_indicies_mapping = get_candidates_indicies(all_playlists_summary, min_count = 20)
    # Load the configs
    if config_path is not None:
        configs = load_configs_from_jsonl(config_path, check_valid_song = True, all_songs = song_info)
        total_budget = sum([c.budget for c in configs])
    else:
        configs = []
    # Get the total budget across all configs
    remaining_possible_inds = list(candidate_pids_indicies_mapping.keys())
    for config in configs:
        (collective_inds, remaining_possible_inds) = train_test_split(remaining_possible_inds, train_size = config.budget, random_state = seed)
        config.collective_controlled_inds = collective_inds

    # The collective's pids can be in the training or the validation set but not the test set. Need to ensure that split is happening properly.     
    # To ensure the candidate pids are not used in test set, we'll mannually split the remaining pids
    full_train_inds, test_inds = train_test_split(remaining_possible_inds, test_size = test_size, random_state = seed)
    # Extend the train_pids with the collective pids
    for config in configs:
        full_train_inds.extend(config.collective_controlled_inds)
    # Now do the train_validation split
    train_inds, val_inds = train_test_split(full_train_inds, test_size = val_size, random_state = seed)
    # Now we have the train, validation and test pids. We now need to actually perform playlist modification for each of the 
    # ids associated with each config. They're modified based on the strategy

    inds_to_keep_track_of = {}
    pids_to_keep_track_of = {}
    collective_pids_mapping = {}
    modified_playlist_mapping = {}
    for config in configs:
        collective_inds = config.collective_controlled_inds
        # Get the pids associated with the collective inds
        collective_pids = [candidate_pids_indicies_mapping[ind] for ind in collective_inds]
        collective_pids_mapping.update({config.name: collective_pids})
        #Update inds to keep track of key is the individuals inds, value is True
        inds_to_keep_track_of.update({ind: True for ind in collective_inds})
        playlist_to_modify = [all_playlists_summary[i] for i in collective_inds]

        track_modification_detail = [song_info[track_id] for track_id in config.targetsongs]

        modified_playlist = modification_wrapper_outer(playlist_to_modify, config.strategy, track_modification_detail)
        modified_playlist_mapping.update(modified_playlist)

    # Now we have the modified playlists, we need to save them in the output path. Then when we stream the playlist again we can use read the main files and then 
    # pertub based on the modified playlist mapping
    with open(os.path.join(out_path, modified_playlist_file), 'w+') as fp:
        json.dump(modified_playlist_mapping, fp, indent=4)

    # Save information about the collective's own pids
    with open(os.path.join(out_path, "collective_pids_mapping.json"), 'w+') as fp:
        json.dump(collective_pids_mapping, fp, indent=4)

    # Save the train/test/va split
    np.save(os.path.join(out_path, 'dataset_split', 'train_indices.npy'), train_inds)
    np.save(os.path.join(out_path, 'dataset_split', 'val_indices.npy'), val_inds)
    np.save(os.path.join(out_path, 'dataset_split', 'test_indices.npy'), test_inds)
    # np.save('%s/train_inds.npy' % out_path, train_inds)
    # np.save('%s/val_inds.npy' % out_path, val_inds)
    # np.save('%s/test_inds.npy' % out_path, test_inds)

def process_mpd_with_modification(raw_path, out_path, modification_dict):
    print("processing MPD")
    global playlists_list
    count = 0
    filenames = os.listdir(raw_path)
    for filename in tqdm.tqdm(sorted(filenames, key=str)):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            playlists_list = []
            fullpath = os.sep.join((raw_path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            process_info(mpd_slice["info"])
            for playlist in mpd_slice["playlists"]:
                pid = str(playlist['pid'])
                if pid in modification_dict:
                    #Modification dict has a different set of tracks. Set them in this playlist 
                    playlist['tracks'] = modification_dict[pid]
                process_playlist(playlist)
            count += 1
            seqfile = open('%s/playlists_seq.csv' % out_path, 'a', newline ='')
            with seqfile:
              write = csv.writer(seqfile) 
              write.writerows(playlists_list)  

            if quick and count > max_files_for_quick_processing:
                break


def modification_wrapper_outer(group_playlist, strategy, to_insert):
    if strategy == "random":
        return strategy_random(to_insert, group_playlist)
    elif strategy == "dirlof_basic":
        return strategy_dirlof_basic(to_insert, group_playlist)
    elif strategy == "inclust":
        return strategy_inclust(to_insert, group_playlist)

    else:
        raise ValueError(f"Unknown strategy {strategy}")


def strategy_random(inserted_track, group_controlled_playlist):
    # Randomly insert the song in the playlist
    updated_playlist = []
    for playlist in group_controlled_playlist:
        random_index = np.random.randint(0, len(playlist))
        modified_playlist = insert_songs_in_playlist(playlist, random_index, inserted_track, insertbefore = True)
        updated_playlist.append(modified_playlist)
    return updated_playlist

def strategy_dirlof_basic(inserted_track, group_controlled_playlist):
    #Insert after the least common song in the group controlled playlist
    track_list = [playlist['tracks'] for playlist in group_controlled_playlist]
    # This is a list of lists. We need to flatten it
    track_list = [item for sublist in track_list for item in sublist]
    song_counter = Counter([t["track_uri"] for t in track_list])

    updated_playlist = {}

    for playlist_info in group_controlled_playlist:
        playlist_tracks = playlist_info['tracks']
        # Just get the track_uri
        playlist_tracks = [t['track_uri'] for t in playlist_tracks]                           
        # For each playlist, pick a song that is most common in the group controlled playlist
        # frequency  counter for each song in this single playlist
        freqs = [(t, song_counter[t]) for t in playlist_tracks]
        # Find the max frequency song in this playlist
        min_freq = min(freqs, key = lambda x: x[1])[1]
        # Filter out the songs that have the max frequency
        candidates = [t[0] for t in freqs if t[1] == min_freq]  
        # Pick one of the most common songs in the playlist
        # Choose one uniformly at random
        song_anchor = np.random.choice(candidates)

        song_anchor_index = playlist_tracks.index(song_anchor)
        modified_playlist = insert_songs_in_playlist(playlist_info['tracks'], song_anchor_index, inserted_track, insertbefore = False)
        updated_playlist[playlist_info['pid']] = modified_playlist
    return updated_playlist

def strategy_inclust(inserted_track, group_controlled_playlist):
    # Compute the occurance of songs in the controlled playlist with a counter

    # Group controlled playlist is a list where each entry is a dict. Each of these dicts has a key 'tracks' which is a dict of the tracks in the playlist with the key 
    # 'track_uri identifiying the song. We need to extract the track_uri and then use that to compute the counter
    track_list = [playlist['tracks'] for playlist in group_controlled_playlist]
    # This is a list of lists. We need to flatten it
    track_list = [item for sublist in track_list for item in sublist]
    song_counter = Counter([t["track_uri"] for t in track_list])
    updated_playlist = {}
    for playlist_info in group_controlled_playlist:
        playlist_tracks = playlist_info['tracks']
        # Just get the track_uri
        playlist_tracks = [t['track_uri'] for t in playlist_tracks]                           
        # For each playlist, pick a song that is most common in the group controlled playlist
        # frequency  counter for each song in this single playlist
        freqs = [(t, song_counter[t]) for t in playlist_tracks]
        # Find the max frequency song in this playlist
        max_freq = max(freqs, key = lambda x: x[1])[1]
        # Filter out the songs that have the max frequency
        candidates = [t[0] for t in freqs if t[1] == max_freq]  
        # Pick one of the most common songs in the playlist
        # Choose one uniformly at random
        song_anchor = np.random.choice(candidates)
        song_anchor_index = playlist_tracks.index(song_anchor)

        modified_playlist = insert_songs_in_playlist(playlist_info['tracks'], song_anchor_index, inserted_track, insertbefore = True)
        #group_controlled_playlist = insert_songs_in_playlist(group_controlled_playlist, song_anchor_index, [inserted_track], insertbefore = True)
        #playlist.append(song)
        #song_counter[song] += 1
        updated_playlist[playlist_info['pid']] = modified_playlist
    return updated_playlist


def select_one_of_most_freq_songs(candidates, frequencies, n, rng):
    #Taken directly from the original code

    candidate_occurences = [(t, frequencies[t]) for t in candidates]
    #print("test max freq: ", max(frequencies.values()))
    # get the most famous song among all those present in this playlist
    most_famous_tracks = [track for (track, count) in heapq.nlargest(n, candidate_occurences, key=lambda x: x[1])]
    #print(f"most_famous_tracks: {most_famous_tracks}")
    sample_of_most_famous_tracks = rng.choice(most_famous_tracks)
    #print(f"sample_of_most_famous_tracks: {sample_of_most_famous_tracks}, with frequency {frequencies[sample_of_most_famous_tracks]}")
    return sample_of_most_famous_tracks


def pick_one_of_most_common(song_counter, based_on_list):
    most_common = song_counter.most_common(1)[0][0]
    most_common_elements = [key for key, value in song_counter.items() if value == song_counter[most_common]]

    # Pick one of the most common elements at random
    random_element = np.random.choice(most_common_elements)
    return(random_element)

def get_candidates_indicies(all_playlists_summary, min_count = 20):
    candidate_indicies = {i : p['pid']  for (i, p) in enumerate(all_playlists_summary) if p["num_unique_tracks"] >= min_count}
    #candidate_playlistpids = [p["pid"] for p in all_playlists_summary if p["num_unique_tracks"] >= min_count]
    return candidate_indicies



def insert_songs_in_playlist(playlist, index, songs, insertbefore):
    # insert the songs in reverse order, so that the indices do not change
    for i, song in enumerate(reversed(songs)):
        if insertbefore:
            # insert signal before seed song
            playlist.insert(index-1, song)
            #print(f"inserted {song} BEFORE, i.e., at index {index-1}")
        else:
            # insert signal after seed song
            playlist.insert(index, song)
            #print(f"inserted {song} AFTER, i.e., at index {index}")

    return playlist




def process_mpd(raw_path, out_path):
    print("processing MPD")
    global playlists_list
    count = 0
    filenames = os.listdir(raw_path)
    for filename in tqdm.tqdm(sorted(filenames, key=str)):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            playlists_list = []
            fullpath = os.sep.join((raw_path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            process_info(mpd_slice["info"])
            for playlist in mpd_slice["playlists"]:
                process_playlist(playlist)
            count += 1
            seqfile = open('%s/playlists_seq.csv' % out_path, 'a', newline ='')
            with seqfile:
              write = csv.writer(seqfile) 
              write.writerows(playlists_list)  

            if quick and count > max_files_for_quick_processing:
                break

    show_summary()


def show_summary():
    print()
    print("number of playlists", total_playlists)
    print("number of tracks", total_tracks)
    print("number of unique tracks", len(tracks))
    print("number of unique albums", len(albums))
    print("number of unique artists", len(artists))
    print("number of unique titles", len(titles))
    print("number of playlists with descriptions", total_descriptions)
    print("number of unique normalized titles", len(ntitles))
    print("avg playlist length", float(total_tracks) / total_playlists)
    print()
    print("top playlist titles")
    for title, count in title_histogram.most_common(20):
        print("%7d %s" % (count, title))

    print()
    print("top tracks")
    for track, count in track_histogram.most_common(20):
        print("%7d %s" % (count, track))

    print()
    print("top artists")
    for artist, count in artist_histogram.most_common(20):
        print("%7d %s" % (count, artist))

    print()
    print("numedits histogram")
    for num_edits, count in num_edits_histogram.most_common(20):
        print("%7d %d" % (count, num_edits))

    print()
    print("last modified histogram")
    for ts, count in last_modified_histogram.most_common(20):
        print("%7d %s" % (count, to_date(ts)))

    print()
    print("playlist length histogram")
    for length, count in playlist_length_histogram.most_common(20):
        print("%7d %d" % (count, length))

    print()
    print("num followers histogram")
    for followers, count in num_followers_histogram.most_common(20):
        print("%7d %d" % (count, followers))


def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def to_date(epoch):
    return datetime.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d")


def process_playlist(playlist):
    global total_playlists, total_tracks, total_descriptions, unique_track_count, playlists_list

    total_playlists += 1
    # print playlist['playlist_id'], playlist['name']

    if "description" in playlist:
        total_descriptions += 1

    titles.add(playlist["name"])
    nname = normalize_name(playlist["name"])
    ntitles.add(nname)
    title_histogram[nname] += 1

    playlist_length_histogram[playlist["num_tracks"]] += 1
    last_modified_histogram[playlist["modified_at"]] += 1
    num_edits_histogram[playlist["num_edits"]] += 1
    num_followers_histogram[playlist["num_followers"]] += 1
    playlist_id = playlist["pid"]
    playlist_track_count = 0
    playlist_seq = []
    for track in playlist["tracks"]:
      full_name = track["track_uri"].lstrip("spotify:track:")
      if full_name not in tracks_info :
        if "pos" in track:
            del track["pos"]
        tracks_info[full_name] = track
        unique_track_count += 1
        tracks_info[full_name]["id"] = unique_track_count - 1
        tracks_info[full_name]["count"] = 1
      elif playlist_track[playlist_id, tracks_info[full_name]["id"]] != 0 :
        # remove tracks that are already earlier in the playlist
        continue
      else :
        tracks_info[full_name]["count"] += 1
      total_tracks += 1
      albums.add(track["album_uri"])
      tracks.add(track["track_uri"])
      artists.add(track["artist_uri"])
      artist_histogram[track["artist_name"]] += 1
      track_histogram[full_name] += 1
      track_id = tracks_info[full_name]["id"]
      playlist_track_count += 1
      playlist_track[playlist_id, track_id] = playlist_track_count
      playlist_seq.append(str(track_id))
    playlists_list.append(playlist_seq)


def process_info(_):
    pass

def process_album_artist( tracks_info, out_path):
    artist_songs = defaultdict(list)  # a dict where keys are artist ids and values are list of corresponding songs
    album_songs = defaultdict(list)  # a dict where keys are album ids and values are list of corresponding songs
    song_album = np.zeros(n_tracks)  # a 1-D array where the index is the track id and the value is the album id
    song_artist = np.zeros(n_tracks)  # a 1-D array where the index is the track id and the value is the artist id
    album_ids = {}  # a dict where keys are album names and values are album ids
    artist_ids = {}  # a dict where keys are artist names and values are artist ids
    album_names = []  # a list where indices are album ids and values are album names
    artist_names = []  # a list where indices are artist ids and values are album names
    print("Processing albums and artists.")
    for d in tqdm.tqdm(tracks_info.values()):
        album_name = "%s by %s" % (d['album_name'], d['artist_name'])
        artist_name = d['artist_name']
        if album_name not in album_ids:
            album_id = len(album_names)
            album_ids[album_name] = album_id
            album_names.append(album_name)
        else:
            album_id = album_ids[album_name]
        song_album[d['id']] = album_id

        if artist_name not in artist_ids:
            artist_id = len(artist_names)
            artist_ids[artist_name] = artist_id
            artist_names.append(artist_name)
        else:
            artist_id = artist_ids[artist_name]
        song_artist[d['id']] = artist_id
        album_songs[album_id].append(d['id'])
        artist_songs[artist_id].append(d['id'])

    np.save(os.path.join(out_path, 'song_album'), song_album)
    np.save(os.path.join(out_path, 'song_artist'), song_artist)

    with open(os.path.join(out_path, 'album_ids.pkl'), 'wb+') as f:
        pickle.dump(album_ids, f)

    with open(os.path.join(out_path, 'artist_ids.pkl'), 'wb+') as f:
        pickle.dump(artist_ids, f)

    with open(os.path.join(out_path, 'artist_songs.pkl'), 'wb+') as f:
        pickle.dump(artist_songs, f)

    with open(os.path.join(out_path, 'album_songs.pkl'), 'wb+') as f:
        pickle.dump(album_songs, f)

    with open(os.path.join(out_path, 'artist_names.pkl'), 'wb+') as f:
        pickle.dump(artist_names, f)

    with open(os.path.join(out_path, 'album_names.pkl'), 'wb+') as f:
        pickle.dump(album_names, f)
        
    return

def create_initial_embeddings(data_manager, out_path):
    print("Creating initial song embeddings")
    mf_model = MatrixFactorizationModel(data_manager, foldername = os.path.join(out_path, 'embeddings'), retrain=True, emb_size=128)
    return

def create_side_embeddings(data_manager):
    buckets_dur = data_manager.get_duration_bucket(data_manager.song_duration)
    buckets_pop = data_manager.get_pop_bucket(data_manager.song_pop)
    buckets_dur_dict = {i: [] for i in range(40)}
    buckets_pop_dict = {i: [] for i in range(100)}
    print("Creating duration buckets")
    for ind, b in enumerate(buckets_dur):
        buckets_dur_dict[b].append(ind)
    print("Creating popularity buckets")
    for ind, b in enumerate(buckets_pop):
        buckets_pop_dict[b].append(ind)

    print([len(v) for k,v in buckets_pop_dict.items()])
    # Create metadata initial embedding
    song_embeddings = np.load(data_manager.song_embeddings_path)
    print("Creating album embeddings")
    alb_embeddings = np.asarray([song_embeddings[data_manager.album_songs[i]].mean(axis=0) for i in tqdm.tqdm(range(len(data_manager.album_songs)))])
    print("Creating artist embeddings")

    art_embeddings = np.asarray([song_embeddings[data_manager.artist_songs[i]].mean(axis=0) for i in tqdm.tqdm(range(len(data_manager.artist_songs)))])

    pop_embeddings = np.asarray(
        [song_embeddings[buckets_pop_dict[i]].mean(axis=0) for i in tqdm.tqdm(range(len(buckets_pop_dict)))])
    pop_embeddings[np.isnan(pop_embeddings)] = 0

    dur_embeddings = np.asarray(
        [song_embeddings[buckets_dur_dict[i]].mean(axis=0) for i in tqdm.tqdm(range(len(buckets_dur_dict)))])

    np.save(data_manager.album_embeddings_path, alb_embeddings)
    np.save(data_manager.artist_embeddings_path, art_embeddings)
    np.save(data_manager.pop_embeddings_path, pop_embeddings)
    np.save(data_manager.dur_embeddings_path, dur_embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpd_path", type=str, required=False, default="../MPD/data",
                             help = "Path to MPD")
    parser.add_argument("--out_path", type=str, required=False, default="resources/data/rta_input",
                             help = "Path to rta input")
    parser.add_argument("--config_path", type=str, required=False, default=None,
                             help = "Path to config file")
    parser.add_argument("--extra_song_file", type=str, required=False, default=None, help = 'Fake songs to add for the collective')
    parser.add_argument("--test_size", type=int, required=False, default=10000, help = "Size of the test set")
    parser.add_argument("--val_size", type=int, required=False, default=10000, help = "Size of the validation set")
    parser.add_argument("--modified_playlist_file", type = str, required = False, default = 'modified_playlist.json')
    '''MAKE SURE THAT THE extra songs are actually all used in the collective's config in order to ensure proper counting stats'''
    
    args = parser.parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    rta_input_path  = os.path.join(args.out_path, 'rta_input')
    os.makedirs(rta_input_path, exist_ok=True)
    os.makedirs(os.path.join(args.out_path, 'dataset_split'), exist_ok=True)
    # import pdb
    # pdb.set_trace()
    #os.makedirs("resources/models", exist_ok=True)
    #process_mpd(args.mpd_path, args.out_path)
    process_mpd2(args.mpd_path, args.out_path, args.config_path, extra_song_file = args.extra_song_file, test_size = args.test_size, val_size = args.val_size, modified_playlist_file = args.modified_playlist_file)
    # Now reprocess with the modifications
    with open(os.path.join(args.out_path, args.modified_playlist_file), 'r') as fp:
        modification_dict = json.load(fp)
    process_mpd_with_modification(args.mpd_path, args.out_path, modification_dict)

    save_npz('%s/playlist_track.npz' % rta_input_path, playlist_track.tocsr(False))
    with open('%s/tracks_info.json' % rta_input_path, 'w') as fp:
      json.dump(tracks_info, fp, indent=4)
    process_album_artist(tracks_info, rta_input_path)
    # global n_tracks
    # global n_playlists
    #n_tracks = None, n_playlists = None
    data_manager = DataManager(foldername=args.out_path, n_tracks = n_tracks, n_playlists = n_playlists)

    # Manuall set the train/test/validation indicies form the files
    # train_inds = np.load('%s/train_inds.npy' % args.out_path)
    # val_inds = np.load('%s/val_inds.npy' % args.out_path)
    # test_inds = np.load('%s/test_inds.npy' % args.out_path)
    # data_manager.train_set = train_inds
    # data_manager.val_set = val_inds
    # data_manager.test_set = test_inds

    some_metadata = {'n_tracks' : n_tracks, 'n_playlists' : n_playlists}
    with open(os.path.join(args.out_path, 'metainfo.json'), "w+") as fp:
        json.dump(some_metadata, fp, indent = 4)



    print(data_manager.binary_train_set)
    create_initial_embeddings(data_manager, out_path = args.out_path)
    create_side_embeddings(data_manager)


