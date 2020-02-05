from pathlib import Path
from tqdm import tqdm
import pickle

from wavedash.constants import (
    SLIPPI_REPLAYS,
    CACHED_GAMES,
    FOX_MARTH_MATCHUPS
)
from wavedash.game import Game


def get_slippi_files():
    '''
    Returns all slippi files
    '''
    slippi_files = Path(SLIPPI_REPLAYS).rglob('*.slp')

    return list(slippi_files)[:2000]


def yield_slippi_files():
    '''
    Only return games that match the description
    '''
    slippi_files = get_slippi_files()
    for file in slippi_files:
        yield file


def persist_to_file(file_name):
    '''
    Persists the return value of a function to a file
    '''
    def decorator(original_func):
        response = None
        try:
            with open(file_name, 'rb') as cached_file:
                print('Using cache')
                response = pickle.load(cached_file)

        except (IOError, ValueError):
            pass

        def new_func():
            nonlocal response
            if response:
                return response

            print('Creating cache')
            response = original_func()

            # cache the words
            with open(file_name, 'wb') as cached_file:
                pickle.dump(response, cached_file)

            return response

        return new_func

    return decorator


# @persist_to_file(CACHED_GAMES)
def get_fox_marth_games():
    '''
    Returns all slippi files in SLIPPI_REPLAYS as Slippi Games
    '''
    games = []
    for game_file in tqdm(FOX_MARTH_MATCHUPS,
                          desc='Retrieving fox-marth matchups'):
        game = Game(game_file)

        # make sure the slippi replay has a corresponding video
        if not game.video_path:
            print('No matching video for {}'.format(game_file))
            continue

        games.append(game)

    return games
