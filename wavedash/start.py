'''
Experiments with Super Smash Bro Melee Slippi Files
'''
from pathlib import Path
from slippi import Game as SlippiGame
from tqdm import tqdm
from collections import defaultdict
import pickle

from wavedash.constants import SLIPPI_FILES_ROOT, CACHED_GAMES


class Game(SlippiGame):
    '''
    Extension of Slippi Game with extra helpful functionality
    '''
    def __init__(self, file):
        super().__init__(file)
        self.path = file

    @property
    def combatants(self):
        '''
        Returns in order the name of the combatants in order
        '''
        combatants = []
        for player in self.start.players:
            if player is not None:
                combatants.append(player.character.name)

        combatants.sort()
        return combatants


def get_slippi_files():
    '''
    Returns all slippi files
    '''
    slippi_files = Path(SLIPPI_FILES_ROOT).rglob('*.slp')

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


@persist_to_file(CACHED_GAMES)
def get_slippi_games():
    '''
    Returns all slippi files in SLIPPI_FILES_ROOT as Slippi Games
    '''
    print('Converting them to Games')
    games = []
    for game_file in tqdm(yield_slippi_files(), desc='Extracting games'):
        game = Game(game_file)
        if game.combatants == ['FOX', 'MARTH']:
            games.append(game)
            print('Got {}'.format(len(games)))
            if len(games) > 2:
                print('Done')
                break

    return games


def get_competition_count():
    '''
    Scores who is playing in each game
    '''
    game_count = defaultdict(int)
    games = get_slippi_games()
    for game in games:
        game_count[str(game.combatants)] += 1

    return game_count


if __name__ == '__main__':
    games = get_slippi_games()
    for game in games:
        print(game.path)
    print('done')
