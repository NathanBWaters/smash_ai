'''
Object wrapping the Slippi replay and recording
'''
import cv2
import glob
import os
import subprocess
from slippi import Game as SlippiGame
import numpy as np

from wavedash.constants import (
    CHARACTERS,
    SLIPPI_REPLAYS,
    SLIPPI_RECORDINGS,
)


def mse_image_similarity(image_a, image_b):
    '''
    Determines the similarity between two images
    '''
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


class Game(object):
    '''
    Extension of Slippi Game with extra helpful functionality
    '''
    def __init__(self, file, video_path=None, init_data=True):
        self.slippi_path = file
        self.slippi_game = SlippiGame(self.slippi_path)
        self.game_name = self.slippi_path.split('\\')[-1].split('.slp')[0]
        self._video_path = video_path
        self._video_frames = []
        self._frame_dir = None

        # set up the Game and extract its information
        if init_data:
            self.initialize_data()

    def initialize_data(self):
        '''
        Set up the data for training
        '''
        self.extract_video()

    @property
    def combatants(self):
        '''
        Returns in order the name of the combatants in order
        '''
        combatants = []
        for player in self.slippi_game.start.players:
            if player is not None:
                combatants.append(player.character.name)

        combatants.sort()
        return combatants

    def char_label(self):
        '''
        Converts the combatants into a label for the char model
        '''
        label = []
        for player in self.slippi_game.start.players:
            if player is None:
                char_value = CHARACTERS.NO_PLAYER.value
            else:
                char_value = CHARACTERS[player.character.name].value

            one_hot_encoded = [0] * len(CHARACTERS)
            one_hot_encoded[char_value] = 1
            label.extend(one_hot_encoded)

        return np.array(label, dtype=np.float32)

    @property
    def video_path(self):
        '''
        Returns the path to the video
        '''
        if self._video_path:
            return self._video_path

        video_path = self.slippi_path.replace(SLIPPI_REPLAYS,
                                              SLIPPI_RECORDINGS)
        video_path = video_path.replace('.slp', '.avi')
        if os.path.exists(video_path):
            self._video_path = video_path
            return video_path

        return None

    @property
    def frame_dir(self):
        '''
        Returns path to directory that stores the extracted frames of the game
        '''
        if self._frame_dir:
            return self._frame_dir

        frame_dir = self.video_path.split('.avi')[0]

        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)

        self._frame_dir = frame_dir
        return self._frame_dir

    @property
    def frame_paths(self):
        return glob.glob(os.path.join(self.frame_dir, '*.jpg'))

    def yield_frames(self):
        '''
        Yields the frames of the video
        '''
        for frame in self.frame_paths:
            yield frame, cv2.imread(frame)

    def extract_video(self):
        '''
        Removes the excess frames on a video
        '''
        if self.frame_paths:
            return

        command = [
            'ffmpeg',
            '-i',
            self.video_path,
            '-r',
            '1/1',
            os.path.join(self.frame_dir, '{}%03d.jpg'.format(self.game_name)),
        ]

        subprocess.call(command)

        self.remove_excess_frames()

    def remove_excess_frames(self):
        '''
        Removes the frames that say "Waiting for Game"
        '''
        for frame_path, frame in self.yield_frames():
            if mse_image_similarity(frame, np.zeros(frame.shape)) < 1000:
                print('Deleting excess frame: {}'.format(frame_path))
                os.remove(frame_path)
