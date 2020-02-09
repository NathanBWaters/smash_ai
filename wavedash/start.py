'''
Experiments with Super Smash Bro Melee Slippi Files
'''
import cv2 as cv
from skimage.metrics import structural_similarity

# from wavedash.data import get_fox_marth_games


def export_diff_video():
    '''
    Returns coordinates of the back button
    '''
    cap_a = cv.VideoCapture('/Volumes/Mac Drive/smash_ai_footage/withoutCharactersQuick15.mov')
    cap_b = cv.VideoCapture('/Volumes/Mac Drive/smash_ai_footage/withCharactersQuick15.mov')

    index = 0
    while(True):
        # Capture frame-by-frame
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()

        index += 1
        print('On index: {}'.format(index))
        if index > 500:
            # Display the resulting frame
            # cv.imshow('frame_a', frame_a)
            # cv.imshow('frame_b', frame_b)
            (score, diff) = structural_similarity(frame_a, frame_b, full=True, multichannel=True)
            cv.imwrite('_frame_a.png', frame_a)
            cv.imwrite('_frame_b.png', frame_b)
            cv.imwrite('_testing.png', diff)
            break


if __name__ == '__main__':
    games = export_diff_video()

    print('done')
