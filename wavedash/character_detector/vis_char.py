'''
Helps visualize the data being fed to the models
'''
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from wavedash.constants import CHAR_CLASSES
from wavedash.character_detector.char_loader import CharacterLoader
from wavedash.character_detector.char_model import CharacterPredictModel

torch.set_printoptions(precision=3, sci_mode=False)


def char_output_visualizer(char_data, y_name='char_data'):
    '''
    Visualize the output of the character model for a single image
    '''
    plt.clf()
    data = pd.DataFrame({
        y_name: torch.exp(char_data),
        'class': list(range(CHAR_CLASSES)),
    })
    ax = sns.barplot(x='class', y=y_name, data=data)
    ax.figure.set_figwidth(10)
    ax.figure.set_figheight(5)
    st.pyplot()


def render_data_loader():
    '''
    Renders synthetic letters
    '''
    with torch.no_grad():
        model = CharacterPredictModel(load_weights=True)
        model.eval()
        loader = CharacterLoader()
        criterion = torch.nn.BCELoss()
        for i in range(15):
            feature, label = loader[i]
            output = model(torch.unsqueeze(feature, 0))

            # st.write('Feature: {}'.format(character))
            st.image(feature.numpy().transpose(1, 2, 0), width=128, clamp=True)
            st.write('Label: {}'.format(str(label)))
            st.write('Output: {}'.format(str(output)))
            st.write('Loss: {:3f}'.format(criterion(
                torch.unsqueeze(torch.tensor(label), 0),
                torch.unsqueeze(output, 0)
            ).item()))
            # char_output_visualizer(output[0])

            st.write('-' * 80)


if __name__ == '__main__':
    model = None
    # render_generator(model)
    # render_dataframe()
    # render_synthetic()
    render_data_loader()
