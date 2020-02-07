'''
Models outputs the characters in the game
'''
import os
import wandb
import torch
import torch.optim as optim
from torch.nn import BCELoss
from torchsummary import summary

from wavedash.constants import (
    CHAR_DETECTOR_DIR,
    MODEL_CHECKPOINTS,
    CHAR_INPUT_SIZE
)
from wavedash.character_detector.char_model import CharacterPredictModel
from wavedash.character_detector.char_loader import (
    CharacterLoader,
    get_char_dataframe
)

CHAR_DATAFRAME = os.path.join(CHAR_DETECTOR_DIR, 'char_dataframe.pkl')


class TrainCharModel(object):
    '''
    Class for training a model
    '''

    def __init__(self, name, use_wandb=True, print_summary=True):
        '''
        Create the training object
        '''
        self.name = name
        torch.manual_seed(42)

        dataframe = get_char_dataframe()
        train_frame = dataframe[: 3000]
        test_frame = dataframe[3000:]

        self.batch_size = 64

        self.device = 'cuda'
        loader_data = ({'num_workers': 1, 'pin_memory': True}
                       if self.device == 'cuda' else {})
        self.train_loader = torch.utils.data.DataLoader(
            CharacterLoader(train_frame),
            batch_size=self.batch_size,
            shuffle=True,
            **loader_data
        )
        self.test_loader = torch.utils.data.DataLoader(
            CharacterLoader(test_frame),
            batch_size=self.batch_size,
            shuffle=True,
            **loader_data
        )

        self.model = CharacterPredictModel(load_weights=False)
        self.model.to(self.device)

        self.learning_rate = 3e-5
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate)

        self.epochs = 1000

        self.criterion = BCELoss()

        if print_summary:
            summary(self.model,
                    input_size=(CHAR_INPUT_SIZE[2],
                                CHAR_INPUT_SIZE[1],
                                CHAR_INPUT_SIZE[0]),
                    device=self.device)

        self.use_wandb = use_wandb
        if self.use_wandb:
            print('Using wandb!')
            wandb.init(name=self.name,
                       project='smash_char_predictor',
                       force=True)
            wandb.watch(self.model)

    def train(self):
        '''
        Training the character model
        '''
        self.model.train()
        correct = 0
        training_loss = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)

            loss = self.criterion(output, target)

            # computes gradient for all of the parameters w.r.t the loss
            loss.backward()
            # updates the parameters based on the gradient
            self.optimizer.step()

            training_loss += loss.item()

            # import pdb; pdb.set_trace()
            sum_correct = torch.eq(
                torch.sigmoid(output).data > 0.5,
                target.data > 0.5).sum()
            accuracy = (sum_correct / float(output.numel())).item()
            correct += accuracy

        num_batches = len(self.train_loader.dataset) / self.batch_size
        training_loss /= num_batches
        score = 100. * correct / num_batches

        stats = {'training_loss': training_loss, 'training_accuracy': score}
        if self.use_wandb:
            print(stats)
            wandb.log(stats)
        else:
            print(stats)

    def test(self):
        '''
        Test the model
        '''
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()

                sum_correct = torch.eq(
                    torch.sigmoid(output).data > 0.5,
                    target.data > 0.5).sum()
                accuracy = (sum_correct / float(output.numel())).item()
                correct += accuracy

        num_batches = len(self.train_loader.dataset) / self.batch_size
        test_loss /= num_batches
        score = 100. * correct / num_batches

        stats = {'test_loss': test_loss, 'test_accuracy': score}
        if self.use_wandb:
            print(stats)
            wandb.log(stats)
        else:
            print(stats)

    def loop(self):
        '''
        This is the main training and testing loop
        '''
        for epoch in range(1, self.epochs + 1):
            print('Epoch is: {}'.format(epoch))
            self.train()
            self.test()

            if epoch % 50 == 0:
                print('Saving model at epoch {}'.format(epoch))
                torch.save(
                    self.model.state_dict(),
                    os.path.join(MODEL_CHECKPOINTS, self.name + '__' + str(epoch)))


if __name__ == '__main__':
    training_class = TrainCharModel(
        '2_fixed_softmax_sigmoid_error',
        use_wandb=True
    )
    training_class.loop()
    print('done')
