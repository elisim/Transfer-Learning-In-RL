import warnings
warnings.filterwarnings("ignore")
import argparse
from parmas import get_params
from train_actor_critic import ActorCriticTrainer

parser = argparse.ArgumentParser(description='Assignment 3')
parser.add_argument('--mode', type=str, default='train', help='can be fine, train, or transfer')
parser.add_argument('--trained_model', type=str, default='cart', help='can be car, cart, or acro')
parser.add_argument('--test_game', type=str, default='cart', help='can be car, cart, or acro')

games_names = {'cart': 'CartPole-v1',
               'car': 'MountainCarContinuous-v0',
               'acro': 'Acrobot-v1'}


def main():
    args = parser.parse_args()
    trained_model = games_names[args.trained_model]
    test_game = games_names[args.test_game]

    print(f"==== Running with trained_model: {trained_model}, test_game: {test_game} and mode: {args.mode} ====")

    input_dict = get_params(test_game)
    model = ActorCriticTrainer(input_dict)

    if args.mode == 'fine':
        print(f'Loading {trained_model} and testing on {test_game}')
        model.load(f'models/{trained_model}', input_dict)
        model.train()

    elif args.mode == 'train':
        print(f'Training model for {trained_model}')
        model.train()
        model.save(f'models/{trained_model}')

    elif args.mode == 'transfer':
        print(f'Creating prog model')
        model.create_progressive_network([f'models/{trained_model}', f'models/{test_game}'], input_dict)
        model.train()


if __name__ == '__main__':
    main()
