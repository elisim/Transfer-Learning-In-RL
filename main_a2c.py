from parmas import get_params
from train_actor_critic import ActorCriticTrainer

cart = 'CartPole-v1'
car = 'MountainCarContinuous-v0'
acro = 'Acrobot-v1'

trained_model = cart
test_game = cart

mode = 'train-model'
# mode = 'fine-tune'
# mode = 'transfer-learning'

input_dict = get_params(test_game)

model = ActorCriticTrainer(input_dict)
if mode == 'fine-tune':
    print(f'Loading {trained_model} and testing on {test_game}')
    model.load(f'models/{trained_model}', input_dict)
    model.train()

elif mode == 'train-model':
    print(f'Training model for {trained_model}')
    model.train()
    model.save(f'models/{trained_model}')

elif mode == 'transfer-learning':
    print(f'Creating prog model')
    model.create_progressive_network([f'models/{cart}', f'models/{acro}'], input_dict)
    model.train()

