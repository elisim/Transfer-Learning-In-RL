import numpy as np

from game_configs import cartpole, acrobot
from actor_critic import ActorCriticTrainer

np.random.seed(42)

trained_model_config = cartpole
test_game_config = cartpole

transfer_learning = False
model = ActorCriticTrainer(trained_model_config)
if not transfer_learning:
    model.train()
    model.save(f'models\{trained_model_config.name}')
# else:
#     print(f"loading dataset {trained_model} and playing on {test_game}")
#     # model.load_prog_network([f'models\{acro}',f'models\{cart}'],input_dict)
#     model.load(f'models\{acro}', input_dict)
#     model.train()

pass
