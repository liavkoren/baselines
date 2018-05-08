import json

from baselines import deepq
from google.colab import files
import gym




def callback(lcl, _glb):
    # locals, globals passed from training function. This gets to decide if the
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    info = {
        'episode_rewards': lcl['episode_rewards'],
        'td_errors': lcl['td_errors'],
        'mean_100ep_reward': lcl['mean_100ep_reward'],
    }
    print(json.dumps(info), end='', flush=False)
    files.download(lcl['model_file']) # from colab to browser download
    return is_solved


def main():
    env = gym.make("CartPole-v0")
    model = deepq.models.mlp([64])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
