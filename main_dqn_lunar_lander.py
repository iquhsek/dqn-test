import imp
import gym
from src.dqn_lunar_lander import Agent
from utils.plot_learning_curve import plot_learning_curve
import numpy as np


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_min=0.01, input_dim=[8], lr=0.003)
    scores, eps_hist = [], []
    n_games = 500
    
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_  # to the new state
        scores.append(score)
        eps_hist.append(agent.epsilon)
        
        avg_score = np.mean(scores[-100:])
        
        print('episode {}  |  score {}  |  average score {}'.format(i, score, avg_score))
    
    x = [i + 1 for i in range(n_games)]
    filename = 'img/lunar_lander_2022.png'
    plot_learning_curve(x, scores, eps_hist, filename)
