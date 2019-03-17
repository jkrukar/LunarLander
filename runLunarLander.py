from keras.models import Model,Sequential
from keras.layers import Dense, Input, BatchNormalization, Activation
from keras.optimizers import Adam,RMSprop
from keras.layers.merge import Add
import keras.backend as kb
import keras.callbacks as kc
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gym
from collections import deque
import random
import csv
import datetime

class LunarLander:

    def __init__(self, env):

        self.training_start = datetime.datetime.now()
        self.env = env
        self.experiences = deque(maxlen = 10000)

        #Hyperparameters
        self.alpha = 0.0001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay_rate = 0.9
        self.tau = 0.2

        self.model = self.build_model() #role of the model (self.model) is to do the actual predictions on what action to take
        self.target_model = self.build_model() #target model (self.target_model) tracks what action we want our model to take.

        self.model.load_weights('lunarLanderWeights.h5')

    def store_experience(self, state, next_state, reward, done, action):
        
        self.experiences.appendleft([state,next_state,reward,action,done])
        # self.experiences.append([state,next_state,reward,action,done])
        # print("observation:",observation)
        # print("reward:",reward)
        # print("done:",done)

    #build model, got help from keras tutorial
    #https://keras.io/getting-started/sequential-model-guide/
    def build_model(self):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim=8))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(4, activation='linear'))

        # model.add(Dense(64, use_bias=False, input_dim=8))
        # model.add(BatchNormalization())
        # model.add(Activation("relu"))
        # model.add(Dense(64, use_bias=False))
        # model.add(BatchNormalization())
        # model.add(Activation("relu"))
        # model.add(Dense(4, activation='linear')) #CHANGE TO RELU? OR LINEAR
        model.compile(optimizer=Adam(lr=self.alpha),loss='mean_squared_error',metrics=['accuracy'])
        return model

    def learn_from_experiences(self):
        #take batch
        batch_size = 128

        if batch_size > len(self.experiences):
            return

        sample_experiences = random.sample(self.experiences, batch_size)
        for i in range(batch_size):
            next_sample_experience = sample_experiences[i]
            state,next_state,reward,action,done = next_sample_experience
            target_prediction = self.target_model.predict(state)

            if done:
                target_prediction[0][action] = reward
            else:
                future_reward = max(self.target_model.predict(next_state)[0])
                target_prediction[0][action] = reward + future_reward * self.gamma

            # self.model.fit(state,target_prediction, verbose=0, callbacks=[self.tensorboard])
            self.model.fit(state,target_prediction, verbose=0)
            #print(self.history.history)


    def update_target_model_weights(self):
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()

        for i in range(len(model_weights)):
            target_model_weights[i] = self.tau * model_weights[i] + (1-self.tau) * target_model_weights[i]

        self.target_model.set_weights(target_model_weights)

    def get_action(self, state):
        if self.epsilon > 0.05:
            self.epsilon *= self.epsilon_decay_rate

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state)[0])

    def save_weights(self,avg_reward_last_hundred):
        file_name = "lunarLander_"+str(avg_reward_last_hundred)+".h5"
        self.model.save_weights(file_name)

    def save_model(self, avg_reward_last_hundred):
        file_name = "lunarLander_Model_"+str(avg_reward_last_hundred)
        self.model.save(file_name)


def log_csv(lunar_lander, episode_rewards,hundred_ep_reward_avgs,episode_lengths):
    file_name = 'Logs/Log_' + str(lunar_lander.training_start) + '.csv'
    episode_rewards_length = len(episode_rewards)
    hundred_ep_reward_avgs_length = len(hundred_ep_reward_avgs)
    episode_lengths_length = len(episode_lengths)
    
    with open(file_name, mode='w') as log_file:
        log_writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        log_writer.writerow(['Episodes = '+str(len(episode_rewards)),
            'alpha='+str(lunar_lander.alpha),
            'gamma='+str(lunar_lander.gamma),
            'epsilon='+str(lunar_lander.epsilon),
            'epsilon_decay_rate='+str(lunar_lander.epsilon_decay_rate),
            'tau='+str(lunar_lander.tau),
            'batch=128',
            '128relu/128relu/linear',
            'RMSprop opt',
            'Loss mse'])

        log_writer.writerow(['episode_rewards','hundred_ep_reward_avgs','episode_lengths'])

        for i in range(episode_rewards_length):
            next_row = []

            next_row.append(episode_rewards[i])
            if i<hundred_ep_reward_avgs_length:
                next_row.append(hundred_ep_reward_avgs[i])

            if i<episode_lengths_length:
                next_row.append(episode_lengths[i])

            log_writer.writerow(next_row)





# np.random.seed(5)

for k in range(2):

    env = gym.make('LunarLander-v2')
    env.reset()

    observation = env.reset()
    done = False
    lunar_lander = LunarLander(env)

    episode_rewards = []
    hundred_ep_reward_avgs = []
    episode_lengths = []

    episodes = 1100
    episode_length = 500

    avg_reward_last_hundred = 0

    for i in range(episodes):

        episode_reward = 0
        state = env.reset().reshape(1,8)

        for j in range(episode_length):

            env.render()

            action = lunar_lander.get_action(state)
            # action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            next_state = next_state.reshape(1,8)
            lunar_lander.store_experience(state, next_state, reward, done, action)

            state = next_state

            if done:
                episode_lengths.append(j)
                break

        episode_rewards.append(episode_reward)

        if i%10 == 0:
            plt.figure(1)
            plt.clf()
            plt.plot(episode_rewards)
            plt.title('Episode Rewards')
            plt.ylabel('Reward')
            plt.xlabel('Episode')

            plt.figure(2)
            plt.clf()
            plt.plot(hundred_ep_reward_avgs)
            plt.title('Previous 100 Episodes Avg Rewards')
            plt.ylabel('Reward')
            plt.xlabel('Measurements')


            plt.show(block=False)
            plt.pause(0.001)

            if i>100:
                avg_reward_last_hundred = sum(episode_rewards[i-100:])/100
                print("Average reward over last hundred episodes: ",avg_reward_last_hundred)
                hundred_ep_reward_avgs.append(avg_reward_last_hundred)


            log_csv(lunar_lander,episode_rewards,hundred_ep_reward_avgs,episode_lengths)

        # if i ==0:
        #     lunar_lander.save_weights(i)

        print("episode: ",i,"\tReward= ",episode_reward,"\tPrevious Hundred Episode Reward= ",avg_reward_last_hundred)

          # lunar_lander.store_experience(experience,action)