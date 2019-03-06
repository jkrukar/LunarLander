from keras.models import Model,Sequential
from keras.layers import Dense, Input, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.layers.merge import Add
import keras.backend as kb
import keras.callbacks as kc
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gym
from collections import deque
import random

epsilon = 0.95 #Explore vs exploit
#Do I need to use entropy in loss function???
#Should I use tanh as activation function??
#Could add alpha - learning rate param to Adam optimizer
#callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs')] - add to fit method

class LunarLander:

    def __init__(self, env):

        self.env = env
        self.experiences = deque(maxlen = 100000)
        self.experiences = deque()
        self.history = kc.History()

        # self.tensorboard = kc.TensorBoard(log_dir='./logs')
        # print(env.observation_space)
        # print(env.action_space)

        #Hyperparameters
        self.alpha = 0.0001
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_decay_rate = 0.995
        self.tau = 0.125

        self.model = self.build_model() #role of the model (self.model) is to do the actual predictions on what action to take
        self.target_model = self.build_model() #target model (self.target_model) tracks what action we want our model to take.

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
        model.add(Dense(64, activation='relu', input_dim=8))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(4))

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
        batch_size = 64

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
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay_rate

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state)[0])

    # def save_weights(self,episode):
    #     file_name = "lunarLander_"+str(episode)+".h5"
    #     self.model.save_weights(file_name)



# np.random.seed(5)


env = gym.make('LunarLander-v2')
env.reset()

observation = env.reset()
done = False
lunar_lander = LunarLander(env)

episode_rewards = []

episodes = 5000
episode_length = 1000

# kc.TensorBoard(log_dir='./logs')

for i in range(episodes):
    episode_reward = 0
    state = env.reset().reshape(1,8)
    for j in range(episode_length):

        env.render()

        action = lunar_lander.get_action(state)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        next_state = next_state.reshape(1,8)
        lunar_lander.store_experience(state, next_state, reward, done, action)
        lunar_lander.learn_from_experiences()
        lunar_lander.update_target_model_weights()

        state = next_state

        if done:
            break

    episode_rewards.append(episode_reward)


    # if i%10 == 0:
    plt.clf()
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.show(block=False)
    plt.pause(0.001)

    # avg_reward_last_ten = sum(episode_rewards[len(episode_rewards)-10:])/10
    # print("Average reward over last ten episodes: ",avg_reward_last_ten)

    # if avg_reward_last_ten > 200 or i%100==0:
    #     lunar_lander.save_weights(i)

    # if i ==0:
    #     lunar_lander.save_weights(i)

    print("episode: ",i," Reward= ",episode_reward)

      # lunar_lander.store_experience(experience,action)