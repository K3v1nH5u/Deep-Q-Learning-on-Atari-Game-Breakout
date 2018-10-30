import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Convolution2D, Flatten, Dense, LeakyReLU, Multiply, Maximum, Add, merge, Lambda
from keras.optimizers import RMSprop,Adam
import keras.backend as K
from collections import deque 
from utilities import *
from tensorflow.python.client import device_lib

class DQN_agent(object):
	def __init__(self, env):
		self.num_actions = env.action_space.n

		self.frame_width = 84
		self.frame_height = 84
		self.state_length = 4
		self.learning_rate = 0.00025

		self.MIN_GRAD = 0.01
		self.MOMENTUM = 0.95
		self.BATCH_SIZE = 32
		self.GAMMA = 0.99  # Discount factor
		self.EXPLORATION_STEPS = 1000000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
		self.INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
		self.FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
		self.INITIAL_REPLAY_SIZE = 200  # Number of steps to populate the replay memory before training starts
		self.NUM_REPLAY_MEMORY = 40
		self.SAVE_INTERVAL  = 1000 #300000
		self.TRAIN_INTERVAL = 4

		self.checkpoints_path = "checkpoints/"
		self.exp_name = "breakout_v0"

		self.timeStep = 0
		self.total_loss = 0
		self.total_reward = 0
		self.total_q_max = 0
		self.duration = 0
		self.episode = 0

		self.epsilon = self.INITIAL_EPSILON
		self.epsilon_step = (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORATION_STEPS

		self.dummy_input = np.zeros((1,self.num_actions))
		self.dummy_batch = np.zeros((self.BATCH_SIZE, self.num_actions))
		# init replay memory
		self.memory = deque()
		# Create optimizer
		self.opt = RMSprop(lr=self.learning_rate, decay=0, rho=0.99, epsilon=self.MIN_GRAD)
		# Create q network
		self.q_network = self.create_network()

		self.cost_history = []

		# log file
		self.reward_file = open("logs_" + self.exp_name + "/reward.txt", 'w')
		self.q_value_file = open("logs_" + self.exp_name + "/q_value.txt", 'w')

		print device_lib.list_local_devices()

	def create_network(self):
		ATARI_SHAPE = (self.state_length, self.frame_height, self.frame_width )

		frames_input = Input(shape=(self.frame_width, self.frame_height, self.state_length))
		actions_input = Input(shape=(self.num_actions,))
		conv1 = Convolution2D(16, 8, 8, subsample=(4, 4), activation='relu')(frames_input)
		conv2 = Convolution2D(32, 4, 4, subsample=(2, 2), activation='relu')(conv1)
		#conv3 = Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu')(conv2)
		flat_feature = Flatten()(conv2)
		hidden_feature = Dense(256)(flat_feature)
		lrelu_feature = LeakyReLU()(hidden_feature)
		q_value_prediction = Dense(self.num_actions)(lrelu_feature)

		select_action = Multiply()([q_value_prediction, actions_input])
		q_action = Lambda(lambda x: K.sum(x, axis=1, keepdims=True), output_shape=lambda_out_shape)(select_action)
		model = Model(input=[frames_input, actions_input], output=[q_value_prediction, q_action])
		model.compile(loss=['mse', 'mse'], loss_weights=[0.0,1.0], optimizer=self.opt)
		return model

	def train_network(self):
		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(self.memory,self.BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		nextState_batch = [data[3] for data in minibatch]
		terminal_batch = [data[4] for data in minibatch]

		# Step 2: calculate y
		terminal_batch = np.array(terminal_batch) + 0
		q_value_batch = self.q_network.predict([list2np(nextState_batch), self.dummy_batch])[0]
		y_batch = reward_batch + (1 - terminal_batch) * self.GAMMA * np.max(q_value_batch, axis=-1) 

		action_ont_hot = np.zeros((self.BATCH_SIZE, self.num_actions))
		for idx, ac in enumerate(action_batch):
			action_ont_hot[idx, ac] = 1.0

		loss = self.q_network.train_on_batch([list2np(state_batch), action_ont_hot], [self.dummy_batch, y_batch])

		self.cost_history.append(loss)

		# Save network
		if self.timeStep % self.SAVE_INTERVAL == 0:
			save_path = self.checkpoints_path + '/' + self.exp_name+'_'+str(self.timeStep)+'.h5'
			self.q_network.save(save_path)
			print('Successfully saved: ' + save_path)
		
		self.total_loss += loss[1]
		#print 'Loss: {}'.format(self.total_loss)


	def get_observation(self, action, reward, terminal, observation):
		# pop out the first frame
		nextState = np.dstack((observation, self.currentState[:,:,1:]))
		#nextState = np.append(observation, self.currentState[:,:,1:],  axis = 2)
		self.memory.append((self.currentState, action, reward, nextState, terminal))

		if len(self.memory) > self.NUM_REPLAY_MEMORY:
			self.memory.popleft()
			#print 'time_stamep:{}... Explore!'.format(self.timeStep)

		if self.timeStep > self.INITIAL_REPLAY_SIZE and self.timeStep % self.TRAIN_INTERVAL == 0:
			self.train_network()
			#print 'time_stamep:{}... Training!'.format(self.timeStep)
		
		self.total_reward += reward
		self.total_q_max += np.max(self.q_network.predict([np.expand_dims(self.currentState, axis=0),self.dummy_input])[0])

		self.currentState = nextState
		self.duration += 1
		self.timeStep += 1

		if terminal:
			print 'EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / AVG_REWARD: {4:2.3f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f}'.format(
				self.episode + 1, self.timeStep, self.duration, self.epsilon,
				self.total_reward / float(self.duration), self.total_q_max / float(self.duration),
				self.total_loss / (float(self.duration) / float(self.TRAIN_INTERVAL)))
			# Init for new game
			self.total_reward = 0
			self.total_q_max = 0
			self.total_loss = 0
			print 'total_loss:{}'.format(self.total_loss)
			self.duration = 0
			self.episode += 1

	def get_action(self, train=False):
		if train:

			if self.epsilon >= random.random() or self.timeStep < self.INITIAL_REPLAY_SIZE:
				#print 'time_stamep:{}...Random!'.format(self.timeStep)
				action = random.randrange(self.num_actions)
			else:
				#print 'time_stamep:{}...Experience!'.format(self.timeStep)
				action = np.argmax(self.q_network.predict([np.expand_dims(self.currentState,axis=0), self.dummy_input])[0])

			if self.epsilon > self.FINAL_EPSILON and self.timeStep >= self.INITIAL_REPLAY_SIZE:
				self.epsilon -= self.epsilon_step
		else:
			if 0.005 >= random.random():
				action = random.randrange(self.num_actions)
			else:
				action = np.argmax(self.q_network.predict([np.expand_dims(self.currentState,axis=0), self.dummy_input])[0])

		return action

	def initialization(self, observation):
		self.currentState = np.stack( (observation, observation, observation, observation), axis=2)
		print ('Init:{}'.format(self.currentState.shape))


	def get_costHistory(self):
		return self.cost_history





	

