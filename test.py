# Import the gym module
import random
import gym
import numpy as np
import matplotlib.pyplot as plt


from agent import DQN_agent
from utilities import *

# parameters
ACTIONS = 2
NUM_EPISODES = 12000 
NO_OP_STEPS = 30

def playAtariGame():
	# Create a breakout environment
	env = gym.make('Breakout-v0')
	# init agent
	agent = DQN_agent(env)

	# Reset it, returns the starting frame
	frame = env.reset()	

	frame, reward, is_done, _ = env.step(0) # do nothing
	frame = preprocess2(frame, 84, 84)
	print ('after:{}'.format(frame.shape))

	plt.imshow(frame, cmap='gray')
	plt.show()
	agent.initialization(frame)
	#frame = pre.preprocess(frame)
	# Render
	env.render()

	print(env.action_space)
	#print(env.observation_space)
	#print(env.observation_space.high)
	#print(env.observation_space.low)

	
	for _ in range(NUM_EPISODES):

		terminal = False
		frame = env.reset()
		'''
		for _ in range(random.randint(1, NO_OP_STEPS)):
			last_frame = frame
			frame, _, _, _ = env.step(0)  # Do nothing
		'''
		frame = preprocess(frame, 84, 84)
		agent.initialization(frame)
		while not terminal:
			last_frame = frame
			action = agent.get_action(train=True)
			frame, reward, terminal, _ = env.step(action)
			env.render()
			frame = preprocess(frame, 84, 84)
			reward = transform_reward(reward)
			agent.get_observation(action, reward, terminal, frame)

		cost_history = agent.get_costHistory()
		#plot_cost(cost_history)
	

def main():
	playAtariGame()

if __name__ == '__main__':
	main()