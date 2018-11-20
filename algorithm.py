from math import log
from copy import deepcopy
import sys
from random import random, choice

# Agents and boxes
# Environment (box)
# states and local states
# State action pairs
# RL algorithms
# Q values for every state action pair
# Epsilon reduction every 5000 steps

# Box world dimensions
HEIGHT = 5
WIDTH = 10

# Epsilon values
epsilon = 0.1
reduction_rate = 5000
epsilon_reduction = 0.01
epsilon_threshold = 0.01
number_of_reductions = 0
reduction_factor = 0

T = 100000


# Q learning values
alpha = 0.5
gamma = 1.0

# Q values data structure
q_value_1 = {}
q_value_2 = {}

# Rewards for boxes
large_box_rew = 1000
small_box_rew = 50

preload_1 = "result1.txt"
preload_2 = "result2.txt"
trans_file = "input_backup.txt"
# need to use the preloaded heuristic values

# Dec-POMDP inputs
joint_reward = 0.0
state = {'a1':(2,4), 'a2':(7,4), 'b1':(1,3), 'lb1':(4,3), 'lb2':(5,3), 'b2':(8,3)}
local_state_1 = {'loc':(2,4), "up":"empty", "right":"empty", "down":"wall", "left":"empty"}
local_state_2 = {'loc':(7,4), "up":"empty", "right":"empty", "down":"wall", "left":"empty"}
# wall, agent, smallBox1, smallBox2, largeBox, empty
actions = ["up", "down", "left", "right", "wait"]
action_count = 5

def ls_to_str(ls):
	return str(ls["loc"][0]) + " " + str(ls["loc"][1]) + " " + ls["up"] + " " + ls["down"] + " " + ls["left"] + " " + ls["right"]

def initialize_q ():
	#initialize q values for both agents by parsing
	# the respective preload files
	# Format: q_value_1 [ ([1 2 0 0 1 1 4 3 5 3],"up") ] = 2366.68726333

	## For agent 1
	global q_value_1, q_value_2, actions
	with open('result1.txt', 'r') as f:
		state_line = f.readline()
		while state_line:
			coords = state_line.strip('\n').split()
			coords = [int(x) for x in coords]
			gs = {}
			gs['a1'] = (coords[0], coords[1])
			gs['a2'] = (coords[2], coords[3])
			gs['b1'] = (coords[4], coords[5])
			gs['lb1'] = (coords[6], coords[7])
			gs['lb2'] = (coords[8], coords[9])
			gs['b2'] = (coords[10], coords[11])

			ls = get_local_state_from_world_state(gs, 1)
			for action in actions:
				act = f.readline().strip('\n')
				util = float(f.readline().strip('\n'))
				q_value_1[(ls_to_str(ls), action)] = util

			state_line = f.readline()

	## For agent 2
	with open('result2.txt', 'r') as f:
		state_line = f.readline()
		while state_line:
			coords = state_line.strip('\n').split()
			coords = [int(x) for x in coords]
			gs = {}
			gs['a1'] = (coords[0], coords[1])
			gs['a2'] = (coords[2], coords[3])
			gs['b1'] = (coords[4], coords[5])
			gs['lb1'] = (coords[6], coords[7])
			gs['lb2'] = (coords[8], coords[9])
			gs['b2'] = (coords[10], coords[11])


			ls = get_local_state_from_world_state(gs, 2)
			for action in actions:
				act = f.readline().strip('\n')
				util = float(f.readline().strip('\n'))
				q_value_2[(ls_to_str(ls), action)] = util

			state_line = f.readline()

def get_local_state_from_world_state (s, agent) :
	global HEIGHT, WIDTH

	ls = {}
	if agent == 1:
		ls['loc'] = s['a1']
		other_agent = 'a2'
	else :
		ls['loc'] = s['a2']
		other_agent = 'a1'

	# {'loc':(2,4), "up":"empty", "right":"empty", "down":"wall", "left":"empty"}

	#Up
	x = ls['loc'][0]
	y = ls['loc'][1] - 1
	if y < 0 :
		ls['up'] = "wall"
	elif x == s[other_agent][0] and y == s[other_agent][1] :
		ls['up'] = "agent"
	elif x == s['b1'][0] and y == s['b1'][1] :
		ls['up'] = "smallBox1"
	elif x == s['b2'][0] and y == s['b2'][1] :
		ls['up'] = "smallBox2"
	elif x == s['lb1'][0] and y == s['lb1'][1] :
		ls['up'] = "largeBox"
	elif x == s['lb2'][0] and y == s['lb2'][1] :
		ls['up'] = "largeBox"
	else :
		ls['up'] = "empty"

	#Right
	x = ls['loc'][0] + 1
	y = ls['loc'][1]
	if x >= WIDTH :
		ls['right'] = "wall"
	elif x == s[other_agent][0] and y == s[other_agent][1] :
		ls['right'] = "agent"
	elif x == s['b1'][0] and y == s['b1'][1] :
		ls['right'] = "smallBox1"
	elif x == s['b2'][0] and y == s['b2'][1] :
		ls['right'] = "smallBox2"
	elif x == s['lb1'][0] and y == s['lb1'][1] :
		ls['right'] = "largeBox"
	else :
		ls['right'] = "empty"

	#Down
	x = ls['loc'][0]
	y = ls['loc'][1] + 1
	if y >= HEIGHT :
		ls['down'] = "wall"
	elif x == s[other_agent][0] and y == s[other_agent][1] :
		ls['down'] = "agent"
	elif x == s['b1'][0] and y == s['b1'][1] :
		ls['down'] = "smallBox1"
	elif x == s['b2'][0] and y == s['b2'][1] :
		ls['down'] = "smallBox2"
	elif x == s['lb1'][0] and y == s['lb1'][1] :
		ls['down'] = "largeBox"
	elif x == s['lb2'][0] and y == s['lb2'][1] :
		ls['down'] = "largeBox"
	else :
		ls['down'] = "empty"

	#Left
	x = ls['loc'][0] - 1
	y = ls['loc'][1]
	if x < 0 :
		ls['left'] = "wall"
	elif x == s[other_agent][0] and y == s[other_agent][1] :
		ls['left'] = "agent"
	elif x == s['b1'][0] and y == s['b1'][1] :
		ls['left'] = "smallBox1"
	elif x == s['b2'][0] and y == s['b2'][1] :
		ls['left'] = "smallBox2"
	elif x == s['lb2'][0] and y == s['lb2'][1] :
		ls['left'] = "largeBox"
	else :
		ls['left'] = "empty"

	return ls 

def update_q (a1, a2, r):
	# update q values for both agents
	# after every joint action taken, update q values
	global state, curr_state, q_value_1, q_value_2, alpha, gamma

	ls1 = get_local_state_from_world_state (curr_state, 1)
	ls2 = get_local_state_from_world_state (curr_state, 2)

	if (ls_to_str(ls1), a1) not in q_value_1 :
		q_value_1[(ls_to_str(ls1), a1)] = 0.0
	if (ls_to_str(ls2), a2) not in q_value_2 :
		q_value_2[(ls_to_str(ls2), a2)] = 0.0
	q_v1 = deepcopy( q_value_1[(ls_to_str(ls1), a1)] )
	q_v2 = deepcopy( q_value_2[(ls_to_str(ls2), a2)] )
	new_q1 = q_v1 + alpha * (r)
	new_q2 = q_v2 + alpha * (r)

	sap_1 = get_feasible_local_state_action_pairs(curr_state, 1)
	# sap_1 has atmost 5 elements
	# sap_2 has atmost 5 elements
	sap_2 = get_feasible_local_state_action_pairs(curr_state, 2)

	max_r1 = sys.float_info.min
	max_r2 = sys.float_info.min

	for sap in sap_1 :
		if sap not in q_value_1 :
			q_value_1[sap] = 0.0
		if max_r1 < q_value_1[sap]:
			max_r1 = q_value_1[sap]

	for sap in sap_2 :
		if sap not in q_value_2:
			q_value_2[sap] = 0.0
		if max_r2 < q_value_2[sap]:
			max_r2 = q_value_2[sap]

	new_q1 += alpha * gamma * max_r1
	new_q2 += alpha * gamma * max_r2

	q_value_1[(ls_to_str(ls1), a1)] = new_q1
	q_value_2[(ls_to_str(ls2), a2)] = new_q2

def perform_joint_action (source, act_1, act_2):
	global large_box_rew, small_box_rew
	#rew = -2
	rew = 0
	# reward += new_reward -1 for every move
	ls1 = get_local_state_from_world_state(source, 1)
	ls2 = get_local_state_from_world_state(source, 2)
	flag1 = True
	flag2 = True
	flag_b1_push = False
	flag_b2_push = False
	flag_lb_push = False

	res_x1 = ls1["loc"][0]
	res_y1 = ls1["loc"][1]
	res_x2 = ls2["loc"][0]
	res_y2 = ls2["loc"][1]

	if act_1 == "up":
		res_y1 -= 1
	if act_1 == "down":
		res_y1 += 1
	if act_1 == "left":
		res_x1 -= 1
	if act_1 == "right":
		res_x1 += 1

	if act_2 == "up":
		res_y2 -= 1
	if act_2 == "down":
		res_y2 += 1
	if act_2 == "left":
		res_x2 -= 1
	if act_2 == "right":
		res_x2 += 1

	if (res_x1, res_y1) == (res_x2, res_y2):
		flag1 = False
		flag2 = False
		#return source, -2
		return source, 0

	if act_1 == "wait" and act_2 == "wait":
		#return source, -2
		return source, 0

	if act_1 != "wait":
		if ls1[act_1] == "smallBox1" or ls1[act_1] == "smallBox2" :
			num = random()
			if num > 0.8 :
				flag1 = False
			else :
				flag_b1_push = True
				if (res_x2, res_y2) == (res_x1, res_y1 + 2) :
					flag2 = False

	if act_2 != "wait":
		if ls2[act_2] == "smallBox1" or ls2[act_2] == "smallBox2" :
			num = random()
			if num > 0.8 :
				flag_b2_push = True
				flag2 = False
			else :
				if (res_x1, res_y1) == (res_x2, res_y2 + 2) :
					flag1 = False

	if act_1 != "wait" and act_2 != "wait":
		if ls1[act_1] == "largeBox" and ls2[act_2] == "largeBox" :
			num2 = random()
			if num2 > 0.64 :
				# large box push is going to fail
				#return source, -2
				return source, 0
			else:
				flag_lb_push = True
				# pass

		if ls1[act_1] == "largeBox" and ls2[act_2] != "largeBox":
			flag1 = False

		if ls1[act_1] != "largeBox" and ls2[act_2] == "largeBox":
			flag2 = False

	if act_1 != "wait" :
		if ls1[act_1] == "largeBox" :
			flag1 = False


	if act_2 != "wait":
		if ls2[act_2] == "largeBox" :
			flag2 = False


	if flag1 == True :
		if act_1 != "wait" :
			source["a1"] = (res_x1, res_y1)
			if ls1[act_1] == "smallBox1":
				rew += small_box_rew
				source["b1"] = (source["b1"][0], source["b1"][1] - 1) 

			if ls1[act_1] == "smallBox2":
				rew += small_box_rew
				source["b2"] = (source["b2"][0], source["b2"][1] - 1)
		
	if flag2 == True:
		if act_2 != "wait" :
			source["a2"] = (res_x2, res_y2)

			if ls2[act_2] == "smallBox1":
				rew += small_box_rew
				source["b1"] = (source["b1"][0], source["b1"][1] - 1)

			if ls2[act_2] == "smallBox2":
				rew += small_box_rew
				source["b2"] = (source["b2"][0], source["b2"][1] - 1)

	if flag1 == True and flag2 == True:
		if act_1 != "wait":
			if ls1[act_1] == "largeBox":
				rew += large_box_rew
				source["lb1"] = (source["lb1"][0], source["lb1"][1] - 1)
				source["lb2"] = (source["lb2"][0], source["lb2"][1] - 1)

	return source, rew

def box_respawn_check (box_name):
	# state = {'a1':(2,4), 'a2':(7,4), 'b1':(1,3), 'b2':(8,3), 'lb1':(4,3), 'lb2':(5,3)}
	global state
	flag = True

	if box_name == 'b1' :
		for key, value in state :
			if value == (1,3) :
				flag = False

	elif box_name == 'b2' :
		for key, value in state :
			if value == (8,3) :
				flag = False

	elif box_name == 'lb' :
		for key, value in state :
			if value == (4,3) or value == (5,3):
				flag = False

	return flag

def reduceEpsilon() :
	global epsilon, epsilon_threshold, epsilon_reduction, number_of_reductions, reduction_rate, reduction_factor
	number_of_reductions += 1
	if number_of_reductions == reduction_rate:
		reduction_factor += 1
		if epsilon > epsilon_threshold :
			epsilon = epsilon / reduction_factor
		if epsilon < epsilon_threshold :
			epsilon = epsilon_threshold
		number_of_reductions = 0

def get_feasible_local_state_action_pairs(s, agent):
	# {'loc':(2,4), "up":"empty", "right":"empty", "down":"wall", "left":"empty"}

	ls = get_local_state_from_world_state (s, agent)
	fsap = []
	for action in actions:
		if action == "wait":
			fsap.append((ls_to_str(ls), action))
			continue
		if ls[action] == "empty" or ls[action] == "agent":
			fsap.append((ls_to_str(ls), action))
			continue
		if ls[action] == "wall":
			continue
		if ls[action] == "largeBox" or ls[action] == "smallBox1" or ls[action] == "smallBox2":
			if action == "up":
				if ls["loc"][1] != 1:
					fsap.append((ls_to_str(ls), action))

	return fsap

initialize_q()
start_state = state
curr_state = deepcopy(state)

while T > 0:

	curr_state = deepcopy(state)
	sap_1 = get_feasible_local_state_action_pairs(curr_state, 1)
	sap_2 = get_feasible_local_state_action_pairs(curr_state, 2)

	max_r1 = sys.float_info.min
	max_r2 = sys.float_info.min
	act_max_1 = "wait"
	act_max_2 = "wait"

	num = random()
	if num < epsilon :
		act_max_1 = choice(sap_1)[1]
		act_max_2 = choice(sap_2)[1]

	else :
		for sap in sap_1 :
			if sap not in q_value_1 :
				q_value_1[sap] = 0.0
			if max_r1 < q_value_1[sap]:
				max_r1 = q_value_1[sap]
				act_max_1 = sap[1]

		for sap in sap_2 :
			if sap not in q_value_2:
				q_value_2[sap] = 0.0
			if max_r2 < q_value_2[sap]:
				max_r2 = q_value_2[sap]
				act_max_2 = sap[1]

	# both agents take actions
	# update world state
	state, r = perform_joint_action(state, act_max_1, act_max_2)
	# if r > 0:
		# print r, T, joint_reward
	#print state["b1"]
	if state["b1"][1] == 0:
		state["b1"] = (1,3)
		#print box_respawn_check("b1")
		#if box_respawn_check("b1"):
			#state["b1"] = (1, 3)  

	if state["b2"][1] == 0:
		state["b2"] = (8, 3) 
		# if box_respawn_check("b2"):
			# state["b2"] = (8, 3)  

	if state["lb1"][1] == 0 and state["lb2"][1] == 0:
		state["lb1"] = (4, 3)
		state["lb2"] = (5, 3)
		'''
		if box_respawn_check("lb"):
			state["lb1"] = (4, 3)
			state["lb2"] = (5, 3)
		'''

	reduceEpsilon()
	# update q values
	update_q(act_max_1, act_max_2, r)

	joint_reward += r

	T -= 1

print joint_reward