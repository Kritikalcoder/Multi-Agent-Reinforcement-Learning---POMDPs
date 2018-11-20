# Tuple: < x1, y1, x2, y2, x_box1, y_box1, x1_box2, y1_box2,
# x2_box2, y2_box2, x_box3, y_box3 >
# Tuple size: 12
# Grid world dimensions: 10 x 5
# Action count: 5 < up, right, down, left, wait >
# 2 agents
# input transition probabilities file: input.txt

#from math import inf
from math import log
from copy import deepcopy
import sys

actions = []
exp_map = {}
loc_map = {}
ACTS = ["up", "down", "left", "right", "wait"]

input_file = "input_backup.txt"
output_file_1 = "output1.txt"
output_file_2 = "output2.txt"
final_file_1 = "result1.txt"
final_file_2 = "result2.txt"

min_reward = sys.float_info.max
max_reward = sys.float_info.min
action_count = 5

# Step 1: Agent 1: Find enabling actions
# Step 2: Agent 2: Find enabling actions
# Step 3: Agent 1: Rate enabling actions
# Step 4: Agent 2: Rate enabling actions

def find_enabling_actions_agent_1 ():
	partner = 2
	global exp_map, loc_map, min_reward, max_reward, action_count
	exp_map.clear()
	with open(input_file, 'r') as f1 :
		state = f1.readline()
		while state:
			loc_state_2 = get_local_state (state, partner)

			if loc_state_2 in exp_map:
				loc_map = exp_map[loc_state_2]
			else :
				#do nothing
				loc_map.clear()
				# need to add loc_state_2 to exp_map

			for action_1 in xrange(action_count) :
				for action_2 in xrange(action_count) :
					joint_action = f1.readline()
					prob_m = 0.0
					while prob_m < 1.0 :
						next_state = f1.readline().strip()
						loc_state_new = get_local_state (next_state, partner)

						if loc_state_new in loc_map:
							actions = loc_map[loc_state_new]
						else:
							actions = []

						if ACTS[action_1] not in actions:
							actions.append(ACTS[action_1])
							loc_map[loc_state_new] = actions[:]

						prob = float(f1.readline())
						rwd = float(f1.readline())

						if rwd < min_reward:
							min_reward = deepcopy(rwd)
						if rwd > max_reward:
							max_reward = deepcopy(rwd)

						prob_m += prob

			exp_map[loc_state_2] = deepcopy(loc_map)
			#print exp_map[loc_state_2]
			#Change above statement
			state = f1.readline()

	f1.close()

def find_enabling_actions_agent_2 ():
	partner = 1
	global exp_map, loc_map, min_reward, max_reward, action_count
	exp_map.clear()
	with open(input_file, 'r') as f1 :
		state = f1.readline()
		while state:
			loc_state_1 = get_local_state (state, partner)

			if loc_state_1 in exp_map:
				loc_map = exp_map[loc_state_1]
			else :
				#do nothing
				loc_map.clear()
				# need to add loc_state_2 to exp_map

			for action_1 in xrange(action_count) :
				for action_2 in xrange(action_count) :
					joint_action = f1.readline()
					prob_m = 0.0
					while prob_m < 1.0 :
						next_state = f1.readline().strip()
						loc_state_new = get_local_state (next_state, partner)

						if loc_state_new in loc_map:
							actions = loc_map[loc_state_new]
						else:
							actions = []

						if ACTS[action_2] not in actions:
							actions.append(ACTS[action_2])
							loc_map[loc_state_new] = actions[:]

						prob = float(f1.readline())
						rwd = float(f1.readline())

						if rwd < min_reward:
							min_reward = deepcopy(rwd)
						if rwd > max_reward:
							max_reward = deepcopy(rwd)

						prob_m += prob

			exp_map[loc_state_1] = deepcopy(loc_map)
			#Change above statement
			state = f1.readline()

	f1.close()

def rate_enabling_actions_agent_1 ():
	global exp_map, loc_map, min_reward, max_reward, action_count
	other = 2
	with open(input_file, 'r') as f1, open(output_file_1, 'w+') as f2:
		state = f1.readline().strip('\n')
		while state:
			f2.write(state + '\n')

			loc_state_2 = get_local_state(state, other)

			#print "\nGLOBAL STATE: " + state
			#print "  AGENT 2's STATE: " + loc_state_2

			loc_map = exp_map[loc_state_2]

			for action_1 in xrange(action_count):
				f2.write("   " + ACTS[action_1] + '\n')
				enable_value = 0
				for action_2 in xrange(action_count):
					acts = f1.readline()
					prob_m = 0.0
					while prob_m < 1.0:
						next_state = f1.readline().strip('\n').strip()
						loc_state_new = get_local_state(next_state, other)
						#print "  AGENT 2's NEXT STATE: " + loc_state_new + '\n'

						prob = float(f1.readline().strip('\n'))
						rwd = float(f1.readline().strip('\n'))

						actions = loc_map[loc_state_new]

						if len(actions) < action_count and ACTS[action_1] in actions:
							if "-1" not in loc_state_new:
								enable_value += divergence(prob) * (rwd + max_abs_rew)
								#print "rwd: " + rwd + ", abs: " + max_abs_rew

						else:
							pass
							#print "    num acts: " + len(actions) + " " + ACTS[action_1] + " " + ("1" if ACTS[action_1] in actions else "0")


						prob_m += prob

				#print "   " + enable_value
				f2.write("   " + str(float(enable_value)) + '\n')

			state = f1.readline().strip('\n')		

def rate_enabling_actions_agent_2 ():
	global exp_map, loc_map, min_reward, max_reward, action_count
	other = 1
	with open(input_file, 'r') as f1, open(output_file_2, 'w+') as f2:
		state = f1.readline().strip('\n')
		while state:
			f2.write(state + '\n')

			loc_state_1 = get_local_state(state, other)

			#print "\nGLOBAL STATE: " + state
			#print "  AGENT 1's STATE: " + loc_state_1

			loc_map = exp_map[loc_state_1]

			act_vals = [0.0 for i in xrange(action_count)]

			for action_1 in xrange(action_count):
				for action_2 in xrange(action_count):
					acts = f1.readline()
					prob_m = 0.0
					while prob_m < 1.0:
						next_state = f1.readline().strip('\n').strip()
						loc_state_new = get_local_state(next_state, other)
						#print "  AGENT 1's NEXT STATE: " + loc_state_new + '\n'

						prob = float(f1.readline().strip('\n'))
						rwd = float(f1.readline().strip('\n'))

						actions = loc_map[loc_state_new]

						if len(actions) < action_count and ACTS[action_2] in actions:
							act_vals[action_2] += divergence(prob) * (rwd + max_abs_rew)


						prob_m += prob

			for a in xrange(action_count):
				f2.write("   " + ACTS[a] + '\n')
				f2.write("   " + str(float(act_vals[a])) + '\n')

			state = f1.readline().strip('\n')

def finalize ():
	global exp_map, loc_map, min_reward, max_reward, action_count
	# reduce output file 1
	with open(output_file_1, 'r') as f1 :
		with open(final_file_1, 'w') as w1:

			data1 = [""] * (1 + (action_count * 2))
			state = f1.readline()
			while state:
				data1[0] = state
				non_zero = False
				idx = 1
				for a in xrange(action_count) :
					data1[idx] = f1.readline()
					idx += 1
					data1[idx] = f1.readline()
					if float(data1[idx].strip()) != 0.0 :
						non_zero = True
					idx += 1

				if non_zero == True:
					for d in xrange(len(data1)) :
						w1.write(data1[d])

				state = f1.readline()

		w1.close()
	f1.close()

	# reduce output file 2
	with open(output_file_2, 'r') as f2 :
		with open(final_file_2, 'w') as w2:

			data2 = [""] * (1 + (action_count * 2))
			state = f2.readline()
			while state:
				data2[0] = state
				non_zero = False
				idx = 1
				for a in xrange(action_count) :
					data2[idx] = f2.readline()
					idx += 1
					data2[idx] = f2.readline()
					if float(data2[idx].strip()) != 0.0 :
						non_zero = True
					idx += 1

				if non_zero == True:
					for d in xrange(len(data2)) :
						w2.write(data2[d])

				state = f2.readline()

		w2.close()
	f2.close()

def divergence (num):
	jd = num * log(2.0)
	norm = (num + 1.0) / 2.0
	jd += (num * log(num / norm)) + log(1.0 / norm)
	return jd
	'''
	div = num * log(2)
	div += num * log((num * 2.0) / (num + 1.0))
	div += log(2.0 / (num + 1.0))
	return div
	'''

def k_divergence (num):
	return 1.0

def get_local_state (state, partner):
	ls = ""
	state = state.strip()
	s = state.split()
	if partner == 1:
		ls += s[0] + " " + s[1] + " "
	else :
		ls += s[2] + " " + s[3] + " "

	ls += s[4] + " " + s[5] + " " + s[6] + " " + s[7] + " " + s[8] + " " + s[9] + " " + s[10] + " " + s[11]

	return ls

find_enabling_actions_agent_1 ()
max_abs_rew = max (abs(min_reward), abs(max_reward))
rate_enabling_actions_agent_1 ()

find_enabling_actions_agent_2 ()
rate_enabling_actions_agent_2 ()
finalize ()