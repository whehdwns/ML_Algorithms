#!/usr/bin/env python
# coding: utf-8

#Dongjun Cho Project 5

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import os
import glob
import math
import copy
import random
from collections import Counter # Mode
import warnings #Remove Warning Message

warnings.filterwarnings('ignore')
path = os.getcwd()
all_files = glob.glob(path + "/dataset/*.txt")
filesnames = os.listdir('dataset/')

# Open Track text dataset
# It loads txt file dataset. 

def open_dataset(track_type):
    for i in all_files:
        if track_type in i:
            track_txt = pd.read_csv(i, skiprows = 1, header = None)
            track = np.array(track_txt)
    return track

#It converts txt dataset to numpy list.

def track_nplist(track_matrix):
    full_track =list()
    for i in range(len(track_matrix)):
        track = list()
        for j in track_matrix[i][0]:
            track.append(j)
        full_track.append(track)
    return np.array(full_track, dtype='object')

# It finds the position of each type of position. 
# "S": Starting point
# "F": Finishing point
# "#": Wall
# "*": Track list

def find_position(track, char):
    position_list= []
    for i, index in enumerate(track):
        for j , character in enumerate(index):
            if character ==char:
                position_list.append((i,j))
    return position_list

# It set up the state of the track list.
# It returns the list that exclude wall state.
# It also returns the velocity of the car at any given time is limited to -5 to +5.
# It also returns the actions state of car. {-1, 0, +1}  

def set_up_board_state(track_list):
    not_wall_state = []
    for row_index, row in enumerate(track_list):
        for col_index, col in enumerate(row):
            if track_list[row_index][col_index] != '#':
                not_wall_state.append([col_index, row_index])
    states = []
    possible_velocity_car = [i for i in range(-5, 6)]
    for loc in not_wall_state:
        for x_vel in possible_velocity_car:
            for y_vel in possible_velocity_car:
                states.append([loc[0], loc[1], x_vel, y_vel])
    actions = []
    possible_action_value = [-1, 0 , 1]
    for i in possible_action_value:
        for j in possible_action_value:
            actions.append([i, j])
    return not_wall_state, states, actions

# It checks the velocity of the car at any given time is limited to -5 to +5.
# If the velocity is below -5, it sets to -5. If the velocity is above +5, it sets to +5

def check_velocity_limit(x):
    if x>=5:
        return 5
    elif x<=-5:
        return -5
    else:
        return x


# ## Bresenham Algorithm
# Bresenham’s Line Generation Algorithm (https://www.geeksforgeeks.org/bresenhams-line-generation-algorithm/)
# It generates the line between starting (x1, y1) to (x2, y2).
#     Formula
#         dx = (x2-x1), dy = (y2-y1)
#         m = dx/dy
#         y=mx+c

def bresenham_algorithm(x1, y1, x2, y2):
    path = []
    dx = abs(x2-x1)
    dy = abs(y2-y1)
    slope_x =1 if x1<x2 else slope_x = -1
    slope_y =1 if y1<y2 else slope_y = -1
    error = dx-dy
    while True:
        path.append((x1,y1))
        if x1 == x2 and y1 == y2:
            return path
        pk = 2*error
        if pk > -dy:
            error = error - dy
            x1 = x1 + slope_x
        if pk < dx:
            error = eror + dx
            y1 = y1 + slope_y 

# Perform action
# It updates the states(x, y, velocity of x, velocity of y)
# It computes the possible path from starting (x,y) to (end x, end y) using bresenham algorithm.
# It checks whether the possible path contains either finishing points or wall states.
# If it is finishing poitns, it returns finishing x, y and sets velocity to 0.
# If it is in wall, it checks the crash type. It sets velocity to 0.
#    If the crash type is soft crash, the position returns nearest position on the track to the place where it crashed.
#    If the crash type is harsh crash, position is set back to the original starting position
# It returns updated x,y, velocity of x, velocity of y.

def perform_action(x, y, vel_x, vel_y, finish_locs, wall_locs, track_list, crash_type):
    end_x = x + vel_x
    end_y = y + vel_y
    path = bresenham_algorithm(x, y, end_x, end_y)
    finish_detect = False
    crash_detect=False
    for p in path:
        if (p[1], p[0]) in finish_locs:
            finish_detect = True
            path_point = p
            break
        elif (p[1], p[0]) in wall_locs:
            crash_detect=True
            break
        path_point = p
    if finish_detect:
        x, y, vel_x, vel_y = path_point[0], path_point[1], 0, 0
    elif crash_detect and crash_type == 'soft_crash':
        x, y, vel_x, vel_y = path_point[0], path_point[1], 0, 0
    elif crash_detect and crash_type == 'harsh_crash':
        random_start = random.choice(start_locs)
        x, y, vel_x, vel_y = random_start[1], random_start[0], 0, 0
    else:
        x+=vel_x
        y+=vel_y
    return [x, y, vel_x, vel_y]


# ## Value Iteration Algorithm

# Value Iteration
#     It keeps the copy of the v values.
#     It iterates all posible states
#     It performs actions based on x,y, velocity. 
#         If it reaches to finishing position, it returns finishing reward
#         If it doesn't reach to finishing position, it udpates the reward
#     It iterates all possible actions
#     It computes the case when acceleration attempt is successful.
#          Acceleration = velocity + acceleration
#          It checks whether the acceleration is within -5~+5
#     It performs actions based on the x,y and updated velocity. 
#         If it reaches to finishing position, it returns finishing reward
#         If it doesn't reach to finishing position, it udpates the reward
#     For this assignment, % of successful attempt is 80% , and 20& of unsuccessful attempt
#     It computes action reward of the action from successful attempt and unsuccessful attempt, then multiply with learning factor to comptues v value.
#     It updates v value if it finds better path. 
#  After it iterates, it computes for delta (current v value - previous v value)
#  If the delta meets error threshold, it breaks.     

def value_iteration(states, finish_locs, epsilon, actions, v_vals, finish_locs, wall_locs, track_list, learning_rate, crash_type):
    deltas = []
    delta = epsilon + 1
    while delta > epsilon:
        v_vals_prev = v_vals.copy()
        for state_index, state in enumerate(states):
            max_reward = -math.inf
            x, y, vel_x, vel_y = state[0], state[1], state[2], state[3]
            acc_state_unsuccess = perform_action(x, y, vel_x, vel_y, finish_locs, wall_locs,track_list, crash_type) 
            if (acc_state_unsuccess[1],acc_state_unsuccess[0]) in finish_locs:
                unsuccess_reward = 1
            else:
                for unsuccess_index, unsuccess_state in enumerate(states):
                    if unsuccess_state == acc_state_unsuccess:
                        unsuccess_reward = v_vals_prev[unsuccess_index]
            for a in actions:
                acc_x_attempt = vel_x + a[0]
                acc_y_attempt = vel_y + a[1]
                action_v_x = check_velocity_limit(acc_x_attempt)
                action_v_y = check_velocity_limit(acc_y_attempt)
                acc_state_success = perform_action(x, y,action_v_x, action_v_y, finish_locs, wall_locs, track_list, crash_type) 
                if (acc_state_success[1],acc_state_success[0]) in finish_locs:
                    success_reward = 1
                else:
                    for success_index, success_state in enumerate(states):
                        if success_state == acc_state_success:
                            success_reward = v_vals_prev[success_index]
                action_reward = 0.2*unsuccess_reward + 0.8*success_reward
                if action_reward > max_reward:
                    max_reward = action_reward
            if v_vals_prev[state_index] < max_reward *learning_rate:
                v_vals[state_index] = max_reward *learning_rate
            else:
                v_vals[state_index] = v_vals_prev[state_index]   
        delta =0
        for v_val_index, v_val in enumerate(v_vals):
            delta += abs(v_vals[v_val_index] - v_vals_prev[v_val_index])
        print(delta)
        deltas.append(delta)
    return v_vals, deltas

# Value iteration Testing
# This is the algorithm that tests value iteration on track. 
# It process almost same method as normal value iteration. 
# For this assignment, % of successful attempt is 80% , and 20& of unsuccessful attempt
# It keeps iterationg the states based on the v value that is updated in value iteration algorithm.
# If it finds the finishing points, it returns final points, and break the loop.
# If it doens't find the finishing points, it moves to next state. 

def value_iteration_testing(start_locs, states, v_vals, finish_locs, wall_locs, track_list, crash_type):
    reward = -1
    starting_points  = random.choice(start_locs)
    state =starting_points[1], starting_points[0], 0,0 
    max_reward = -math.inf
    steps =0
    step_list =[]
    while reward < 0:
        print(state)
        step_list.append(state)
        x, y, vel_x, vel_y = state[0], state[1], state[2], state[3]
        acc_state_unsuccess = perform_action(x, y, vel_x, vel_y, finish_locs, wall_locs, track_list, crash_type) 
        if (acc_state_unsuccess[1],acc_state_unsuccess[0]) in finish_locs:
            unsuccess_reward = 1
        else:
            for unsuccess_index, unsuccess_state in enumerate(states):
                if unsuccess_state == acc_state_unsuccess:
                    unsuccess_reward = v_vals[unsuccess_index]
        for action in actions:
            action_v_x = vel_x +action[0]
            action_v_y = vel_y + action[1]
            action_v_x = check_velocity_limit(action_v_x)
            action_v_y = check_velocity_limit(action_v_y)
            acc_state_success = perform_action(x, y,action_v_x, action_v_y, finish_locs, wall_locs, track_list, crash_type) 
            if (acc_state_success[1],acc_state_success[0]) in finish_locs:
                success_reward = 1
            else:
                for success_index, success_state in enumerate(states):
                    if success_state == acc_state_success:
                        success_reward = v_vals[success_index]
            action_reward = 0.2 * unsuccess_reward + 0.8 * success_reward
            if action_reward > max_reward:
                max_reward = action_reward
                best_action = action
        if random.random() <= 0.8:
            vel_x = vel_x + best_action[0]
            vel_y = vel_y + best_action[1]
            vel_x  = check_velocity_limit(vel_x )
            vel_y = check_velocity_limit(vel_y)
        else:
            vel_x = vel_x
            vel_y = vel_y
        move_state =  perform_action(x, y, vel_x, vel_y, finish_locs, wall_locs, track_list, crash_type)

        if (move_state[1], move_state[0]) in finish_locs: 
            reward = 1
            step_list.append(move_state)
            break
        else:
            steps +=1
            reward = -1
            state = move_state
    return steps, step_list


# ## Q Learning Algorithm

# Epsilon Greedy Policy
# It is use to find the optimal balance between exploration and exploitation.
# If the random probability is smaller than epsilon, it takes the random action.
# If the random probability is larger than epsilon, it takes current best action.

def epsilon_greedy_policy(state, epsilon, q_vals):
    if random.random() < epsilon:
        action = random.randint(0,8)
    else:
        action = np.argmax(q_vals[state,:])
    return action

# Q Learning
#     It randomly select the starting points
#     Until it reaches the terminal states
#       it choose best action using epsilon greedy policy for given state
#       For this assignment, % of successful attempt is 80% , and 20& of unsuccessful attempt
#       It performs action and observe reward and new state
#       It updates the q value using this formula
#          Q(s,a) = Q(s,a)+alpha(r+learning rate maximum action Q(s', a') - Q(s,a))
#       It updates the current state.
#     It repeats these process until it reaches to maximum number of iteration. 

def q_learning(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type, alpha, learning_rate, epsilon, total_episodes):
    step_list =[]
    for i in range(total_episodes):
        start_pos = random.choice(start_locs)
        state = [start_pos[1], start_pos[0], 0, 0]
        for state_index_prev, state_prev in enumerate(states):
            if state_prev == state:
                state_index = state_index_prev
        reward = -1
        steps = 0
        while reward < 0:
            a_index = epsilon_greedy_policy(state_index, epsilon, q_vals)
            action  = actions[a_index]
            if random.uniform(0, 1) <= 0.8:
                vel_x = state[2]+action[0]
                vel_y = state[3]+action[1]
                vel_x_check = check_velocity_limit(vel_x)
                vel_y_check = check_velocity_limit(vel_y)
            else:
                vel_x = state[2]
                vel_y = state[3]
                vel_x_check = check_velocity_limit(vel_x)
                vel_y_check = check_velocity_limit(vel_y)
            next_x, next_y, next_vel_x, next_vel_y = perform_action(state[0], state[1], vel_x_check, vel_y_check, finish_locs, wall_locs, track_list,crash_type)
            next_state_list = [next_x, next_y, next_vel_x, next_vel_y ]
            for state_index_next, state_next in enumerate(states):
                if state_next == next_state_list:
                    next_state_index = state_index_next
            if (next_y, next_x) in finish_locs: 
                reward = 1
            else:
                reward = -1
            steps +=1
            q_vals[state_index, a_index] = (1-alpha)*q_vals[state_index, a_index]+alpha*(reward+learning_rate*np.max(q_vals[next_state_index]) - q_vals[state_index, a_index])
            state = next_state_list
            state_index = next_state_index
        print('Currnet Episode: ', i)
        step_list.append(steps)
    return q_vals, step_list

# Q Learning Test
#   This is the algorithm that tests Q Learning on track.
#   Using updated q value, it process almost same method as normal Q Learning.
#   For this assignment, % of successful attempt is 80% , and 20& of unsuccessful attempt
#   It performs action state(x,y) and velocity. 
#   If it finds the finishing points, it returns number of steps and break the loop
#   If it doesn't find the finishing pionts, it moves and updates the state. 

def q_learning_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon):
    start_pos = random.choice(start_locs)
    state = [start_pos[1], start_pos[0], 0, 0]
    for state_index_prev, state_prev in enumerate(states):
        if state_prev == state:
            state_index = state_index_prev
    reward = -1
    moves_list = []
    moves = 0
    while reward < 0:
        print(state)
        a = epsilon_greedy_policy(state_index, epsilon, q_vals)
        action = actions[a]
        if random.uniform(0, 1) <= 0.8:
            vel_x = state[2]+action[0]
            vel_y = state[3]+action[1]
            vel_x_check = check_velocity_limit(vel_x)
            vel_y_check = check_velocity_limit(vel_y)
        else:
            vel_x = state[2]
            vel_y = state[3]
            vel_x_check = check_velocity_limit(vel_x)
            vel_y_check = check_velocity_limit(vel_y)
        new_x, new_y, new_vel_x, new_vel_y = perform_action(state[0], state[1], vel_x_check, vel_y_check, finish_locs, wall_locs, track_list, crash_type)
        new_state = [new_x, new_y, new_vel_x, new_vel_y]
        for state_index_new, state_new in enumerate(states):
            if state_new == new_state:
                next_new_state_index = state_index_new  
        if (new_y, new_x) in finish_locs: 
            reward = 1
            break
        else:
            reward = -1
        moves +=1
        state = new_state
        state_index = next_new_state_index
    return moves


# ## SARSA Algorithm

# SARSA
#     It randomly select the starting points
#     It choose best action using epsilon greedy policy for given state
#     Until it reaches the terminal states
#       For this assignment, % of successful attempt is 80% , and 20& of unsuccessful attempt
#       It performs action and observe reward and new state
#       It updates the q value using this formula
#          Q(s,a) = Q(s,a)+alpha(r+learning factor*Q(s', a') - Q(s,a))
#       It updates the current state and actions.
#     It repeats these process until it reaches to maximum number of iteration. 

def SARSA(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type, alpha, learning_rate, epsilon, total_episodes):
    step_list =[]
    for i in range(total_episodes):
        start_pos = random.choice(start_locs)
        state = [start_pos[1], start_pos[0], 0, 0]
        for state_index_prev, state_prev in enumerate(states):
            if state_prev == state:
                state_index = state_index_prev     
        a_index = epsilon_greedy_policy(state_index, epsilon, q_vals)
        action  = actions[a_index]
        reward = -1
        steps = 0
        while reward < 0:
            if random.uniform(0, 1) <= 0.8:
                vel_x = state[2]+action[0]
                vel_y = state[3]+action[1]
                vel_x_check = check_velocity_limit(vel_x)
                vel_y_check = check_velocity_limit(vel_y)
            else:
                vel_x = state[2]
                vel_y = state[3]
                vel_x_check = check_velocity_limit(vel_x)
                vel_y_check = check_velocity_limit(vel_y)
            next_x, next_y, next_vel_x, next_vel_y = perform_action(state[0], state[1], vel_x_check, vel_y_check, finish_locs, wall_locs, track_list, crash_type)
            next_state_list = [next_x, next_y, next_vel_x, next_vel_y ]
            for state_index_next, state_next in enumerate(states):
                if state_next == next_state_list:
                    next_state_index = state_index_next 
            if (next_y, next_x) in finish_locs: 
                reward = 1
            else:
                reward = -1
            steps +=1
            new_a_index = epsilon_greedy_policy(next_state_index, epsilon, q_vals)
            new_action  = actions[new_a_index]
            q_vals[state_index, a_index] = (1-alpha)*q_vals[state_index, a_index]+alpha*(reward+learning_rate*q_vals[next_state_index, new_a_index] - q_vals[state_index, a_index])
            state = next_state_list
            state_index = next_state_index
            action = new_action
            a_index = new_a_index
        print('Episode Steps', i)
        step_list.append(steps)
    return q_vals, step_list

# SARSA Test
#     This is the algorithm that tests SARSA on track.
#     It randomly select the starting points
#     It choose best action using epsilon greedy policy for given state
#     Using updated q value, it process almost same method as normal SARSA.
#     For this assignment, % of successful attempt is 80% , and 20& of unsuccessful attempt
#     It performs action state(x,y) and velocity. 
#     If it finds the finishing points, it returns number of steps and break the loop
#     If it doesn't find the finishing pionts, it moves and updates the state. 

def SARSA_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon):
    start_pos = random.choice(start_locs)
    state = [start_pos[1], start_pos[0], 0, 0]
    for state_index_prev, state_prev in enumerate(states):
        if state_prev == state:
            state_index = state_index_prev
    a_index = epsilon_greedy_policy(state_index, epsilon, q_vals)
    action = actions[a_index]
    reward = -1
    moves_list = []
    moves = 0
    while reward < 0:
        print(state)
        if random.uniform(0, 1) <= 0.8:
            vel_x = state[2]+action[0]
            vel_y = state[3]+action[1]
            vel_x_check = check_velocity_limit(vel_x)
            vel_y_check = check_velocity_limit(vel_y)
        else:
            vel_x = state[2]
            vel_y = state[3]
            vel_x_check = check_velocity_limit(vel_x)
            vel_y_check = check_velocity_limit(vel_y)
        new_x, new_y, new_vel_x, new_vel_y = perform_action(state[0], state[1], vel_x_check, vel_y_check, finish_locs, wall_locs, track_list,crash_type)
        new_state = [new_x, new_y, new_vel_x, new_vel_y]
        for state_index_new, state_new in enumerate(states):
            if state_new == new_state:
                next_new_state_index = state_index_new    
        if (new_y, new_x) in finish_locs: 
            reward = 1
            break
        else:
            reward = -1
        moves +=1
        new_a_index = epsilon_greedy_policy(next_new_state_index, epsilon, q_vals)
        new_action  = actions[new_a_index]
        state = new_state
        state_index = next_new_state_index
        action = new_action
        a_index = new_a_index
    return moves


# # Value Iteration

# ## Value Iteration L-track Soft Crash

track_matrix = open_dataset('L-track')
track_list= track_nplist(track_matrix)
start_locs = find_position(track_list, 'S')
finish_locs = find_position(track_list, 'F')
wall_locs = find_position(track_list, '#')
not_wall_state, states, actions = set_up_board_state(track_list)
v_vals = [0 for i in range(len(states))]
for v_index, v in enumerate(v_vals):
    state_pos = (states[v_index][1], states[v_index][0])
    if (state_pos) in finish_locs:
        v_vals[v_index] = 1

crash_type ='soft_crash'
learning_rate = 0.9
epsilon = 0.01
v_vals, deltas = value_iteration(states, finish_locs, epsilon, actions, v_vals, finish_locs, wall_locs, track_list, learning_rate, crash_type)
step, step_list = value_iteration_testing(start_locs, states, v_vals, finish_locs, wall_locs, track_list, crash_type)
total_moves_list = []
for i in range(10):
    total_moves = value_iteration_testing(start_locs, states, v_vals, finish_locs, wall_locs, track_list, crash_type)[0]
    total_moves_list.append(total_moves)
print('average moves: ', int(np.mean(total_moves_list)))


# ## Value Iteration L-track Harsh Crash

track_matrix = open_dataset('L-track')
track_list= track_nplist(track_matrix)
start_locs = find_position(track_list, 'S')
finish_locs = find_position(track_list, 'F')
wall_locs = find_position(track_list, '#')
not_wall_state, states, actions = set_up_board_state(track_list)
v_vals = [0 for i in range(len(states))]
for v_index, v in enumerate(v_vals):
    state_pos = (states[v_index][1], states[v_index][0])
    if (state_pos) in finish_locs:
        v_vals[v_index] = 1

crash_type ='harsh_crash'
learning_rate = 0.9
epsilon = 0.01
v_vals, deltas = value_iteration(states, finish_locs, epsilon, actions, v_vals, finish_locs, wall_locs, track_list, learning_rate, crash_type)
step, step_list = value_iteration_testing(start_locs, states, v_vals, finish_locs, wall_locs, track_list, crash_type)
total_moves_list = []
for i in range(10):
    total_moves = value_iteration_testing(start_locs, states, v_vals, finish_locs, wall_locs, track_list, crash_type)[0]
    total_moves_list.append(total_moves)
print('average moves: ', int(np.mean(total_moves_list)))


# ## Value Iteration O-track Soft Crash

track_matrix = open_dataset('O-track')
track_list= track_nplist(track_matrix)
start_locs = find_position(track_list, 'S')
finish_locs = find_position(track_list, 'F')
wall_locs = find_position(track_list, '#')
not_wall_state, states, actions = set_up_board_state(track_list)
v_vals = [0 for i in range(len(states))]
for v_index, v in enumerate(v_vals):
    state_pos = (states[v_index][1], states[v_index][0])
    if (state_pos) in finish_locs:
        v_vals[v_index] = 1

crash_type ='soft_crash'
learning_rate = 0.9
epsilon = 0.01
v_vals, deltas = value_iteration(states, finish_locs, epsilon, actions, v_vals, finish_locs, wall_locs, track_list, learning_rate, crash_type)
step, step_list = value_iteration_testing(start_locs, states, v_vals, finish_locs, wall_locs, track_list, crash_type)
total_moves_list = []
for i in range(10):
    total_moves = value_iteration_testing(start_locs, states, v_vals, finish_locs, wall_locs, track_list, crash_type)[0]
    total_moves_list.append(total_moves)
print('average moves: ', int(np.mean(total_moves_list)))


# ## Value Iteration O-track Harsh Crash

track_matrix = open_dataset('O-track')
track_list= track_nplist(track_matrix)
start_locs = find_position(track_list, 'S')
finish_locs = find_position(track_list, 'F')
wall_locs = find_position(track_list, '#')
not_wall_state, states, actions = set_up_board_state(track_list)
v_vals = [0 for i in range(len(states))]
for v_index, v in enumerate(v_vals):
    state_pos = (states[v_index][1], states[v_index][0])
    if (state_pos) in finish_locs:
        v_vals[v_index] = 1

crash_type ='harsh_crash'
learning_rate = 0.9
epsilon = 0.01
v_vals, deltas = value_iteration(states, finish_locs, epsilon, actions, v_vals, finish_locs, wall_locs, track_list, learning_rate, crash_type)
step, step_list = value_iteration_testing(start_locs, states, v_vals, finish_locs, wall_locs, track_list, crash_type)
total_moves_list = []
for i in range(10):
    total_moves = value_iteration_testing(start_locs, states, v_vals, finish_locs, wall_locs, track_list, crash_type)[0]
    total_moves_list.append(total_moves)
print('average moves: ', int(np.mean(total_moves_list)))


# ## Value Iteration R-track Soft Crash

track_matrix = open_dataset('R-track')
track_list= track_nplist(track_matrix)
start_locs = find_position(track_list, 'S')
finish_locs = find_position(track_list, 'F')
wall_locs = find_position(track_list, '#')
not_wall_state, states, actions = set_up_board_state(track_list)
v_vals = [0 for i in range(len(states))]
for v_index, v in enumerate(v_vals):
    state_pos = (states[v_index][1], states[v_index][0])
    if (state_pos) in finish_locs:
        v_vals[v_index] = 1

crash_type ='soft_crash'
learning_rate = 0.9
epsilon = 0.01
v_vals, deltas = value_iteration(states, finish_locs, epsilon, actions, v_vals, finish_locs, wall_locs, track_list, learning_rate, crash_type)
step, step_list = value_iteration_testing(start_locs, states, v_vals, finish_locs, wall_locs, track_list, crash_type)
total_moves_list = []
for i in range(10):
    total_moves = value_iteration_testing(start_locs, states, v_vals, finish_locs, wall_locs, track_list, crash_type)[0]
    total_moves_list.append(total_moves)
print('average moves: ', int(np.mean(total_moves_list)))


# ## Value Iteration R-track Harsh Crash

track_matrix = open_dataset('R-track')
track_list= track_nplist(track_matrix)
start_locs = find_position(track_list, 'S')
finish_locs = find_position(track_list, 'F')
wall_locs = find_position(track_list, '#')
not_wall_state, states, actions = set_up_board_state(track_list)
v_vals = [0 for i in range(len(states))]
for v_index, v in enumerate(v_vals):
    state_pos = (states[v_index][1], states[v_index][0])
    if (state_pos) in finish_locs:
        v_vals[v_index] = 1

crash_type ='harsh_crash'
learning_rate = 0.9
epsilon = 0.01
v_vals, deltas = value_iteration(states, finish_locs, epsilon, actions, v_vals, finish_locs, wall_locs, track_list, learning_rate, crash_type)
step, step_list = value_iteration_testing(start_locs, states, v_vals, finish_locs, wall_locs, track_list, crash_type)
total_moves_list = []
for i in range(10):
    total_moves = value_iteration_testing(start_locs, states, v_vals, finish_locs, wall_locs, track_list, crash_type)[0]
    total_moves_list.append(total_moves)
print('average moves: ', int(np.mean(total_moves_list)))


# # Q Learning

# ## Q Learning L-track Soft Crash

track_matrix = open_dataset('L-track')
track_list= track_nplist(track_matrix)
start_locs = find_position(track_list, 'S')
finish_locs = find_position(track_list, 'F')
wall_locs = find_position(track_list, '#')
not_wall_state, states, actions = set_up_board_state(track_list)
q_vals = np.zeros((len(states), len(actions)))

epsilon = 0.1
total_episodes = 500
alpha = 0.1
learning_rate = 0.9
crash_type = 'soft_crash'
q_vals, step_list = q_learning(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type, alpha, learning_rate, epsilon, total_episodes)
total_moves = q_learning_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
total_moves_list = []
for i in range(10):
    total_moves =q_learning_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
    #print(total_moves)
    total_moves_list.append(total_moves)
print('average moves: ', np.mean(total_moves_list))


# ## Q Learning L-track Harsh Crash

track_matrix = open_dataset('L-track')
track_list= track_nplist(track_matrix)
start_locs = find_position(track_list, 'S')
finish_locs = find_position(track_list, 'F')
wall_locs = find_position(track_list, '#')
not_wall_state, states, actions = set_up_board_state(track_list)
q_vals = np.zeros((len(states), len(actions)))

epsilon = 0.1
total_episodes = 500
alpha = 0.1
learning_rate = 0.9
crash_type = 'harsh_crash'
q_vals, step_list = q_learning(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type, alpha, learning_rate, epsilon, total_episodes)
total_moves = q_learning_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
total_moves_list = []
for i in range(10):
    total_moves =q_learning_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
    #print(total_moves)
    total_moves_list.append(total_moves)
print('average moves: ', np.mean(total_moves_list))


# ## Q Learning O-track Soft Crash

track_matrix = open_dataset('O-track')
track_list= track_nplist(track_matrix)
start_locs = find_position(track_list, 'S')
finish_locs = find_position(track_list, 'F')
wall_locs = find_position(track_list, '#')
not_wall_state, states, actions = set_up_board_state(track_list)
q_vals = np.zeros((len(states), len(actions)))

epsilon = 0.1
total_episodes = 500
alpha = 0.1
learning_rate = 0.9
crash_type = 'soft_crash'
q_vals, step_list = q_learning(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type, alpha, learning_rate, epsilon, total_episodes)
total_moves = q_learning_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
total_moves_list = []
for i in range(10):
    total_moves =q_learning_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
    #print(total_moves)
    total_moves_list.append(total_moves)
print('average moves: ', np.mean(total_moves_list))


# ## Q Learning O-track Harsh Crash

track_matrix = open_dataset('O-track')
track_list= track_nplist(track_matrix)
start_locs = find_position(track_list, 'S')
finish_locs = find_position(track_list, 'F')
wall_locs = find_position(track_list, '#')
not_wall_state, states, actions = set_up_board_state(track_list)
q_vals = np.zeros((len(states), len(actions)))

epsilon = 0.1
total_episodes = 500
alpha = 0.1
learning_rate = 0.9
crash_type = 'harsh_crash'
q_vals, step_list = q_learning(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type, alpha, learning_rate, epsilon, total_episodes)
total_moves = q_learning_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
total_moves_list = []
for i in range(10):
    total_moves =q_learning_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
    #print(total_moves)
    total_moves_list.append(total_moves)
print('average moves: ', np.mean(total_moves_list))


# ## Q Learning R-track Soft Crash

track_matrix = open_dataset('R-track')
track_list= track_nplist(track_matrix)
start_locs = find_position(track_list, 'S')
finish_locs = find_position(track_list, 'F')
wall_locs = find_position(track_list, '#')
not_wall_state, states, actions = set_up_board_state(track_list)
q_vals = np.zeros((len(states), len(actions)))

epsilon = 0.1
total_episodes = 500
alpha = 0.1
learning_rate = 0.9
crash_type = 'soft_crash'
q_vals, step_list = q_learning(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type, alpha, learning_rate, epsilon, total_episodes)
total_moves = q_learning_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
total_moves_list = []
for i in range(10):
    total_moves =q_learning_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
    #print(total_moves)
    total_moves_list.append(total_moves)
print('average moves: ', np.mean(total_moves_list))


# ## Q Learning R-track Harsh Crash

track_matrix = open_dataset('R-track')
track_list= track_nplist(track_matrix)
start_locs = find_position(track_list, 'S')
finish_locs = find_position(track_list, 'F')
wall_locs = find_position(track_list, '#')
not_wall_state, states, actions = set_up_board_state(track_list)
q_vals = np.zeros((len(states), len(actions)))

epsilon = 0.1
total_episodes = 500
alpha = 0.1
learning_rate = 0.9
crash_type = 'harsh_crash'
q_vals, step_list = q_learning(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type, alpha, learning_rate, epsilon, total_episodes)
total_moves = q_learning_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
total_moves_list = []
for i in range(10):
    total_moves =q_learning_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
    #print(total_moves)
    total_moves_list.append(total_moves)
print('average moves: ', np.mean(total_moves_list))


# # SARSA

# ## SARSA L-track Soft Crash

track_matrix = open_dataset('L-track')
track_list= track_nplist(track_matrix)
start_locs = find_position(track_list, 'S')
finish_locs = find_position(track_list, 'F')
wall_locs = find_position(track_list, '#')
not_wall_state, states, actions = set_up_board_state(track_list)
q_vals = np.zeros((len(states), len(actions)))

epsilon = 0.1
total_episodes = 500
alpha = 0.1
learning_rate = 0.9
crash_type ='soft_crash'
q_vals, step_list = SARSA(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type, alpha, learning_rate, epsilon, total_episodes)
total_moves = SARSA_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
total_moves_list = []
for i in range(10):
    total_moves = SARSA_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
    total_moves_list.append(total_moves)
print('average moves: ', int(np.mean(total_moves_list)))


# ## SARSA L-track Harsh Crash

track_matrix = open_dataset('L-track')
track_list= track_nplist(track_matrix)
start_locs = find_position(track_list, 'S')
finish_locs = find_position(track_list, 'F')
wall_locs = find_position(track_list, '#')
not_wall_state, states, actions = set_up_board_state(track_list)
q_vals = np.zeros((len(states), len(actions)))

epsilon = 0.1
total_episodes = 500
alpha = 0.1
learning_rate = 0.9
crash_type ='harsh_crash'
q_vals, step_list = SARSA(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type, alpha, learning_rate, epsilon, total_episodes)
total_moves = SARSA_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
total_moves_list = []
for i in range(10):
    total_moves = SARSA_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
    total_moves_list.append(total_moves)
print('average moves: ', int(np.mean(total_moves_list)))


# ## SARSA O-track Soft Crash

track_matrix = open_dataset('O-track')
track_list= track_nplist(track_matrix)
start_locs = find_position(track_list, 'S')
finish_locs = find_position(track_list, 'F')
wall_locs = find_position(track_list, '#')
not_wall_state, states, actions = set_up_board_state(track_list)
q_vals = np.zeros((len(states), len(actions)))

epsilon = 0.1
total_episodes = 500
alpha = 0.1
learning_rate = 0.9
crash_type ='soft_crash'
q_vals, step_list = SARSA(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type, alpha, learning_rate, epsilon, total_episodes)
total_moves = SARSA_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
total_moves_list = []
for i in range(10):
    total_moves = SARSA_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
    total_moves_list.append(total_moves)
print('average moves: ', int(np.mean(total_moves_list)))


# ## SARSA O-track Harsh Crash

track_matrix = open_dataset('O-track')
track_list= track_nplist(track_matrix)
start_locs = find_position(track_list, 'S')
finish_locs = find_position(track_list, 'F')
wall_locs = find_position(track_list, '#')
not_wall_state, states, actions = set_up_board_state(track_list)
q_vals = np.zeros((len(states), len(actions)))

epsilon = 0.1
total_episodes = 500
alpha = 0.1
learning_rate = 0.9
crash_type ='harsh_crash'
q_vals, step_list = SARSA(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type, alpha, learning_rate, epsilon, total_episodes)
total_moves = SARSA_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
total_moves_list = []
for i in range(10):
    total_moves = SARSA_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
    total_moves_list.append(total_moves)
print('average moves: ', int(np.mean(total_moves_list)))


# ## SARSA R-track Soft Crash

track_matrix = open_dataset('R-track')
track_list= track_nplist(track_matrix)
start_locs = find_position(track_list, 'S')
finish_locs = find_position(track_list, 'F')
wall_locs = find_position(track_list, '#')
not_wall_state, states, actions = set_up_board_state(track_list)
q_vals = np.zeros((len(states), len(actions)))

epsilon = 0.1
total_episodes = 500
alpha = 0.1
learning_rate = 0.9
crash_type ='soft_crash'
q_vals, step_list = SARSA(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type, alpha, learning_rate, epsilon, total_episodes)
total_moves = SARSA_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
total_moves_list = []
for i in range(10):
    total_moves = SARSA_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
    total_moves_list.append(total_moves)
print('average moves: ', int(np.mean(total_moves_list)))


# ## SARSA R-track Harsh Crash

track_matrix = open_dataset('R-track')
track_list= track_nplist(track_matrix)
start_locs = find_position(track_list, 'S')
finish_locs = find_position(track_list, 'F')
wall_locs = find_position(track_list, '#')
not_wall_state, states, actions = set_up_board_state(track_list)
q_vals = np.zeros((len(states), len(actions)))

epsilon = 0.1
total_episodes = 500
alpha = 0.1
learning_rate = 0.9
crash_type ='harsh_crash'
q_vals, step_list = SARSA(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type, alpha, learning_rate, epsilon, total_episodes)
total_moves = SARSA_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
total_moves_list = []
for i in range(10):
    total_moves = SARSA_test(states, start_locs, track_list, q_vals, finish_locs, wall_locs, actions, crash_type,epsilon)
    total_moves_list.append(total_moves)
print('average moves: ', int(np.mean(total_moves_list)))

