from policies import base_policy as bp

import numpy as np

EPSILON = 1
RADIUS = 7
conversion_dict = {"N": 21, "S": 22, "E": 23, "W": 24}
# get agruments on cast_string_args for testing
gamma = 0.99
ACTIONS = 3
action_dict = {"R": 0, "L": 1, "F":2}

LR = 0.3333
min_lr = 0.00001

epsilon_decay = 1#0.9999
min_epsilon = 0.01

class Benshapiro(bp.Policy): 
    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['lr'] = float(policy_args['lr']) if 'lr' in policy_args else LR
        policy_args['radius'] = float(policy_args['radius']) if 'radius' in policy_args else RADIUS
        policy_args['decay'] = float(policy_args['decay']) if 'decay' in policy_args else epsilon_decay

        return policy_args

    def init_run(self):
        self.r_sum = 0
        # Initialize the Q-table to 0
        self.n_observations = int(self.radius)**7
        self.Q_table = np.zeros((self.n_observations, ACTIONS))
        self.state_mapping = []

        self.radius = int(self.radius)

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        pass
        # self.log(f"learning in round {round}")

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        board, head = new_state
        head_pos, direction = head
        self.r_sum += reward
        # print(f"LR: {self.lr}, Epsilon: {self.epsilon}, decay: {self.decay}, radius: {self.radius}")
        chosen_action = None
        current_state = self.map_current_state(board, head_pos, direction)
        current_state_index = None

        if np.random.rand() < self.epsilon:
            chosen_action = self.smart_random_action(new_state)
            try:
                current_state_index = self.state_mapping.index(current_state)
            except:
                self.state_mapping.append(current_state)
                current_state_index = self.state_mapping.index(current_state)
                self.Q_table[current_state] = [0, 0, 0]
        else:
            try:
                current_state_index = self.state_mapping.index(current_state)
                chosen_action_index = np.argmax(self.Q_table[current_state_index,:])
                chosen_action = bp.Policy.ACTIONS[chosen_action_index]
            except:
                self.state_mapping.append(current_state)
                current_state_index = self.state_mapping.index(current_state)
                self.Q_table[current_state] = [0, 0, 0]
                chosen_action = self.smart_random_action(new_state)
        
        self.learn_from_act(current_state_index, chosen_action, new_state, reward)

        self.epsilon = max(min_epsilon, self.decay * self.epsilon)
        self.lr = max(min_lr, self.decay * self.lr)

        # if round %500 == 0:
        #     print(self.Q_table)
        return chosen_action

    def learn_from_act(self, current_state_index, chosen_action, old_state, reward):
        new = (1-self.lr) * self.Q_table[current_state_index, action_dict[chosen_action]]
        next_state = self.next_relative_state(chosen_action, old_state)
        try:
            next_state_index = self.state_mapping.index(next_state)
        except:
            self.state_mapping.append(next_state)
            next_state_index = self.state_mapping.index(next_state)
            self.Q_table[next_state_index] = [0, 0, 0]
        new += self.lr*(reward + gamma*max(self.Q_table[next_state_index,:]))
        self.Q_table[current_state_index, action_dict[chosen_action]] = new

    def smart_random_action(self, new_state):
        board, head = new_state
        head_pos, direction = head

        # self.log(f"execute random choice. Current epsilon {self.epsilon}")
        # TODO: check if should optimaize - run over the other possibilies after the initial random action
        counter = 0
        while True:
            counter += 1 
            # while next random action it not resulting in death
            random_action = np.random.choice(bp.Policy.ACTIONS)

            # get a Position object of the position in the relevant direction from the head:
            next_position = head_pos.move(bp.Policy.TURNS[direction][random_action])
            r = next_position[0]
            c = next_position[1]

            # look at the board in the relevant position:
            if board[r, c] > 5 or board[r, c] < 0 or counter > 4:
                # self.log(f"{board[r, c]=}")
                return random_action

    def next_relative_state(self, chosen_action, old_state):
        board, head = old_state
        previous_head_pos, previous_direction = head

        new_direction = None
        if previous_direction == "N":
            if chosen_action == "R":
                new_direction = "E"
            if chosen_action == "L":
                new_direction = "W"
        if previous_direction == "S":
            if chosen_action == "R":
                new_direction = "W"
            if chosen_action == "L":
                new_direction = "E"
        if previous_direction == "W":
            if chosen_action == "R":
                new_direction = "N"
            if chosen_action == "L":
                new_direction = "S"
        if previous_direction == "E":
            if chosen_action == "R":
                new_direction = "S"
            if chosen_action == "L":
                new_direction = "N"

        if chosen_action == "F":
            new_direction = previous_direction

        # get a Position object of the position in the relevant direction from the head:
        next_position = previous_head_pos.move(bp.Policy.TURNS[new_direction][chosen_action])
        r = next_position[0]
        c = next_position[1]

        return self.map_current_state(board, (r,c), new_direction)

    def relative_board(self, board, head_pos, direction):
        # print(board)
        # print(head_pos[0], head_pos[1])
        # print(direction)

        # need to take into acount walls
        # copy over the board with self.radius from right side to left
        extended_board = np.concatenate( (board[:, board.shape[1] - (self.radius-1):], board), axis=1)
        # copy over the board with self.radius from left side to right
        extended_board = np.concatenate( (extended_board, board[:, :self.radius-1]), axis=1)
        # copy over the board with radius from bottom to top
        extended_board_copy = extended_board.copy()
        extended_board = np.concatenate( (extended_board[extended_board.shape[0] - (self.radius-1):, :], extended_board), axis=0)
        # copy over the board with radius from top to bottom
        extended_board = np.concatenate( (extended_board, extended_board_copy[:self.radius-1, :]), axis=0)

        extended_head_pos = [0,0]
        extended_head_pos[0] = head_pos[0] + self.radius - 1
        extended_head_pos[1] = head_pos[1] + self.radius - 1

        relative_board = extended_board[extended_head_pos[0] - (self.radius - 2): extended_head_pos[0] + (self.radius - 1), extended_head_pos[1] - (self.radius - 2): extended_head_pos[1] + (self.radius - 1)]
        # print(relative_board)
        return relative_board

    def map_current_state(self, board, head_pos, direction):
        relative_board = self.relative_board(board, head_pos, direction)
        converted_relative_board = relative_board.copy()
        converted_relative_board[(converted_relative_board >= 0) | (converted_relative_board <=5)] = -2
        converted_relative_board[self.radius -2, self.radius -2] = conversion_dict[direction]
        return list(converted_relative_board.flatten())
