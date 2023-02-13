from policies import base_policy as bp

import numpy as np

class Group1(bp.Policy):
    
    def __init__(self, policy_args, board_size, stateq, actq, modelq, logq, id, game_duration, score_scope):
        super().__init__(policy_args, board_size, stateq, actq, modelq, logq, id, game_duration, score_scope)
        self.q_table = {}
        self.epsilon = 0.05
        self.decay = 0.1
        self.window_size = 3
        self.learning_rate = 0.1
        self.discount_rate = 0.9
    
    def cast_string_args(self, policy_args):
        return policy_args

    def init_run(self):
        self.q_table = {}
        self.buffer = Queue()
        
    def calculate_Q(self, current_state, action, next_state, reward):
        q = self.q_table[current_state][self.action_dict[[bp.Policy.ACTIONS[action]]]]
        m = self.discount_rate * np.max(self.q_table[next_state])
        return q + self.learning_rate * (reward + m * - q)
    
    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        for observation in self.buffer:
            self.q_table[observation[0]][self.action_dict[observation[1]]] = \
            self.calculate_Q(observation[0], observation[1], observation[2], observation[3])
        
        # epsilon decay update
        if round % 100 == 0:
            self.epsilon *= self.decay

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        # update
        self.buffer.put([self.get_window(prev_state), prev_action, self.get_window(new_state), reward])
        self.r_sum += reward

        # choose the next action
        if np.random.rand() < self.epsilon:
            action = np.random.choice(bp.Policy.ACTIONS)
        else:
            if new_state not in self.q_table:
                self.q_table[new_state] = np.zeros(3)
            action = bp.Policy.ACTIONS[np.argmax(self.q_table[new_state])]

        self.log(f"in round {round} using action {action}")

        return action
    
    def get_window(self, state):
        offset = self.window_size
        board = state[0]
        hr, hc = state[1][0].pos[0], state[1][0].pos[1]
        dir = state[1][1]
        row_range = range(hr - offset + 1, hr + offset)
        col_range = range(hc - offset + 1, hc + offset)

        window = np.array([row.take(col_range, mode='wrap') for row in board.take(row_range, axis=0, mode='wrap')])
        if dir == 'E':
            window = np.rot90(window)
        elif dir == 'S':
            window = np.rot90(window, k=2)
        elif dir == 'W':
            window = np.rot90(window, k=3)
        
        window = np.maximum(window, -1)
        window = np.where((window > -1) & (window < 6), 0, window)
        return window
    


