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
        pass
    
    def calculate_Q():
        
        

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        self.log(f"learning in round {round}")

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        a =  np.random.choice(bp.Policy.ACTIONS)
        self.log(f"in round {round} usin action {a}" )
        return a
    
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
    


