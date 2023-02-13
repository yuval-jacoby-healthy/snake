#!/bin/bash

python3 Snake.py --policies "Benshapiro(epsilon=0.5)" --game_duration 5000 --score_scope 1000 --log_file "my_log.log" --to_render 1  --player_init_time 2
