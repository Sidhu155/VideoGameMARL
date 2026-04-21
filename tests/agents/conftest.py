import pytest
import numpy as np

parametrize_learn_bool = pytest.mark.parametrize('learning', [True, False])

parametrize_final_reward = pytest.mark.parametrize('rewards, expected_record', [
    ([0, 1, 0, 1, 0], [0, 1, 0, 1, 0]),
    (['h'], []),
    (["hello"], []),
    ([1.0, 3.5, 3, 2.4], [1.0, 3.5, 3, 2.4]),
    ([1.0, "h", 3.4, "s"], [1.0, 3.4])
])

parametrize_get_action = pytest.mark.parametrize('mask, expected_max, expected_rand', [
    (np.array([1, 1, 1, 1], dtype='int8'), 3, 0),
    (np.array([0, 1, 1, 1], dtype='int8'), 3, 1),
    (np.array([0, 0, 1, 1], dtype='int8'), 3, 2),
    (np.array([0, 0, 0, 1], dtype='int8'), 3, 3),
    (np.array([0, 1, 1, 0], dtype='int8'), 1, 1),
    (np.array([1, 1, 0, 0], dtype='int8'), 1, 0),
    (np.array([1, 0, 1, 0], dtype='int8'), 0, 0),
    
])

parametrize_epsilon = pytest.mark.parametrize('epsilon', [0.6, 0.9])

parametrize_q_table = pytest.mark.parametrize('obs, utilities, action, action_util, max_util, new_util', [
    (np.array([0, 1, 1, 0]), np.array([0.0, 1.0, -1.0, 3.0]), 0, 0.0, 3.0, 10.0),
    (np.array([0, 1, 1, 0]), np.array([0.0, 1.0, -1.0, 3.0]), 1, 1.0, 3.0, 100.0),
    (np.array([0, 1, 1, 0]), np.array([0.0, 1.0, -1.0, 3.0]), 2, -1.0, 3.0, -1000.0),
    (np.array([0, 1, 1, 0]), np.array([0.0, 1.0, -1.0, 3.0]), 3, 3.0, 3.0, 0.573645),
])

parametrize_seed_expected_max_action = pytest.mark.parametrize('seed, expected_action', [
    (1, 1),
    (5, 2),
    (10, 0),
    (11, 3)
])