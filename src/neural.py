from src.util import as_vector, initial_positions, oponent_color, zone_map
from src.qlearn import QFunction
from src.consts import *
from keras.optimizers import Adam
from keras import datasets, layers, models, initializers
from math import ceil
import numpy as np
from keras import backend as K

# Initialization
def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

class NNQFunction(QFunction):
    def __init__(self, color):
        self.rate = 0.001
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.done = False
        self.states = []
        self.color = color
        self.model = self._init_model()

    def _init_model(self):
        in_me = layers.Input(shape=(2,))
        me = layers.Dense(4,activation='relu')(in_me)
        me_out = layers.Dense(1, activation='sigmoid')(me)

        in_opponent = layers.Input(shape=(2,))
        opponent = layers.Dense(4, activation='relu')(in_opponent)
        opponent_out = layers.Dense(1, activation='sigmoid')(opponent)
        
        merged = layers.subtract([me_out, opponent_out])
        out = layers.Dense(1, kernel_initializer=initializers.Ones(), activation='tanh')(merged)
        model = models.Model(inputs=[in_me, in_opponent], outputs=out)

        model.compile(loss='mse', optimizer=Adam(lr=self.rate)) 
        model.summary()
        return model

    def add_state(self, state, value):
        print('Saving state')
        self.clear_states()
        self.states.append((state, value))
        print('State saved')

    def clear_states(self):
        self.states = []

    def adjust(self, current_state):
        if self.states == []:
            return
        state, reward = self.states[0]
        # next_state, next_reward = self.states[i + 1]
        target = reward
        if not self.done:
            target = (self.value(current_state) - reward) * self.gamma + reward 
        target_f = [[target]] 
        self.model.fit(self._reshape(state), target_f, epochs=1, verbose=0)
        # for i, state in enumerate(self.states):
        # if i + 1 == len(self.states):
        #     break
            
    def value(self, state):
        if state[0] == 10:
            # WIN
            return 1
        if state[1] == 10:
            # LOSE
            return -1
        return self.model.predict(self._reshape(state))[0][0]
        
    def _reshape(self, state):
        me = np.asarray([[state[0] / 10, state[3] / 142]]) # normalize 
        opponent = np.asarray([[state[1] / 10, state[2] / 142]])
        # print(f'shapes {me} {opponent}')
        return [me, opponent]