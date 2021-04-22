import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Lambda, LeakyReLU
import keras.backend as K
from keras.optimizers import Adam
from collections import deque

MIN_BATCH_SIZE = 64
gamma = 0.99
MIN_TARGET_COUNTER = 3
MEMORY_SIZE = 20000
episodes = 20000
epsilon_decay = 0.00005
PRE_TRAIN_STEPS = 500
learning_rate = 0.001
TARGET_NETWORK_alpha = 0.001
leaky_relu_rate = 0.01

class Agent:
    def __init__(self):
        self.network = self.create_model()
        self.target_network = self.create_model()
        self.target_network.set_weights(self.network.get_weights())
        self.network.summary()
        self.replay_memory = deque(maxlen=MEMORY_SIZE)
        self.counter = 0

    def create_model(self):

        model = Sequential()
        model.add(Dense(110, input_shape=(144,), name="D1"))
        model.add(LeakyReLU(alpha=leaky_relu_rate, name="L1"))
        model.add(Dense(85, name="D2"))
        model.add(LeakyReLU(alpha=leaky_relu_rate, name="L2"))
        model.add(Dense(65, name="D3"))
        model.add(LeakyReLU(alpha=leaky_relu_rate, name="L3"))
        model.add(Dense(49 + 1, use_bias=False, activation="linear", name="D4"))
        model.add(Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True),
                         output_shape=(49,)))
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        return model

    def update_replay_memory(self, memory):
        self.replay_memory.append(memory)

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_BATCH_SIZE:
            return

        minibatch = np.random.sample(self.replay_memory, MIN_BATCH_SIZE)
        current_states = np.squeeze(np.array([instance[0] for instance in minibatch]))
        current_qs_list = self.network.predict(current_states)
        next_states = np.squeeze(np.array([instance[3] for instance in minibatch]))
        next_qs_list = self.target_network.predict(next_states)

        X = []
        Y = []
        for i, (current_state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:
                max_q = np.max(next_qs_list[i])
                new_q = reward + gamma * max_q
            else:
                new_q = reward

            y = current_qs_list[i]
            y[action] = new_q

            X.append(current_state)
            Y.append(y)

        self.network.fit(np.squeeze(np.array(X)), np.squeeze(np.array(Y)), batch_size=MIN_BATCH_SIZE, verbose=0,
                         shuffle=False)
        if terminal_state:
            self.counter += 1

        if self.counter > MIN_TARGET_COUNTER:
            self.target_network.set_weights(self.network.get_weights())
            self.counter = 0

    def get_qs(self, state):
        return self.network.predict(state)[0]