import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from collections import deque
import gym
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrioritizedReplayBuffer:
    def __init__(self, max_size: int, alpha: float = 0.6):
        if max_size <= 0:
            raise ValueError("max_size must be greater than 0")
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = alpha

    def store_transition(self, transition: tuple):
        if not isinstance(transition, tuple):
            raise ValueError("Transition must be a tuple")
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append(transition)
        self.priorities.append(max_priority)

    def sample_buffer(self, batch_size: int, beta: float = 0.4):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if len(self.buffer) < batch_size:
            logger.warning("Not enough elements in the buffer to sample")
            return None

        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[i] for i in indices]
        samples = np.array(samples, dtype=object)

        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return [np.stack(samples[:, i]) for i in range(samples.shape[1])], indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

class RBMLayer(layers.Layer):
    def __init__(self, num_hidden_units: int):
        super(RBMLayer, self).__init__()
        if num_hidden_units <= 0:
            raise ValueError("Number of hidden units must be positive")
        self.num_hidden_units = num_hidden_units

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError("RBMLayer expects input shape of length 2")
        self.rbm_weights = self.add_weight(shape=(input_shape[-1], self.num_hidden_units),
                                           initializer='glorot_uniform',
                                           trainable=True)
        self.biases = self.add_weight(shape=(self.num_hidden_units,),
                                      initializer='zeros',
                                      trainable=True)

    def call(self, inputs):
        activation = tf.matmul(inputs, self.rbm_weights) + self.biases
        return tf.nn.sigmoid(activation)

class QLearningLayer(layers.Layer):
    def __init__(self, action_space_size: int, learning_rate: float = 0.001, gamma: float = 0.99, epsilon: float = 0.1):
        super(QLearningLayer, self).__init__()
        if action_space_size <= 0:
            raise ValueError("Action space size must be positive")
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.buffer_index = 0
        self.replay_buffer = PrioritizedReplayBuffer(100000)
        self.q_network = self._build_network()
        self.target_q_network = models.clone_model(self.q_network)
        self.q_network.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss='mse')

    def _build_network(self) -> models.Sequential:
        model = models.Sequential([
            layers.Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(64, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(self.action_space_size, activation='linear', kernel_initializer='glorot_uniform')
        ])
        return model

    def call(self, state: np.ndarray) -> tf.Tensor:
        if not isinstance(state, np.ndarray):
            raise ValueError("State must be a numpy array")
        return self.q_network(state)

    def update(self, batch_size: int, beta: float = 0.4):
        data = self.replay_buffer.sample_buffer(batch_size, beta)
        if data is None:
            return
        states, actions, rewards, next_states, dones = data[0]
        indices, weights = data[1], data[2]

        target_q_values = rewards + (1 - dones) * self.gamma * np.max(self.target_q_network.predict(next_states), axis=1)
        with tf.GradientTape() as tape:
            q_values = tf.reduce_sum(self.q_network(states) * tf.one_hot(actions, self.action_space_size), axis=1)
            loss = tf.reduce_mean(weights * tf.square(target_q_values - q_values))
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.q_network.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        self.buffer_index += 1
        if self.buffer_index % 1000 == 0:
            self.target_q_network.set_weights(self.q_network.get_weights())
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        priorities = np.abs(target_q_values - q_values) + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        if not isinstance(state, np.ndarray) or not isinstance(next_state, np.ndarray):
            raise ValueError("State and next_state must be numpy arrays")
        self.replay_buffer.store_transition((state, action, reward, next_state, done))

    def choose_action(self, state: np.ndarray) -> int:
        if not isinstance(state, np.ndarray):
            raise ValueError("State must be a numpy array")
        state = np.array(state).reshape(1, -1)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space_size)
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def save_weights(self, filepath: str):
        if not filepath.endswith('.h5'):
            raise ValueError("File path must end with '.h5'")
        self.q_network.save_weights(filepath)

    def load_weights(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError("The specified file does not exist")
        self.q_network.load_weights(filepath)

def create_neural_network_model(input_dim: int, num_hidden_units: int, action_space_size: int) -> models.Model:
    if input_dim <= 0 or num_hidden_units <= 0 or action_space_size <= 0:
        raise ValueError("Input dimensions and action space size must be positive")
    input_layer = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(input_layer)
    x = layers.Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
    x_rbm = RBMLayer(num_hidden_units)(x)
    q_learning_layer = QLearningLayer(action_space_size)(x_rbm)
    model = models.Model(inputs=input_layer, outputs=q_learning_layer)
    return model

def train_model_in_bipedalwalker(env_name: str, q_learning_layer: QLearningLayer, num_episodes: int, epsilon: float = 0.1):
    try:
        env = gym.make(env_name)
    except gym.error.Error as e:
        logger.error(f"Failed to create environment {env_name}: {e}")
        raise

    for episode in range(num_episodes):
        state = env.reset()
        state = np.array(state).reshape(1, -1)
        done = False
        total_reward = 0

        while not done:
            action = q_learning_layer.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.array(next_state).reshape(1, -1)
            q_learning_layer.store_transition(state, action, reward, next_state, done)
            q_learning_layer.update(batch_size=32)
            state = next_state
            total_reward += reward

        logger.info(f'Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}')

    env.close()
    save_path = 'trained_model.h5'
    q_learning_layer.save_weights(save_path)
    logger.info(f"Model saved successfully at {save_path}")

def evaluate_model(model: models.Model, env_name: str, num_episodes: int):
    try:
        env = gym.make(env_name)
    except gym.error.Error as e:
        logger.error(f"Failed to create environment {env_name}: {e}")
        raise

    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        state = np.array(state).reshape(1, -1)
        done = False
        total_reward = 0

        while not done:
            action = model.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.array(next_state).reshape(1, -1)
            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        logger.info(f'Episode: {episode + 1}, Total Reward: {total_reward}')

    env.close()
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    logger.info(f'Average Reward over {num_episodes} episodes: {avg_reward}')
    logger.info(f'Standard Deviation of Reward: {std_reward}')
    return avg_reward, std_reward

def load_model(model_path: str) -> models.Model:
    try:
        custom_objects = {'RBMLayer': RBMLayer, 'QLearningLayer': QLearningLayer}
        model = models.load_model(model_path, custom_objects=custom_objects)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def main():
    input_dim = 24
    num_hidden_units = 128
    action_space_size = 4

    try:
        model = create_neural_network_model(input_dim, num_hidden_units, action_space_size)
    except ValueError as e:
        logger.error(f"Error creating model: {e}")
        raise

    env_name = 'BipedalWalker-v3'
    num_episodes = 1000
    epsilon = 0.1

    q_learning_layer = QLearningLayer(action_space_size)
    train_model_in_bipedalwalker(env_name, q_learning_layer, num_episodes, epsilon=epsilon)

    model_path = 'trained_model.h5'
    loaded_model = load_model(model_path)

    eval_episodes = 100
    try:
        avg_reward, std_reward = evaluate_model(loaded_model, env_name, eval_episodes)
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

    print(f'Average Reward: {avg_reward}, Standard Deviation of Reward: {std_reward}')

if __name__ == "__main__":
    main()