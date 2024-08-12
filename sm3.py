import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from collections import deque
import gym
import logging
from typing import Tuple, List, Union
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

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

    def store_transition(self, transition: Tuple):
        if not isinstance(transition, tuple):
            raise ValueError("Transition must be a tuple")
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append(transition)
        self.priorities.append(max_priority)

    def sample_buffer(self, batch_size: int, beta: float = 0.4) -> Union[None, Tuple[List[np.ndarray], np.ndarray, np.ndarray]]:
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

    def update_priorities(self, batch_indices: np.ndarray, batch_priorities: np.ndarray):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

class RBMLayer(layers.Layer):
    def __init__(self, num_hidden_units: int, trainable=True, **kwargs):
        super(RBMLayer, self).__init__(trainable=trainable, **kwargs)
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

class QLearningAgent:
    def __init__(self, input_dim: int, num_hidden_units: int, action_space_size: int, num_attention_heads: int, learning_rate: float = 0.001, gamma: float = 0.99, epsilon: float = 0.1, min_epsilon: float = 0.01, epsilon_decay: float = 0.995, buffer_size: int = 100000):
        if input_dim <= 0 or action_space_size <= 0 or num_hidden_units <= 0 or num_attention_heads <= 0:
            raise ValueError("Input dimensions, hidden units, attention heads, and action space size must be positive")
        self.input_dim = input_dim
        self.num_hidden_units = num_hidden_units
        self.action_space_size = action_space_size
        self.num_attention_heads = num_attention_heads
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        self.q_network = self._build_network()
        self.target_q_network = models.clone_model(self.q_network)
        self.q_network.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        self.update_target_network()

    def _build_network(self) -> models.Model:
        """
        Build the Q-Network with integrated RBM and Attention layers.
        """
        inputs = layers.Input(shape=(self.input_dim,))
        
        x = layers.Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.01))(inputs)
        x = layers.Dense(64, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.01))(x)
        
        # Integrate RBMLayer
        rbm_output = RBMLayer(self.num_hidden_units)(x)
        
        # Integrate Multi-Head Attention Block
        attention_output = layers.MultiHeadAttention(num_heads=self.num_attention_heads, key_dim=self.num_hidden_units)(rbm_output, rbm_output)
        
        # Adding a normalization layer after attention
        attention_output = layers.LayerNormalization()(attention_output)
        
        # Combine outputs
        combined = layers.Concatenate()([rbm_output, attention_output])
        
        # Final dense layers for action value prediction
        x = layers.Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.01))(combined)
        outputs = layers.Dense(self.action_space_size, activation='linear', kernel_initializer='glorot_uniform')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def update_target_network(self):
        """
        Updates the target Q-network with the current Q-network's weights.
        """
        self.target_q_network.set_weights(self.q_network.get_weights())

    def update(self, batch_size: int, beta: float = 0.4):
        """
        Updates the Q-network using a batch of transitions sampled from the replay buffer.
        """
        data = self.replay_buffer.sample_buffer(batch_size, beta)
        if data is None:
            return
        states, actions, rewards, next_states, dones = data[0]
        indices, weights = data[1], data[2]

        # Compute target Q-values
        target_q_values = rewards + (1 - dones) * self.gamma * np.max(self.target_q_network.predict(next_states), axis=1)
        with tf.GradientTape() as tape:
            q_values = tf.reduce_sum(self.q_network(states) * tf.one_hot(actions, self.action_space_size), axis=1)
            loss = tf.reduce_mean(weights * tf.square(target_q_values - q_values))
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.q_network.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        # Update priorities in the replay buffer
        priorities = np.abs(target_q_values - q_values) + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)

        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Store a transition in the replay buffer.
        """
        if not isinstance(state, np.ndarray) or not isinstance(next_state, np.ndarray):
            raise ValueError("State and next_state must be numpy arrays")
        self.replay_buffer.store_transition((state, action, reward, next_state, done))

    def choose_action(self, state: np.ndarray) -> int:
        """
        Choose an action based on the current Q-network and epsilon-greedy strategy.
        """
        if not isinstance(state, np.ndarray):
            raise ValueError("State must be a numpy array")
        state = state.reshape(1, -1)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space_size)
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def save_weights(self, filepath: str):
        """
        Save the Q-network's weights to the specified file path.
        """
        if not filepath.endswith('.h5'):
            raise ValueError("File path must end with '.h5'")
        self.q_network.save_weights(filepath)

    def load_weights(self, filepath: str):
        """
        Load the Q-network's weights from the specified file path.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The specified file {filepath} does not exist")
        self.q_network.load_weights(filepath)

def preprocess_state(state: Union[np.ndarray, dict, list, tuple]) -> np.ndarray:
    """
    Ensure the state is a consistent NumPy array of float32 type.
    """
    if isinstance(state, dict):
        state = np.concatenate([np.asarray(s, dtype=np.float32).flatten() for s in state.values()])
    elif isinstance(state, (tuple, list)):
        state = np.concatenate([np.asarray(s, dtype=np.float32).flatten() for s in state])
    else:
        state = np.asarray(state, dtype=np.float32).flatten()
    return state.reshape(1, -1)

def train_agent(env_name: str, agent: QLearningAgent, num_episodes: int, checkpoint_interval: int = 100):
    """
    Train the Q-learning agent in the specified environment.
    """
    try:
        env = gym.make(env_name)
    except gym.error.Error as e:
        logger.error(f"Failed to create environment {env_name}: {e}")
        raise

    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        state = env.reset()
        state = preprocess_state(state)
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update(batch_size=32)
            state = next_state
            total_reward += reward

        logger.info(f'Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}')

        if (episode + 1) % checkpoint_interval == 0:
            checkpoint_path = f'checkpoint_model_{episode + 1}.h5'
            agent.save_weights(checkpoint_path)
            logger.info(f"Checkpoint saved at {checkpoint_path}")

        if (episode + 1) % 100 == 0:
            agent.update_target_network()

    env.close()
    final_save_path = 'trained_model.h5'
    agent.save_weights(final_save_path)
    logger.info(f"Final model saved successfully at {final_save_path}")

def evaluate_agent(agent: QLearningAgent, env_name: str, num_episodes: int) -> Tuple[float, float]:
    """
    Evaluate the trained Q-learning agent over a number of episodes.
    """
    try:
        env = gym.make(env_name)
    except gym.error.Error as e:
        logger.error(f"Failed to create environment {env_name}: {e}")
        raise

    total_rewards = []

    for episode in tqdm(range(num_episodes), desc="Evaluation Episodes"):
        state = env.reset()
        state = preprocess_state(state)
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)
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

def load_agent(model_path: str, input_dim: int, num_hidden_units: int, num_attention_heads: int, action_space_size: int) -> QLearningAgent:
    """
    Load a trained Q-learning agent from the specified model path.
    """
    agent = QLearningAgent(input_dim=input_dim, num_hidden_units=num_hidden_units, num_attention_heads=num_attention_heads, action_space_size=action_space_size)
    try:
        agent.load_weights(model_path)
        logger.info("Model loaded successfully.")
        return agent
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def main():
    input_dim = 24
    num_hidden_units = 128
    num_attention_heads = 4
    action_space_size = 4
    num_episodes = 1000
    eval_episodes = 100

    agent = QLearningAgent(input_dim=input_dim, num_hidden_units=num_hidden_units, num_attention_heads=num_attention_heads, action_space_size=action_space_size)

    env_name = 'BipedalWalker-v3'

    try:
        train_agent(env_name, agent, num_episodes)
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

    model_path = 'trained_model.h5'
    try:
        agent = load_agent(model_path, input_dim, num_hidden_units, num_attention_heads, action_space_size)
    except Exception as e:
        logger.error(f"Error loading agent: {e}")
        raise

    try:
        avg_reward, std_reward = evaluate_agent(agent, env_name, eval_episodes)
        print(f'Average Reward: {avg_reward}, Standard Deviation of Reward: {std_reward}')
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()