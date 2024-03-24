import numpy as np
import matplotlib.pyplot as plt

# The mean and variance graph shows the average and spread of rewards over multiple runs, 
# while the learning curve tracks how the algorithm's performance improves over time during training

# Sample rewards for Q-learning and Policy Gradient
q_learning_rewards = [
    [10, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27, 30],  # Q-learning rewards
    [8, 11, 9, 12, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 27],       # Q-learning rewards
    [9, 13, 11, 14, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29]       # Q-learning rewards
]

policy_gradient_rewards = [
    [20, 22, 21, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31, 34, 33, 36, 35, 38, 37, 40],  # Policy Gradient rewards
    [18, 21, 19, 22, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30, 33, 32, 35, 34, 37],    # Policy Gradient rewards
    [19, 23, 21, 24, 22, 25, 24, 27, 26, 29, 28, 31, 30, 33, 32, 35, 34, 37, 36, 39]     # Policy Gradient rewards
]

# Graph of rewards over episodes with mean and variance

# Calculate mean and variance for Q-learning rewards
q_learning_mean = np.mean(q_learning_rewards, axis=0)
q_learning_variance = np.var(q_learning_rewards, axis=0)

# Calculate mean and variance for Policy Gradient rewards
policy_gradient_mean = np.mean(policy_gradient_rewards, axis=0)
policy_gradient_variance = np.var(policy_gradient_rewards, axis=0)

# Plot Q-learning rewards
plt.plot(q_learning_mean, label='Q-learning', color='blue')
plt.fill_between(range(len(q_learning_mean)),
                 q_learning_mean - q_learning_variance,
                 q_learning_mean + q_learning_variance,
                 color='blue', alpha=0.3)

# Plot Policy Gradient rewards
plt.plot(policy_gradient_mean, label='Policy Gradient', color='red')
plt.fill_between(range(len(policy_gradient_mean)),
                 policy_gradient_mean - policy_gradient_variance,
                 policy_gradient_mean + policy_gradient_variance,
                 color='red', alpha=0.3)

# Add labels and legend
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward over Episode (Q-learning vs Policy Gradient)')
plt.legend()



# Histogram to show which method generally has higher rewards

# # Flatten the rewards list for histogram
# q_learning_rewards_flat = np.concatenate(q_learning_rewards)
# policy_gradient_rewards_flat = np.concatenate(policy_gradient_rewards)

# # Plot histograms
# plt.hist(q_learning_rewards_flat, bins=20, alpha=0.5, label='Q-learning', color='blue')
# plt.hist(policy_gradient_rewards_flat, bins=20, alpha=0.5, label='Policy Gradient', color='red')

# # Add labels and legend
# plt.xlabel('Reward')
# plt.ylabel('Frequency')
# plt.title('Reward Distribution (Q-learning vs Policy Gradient)')
# plt.legend()


# Learning curve

# # Calculate cumulative rewards
# q_learning_cumulative = np.cumsum(q_learning_rewards, axis=1)
# policy_gradient_cumulative = np.cumsum(policy_gradient_rewards, axis=1)

# # Calculate mean cumulative rewards
# q_learning_mean = np.mean(q_learning_cumulative, axis=0)
# policy_gradient_mean = np.mean(policy_gradient_cumulative, axis=0)

# # Plot learning curve
# plt.plot(range(1, 21), q_learning_mean, label='Q-learning', color='blue')
# plt.plot(range(1, 21), policy_gradient_mean, label='Policy Gradient', color='red')

# # Add labels and legend
# plt.xlabel('Episode')
# plt.ylabel('Cumulative Reward')
# plt.title('Learning Curve (Q-learning vs Policy Gradient)')
# plt.legend()

# Show plot
plt.grid(True)
plt.show()
