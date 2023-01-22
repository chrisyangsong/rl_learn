import gym
import numpy as np
import tensorflow as tf

env = gym.make("MountainCar-v0")
env.reset()

model = tf.keras.models.load_model('models/-89.00max_-119.96avg_-200.00min.model')

ep_rewards = []

for i in range(5):
    episilo_reward = 0

    state = env.reset()
    done = False
    while not done:
        qs_list = model.predict(np.array(state).reshape(-1, 2))
        action = np.argmax(qs_list[0])
        new_state, reward, done, _ = env.step(action)
        episilo_reward += reward
        state = new_state
        env.render(mode='human')

    env.close()
    print(f"episilon {i}, reward:{episilo_reward}")
    ep_rewards.append(episilo_reward)

print(np.mean(np.array(ep_rewards))) 
# 作者：Leon小草办 https://www.bilibili.com/read/cv17972518 出处：bilibili