from src.dance import dance_patterns
from src.env import ArrowmancerEnv

# units = [
#     {'name': 'Capricorn', 'level': '6'}, 
#     {'name': 'Aquarius', 'level': '5'}, 
#     {'name': 'Pisces', 'level': 'Alt'}, 
# ]

# env = ArrowmancerEnv(units)

# obs = env.reset()
# done = False

# while not done:
#     # TODO: Replace this with agent's action selection logic
#     action = env.action_space.sample()

#     print(f"Action: {action}")

#     obs, reward, done, info = env.step(action)

#     print(f"Observation: {obs}, Reward: {reward}, Done: {done}")

#     env.render()

# env.close()
# check for -4 to 4
for i in range(-4, 5):
    print((i+1) // 3, (i+1) % 3 - 1)