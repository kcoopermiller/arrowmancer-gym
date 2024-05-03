from src.dance import dance_patterns
from src.env import ArrowmancerEnv

units = [
    {'name': 'Capricorn', 'level': '4A'}, 
    {'name': 'Aquarius', 'level': '5'}, 
    {'name': 'Pisces', 'level': 'Alt'}, 
]

env = ArrowmancerEnv(units)

obs = env.reset()
done = False

while not done:
    # TODO: Replace this with agent's action selection logic
    action = env.action_space.sample()

    obs, reward, done, info = env.step(action)

    print(f"Observation: {obs}, Reward: {reward}, Done: {done}")

    env.render()

env.close()