from src.dance import dance_patterns
from src.env import ArrowmancerEnv

units = [
    {
        "name": "Capricorn",
        "star_rating": 6,
        "primary_dance": dance_patterns["Capricorn"][6],
        "alt_dance": dance_patterns["Capricorn"]["Alt"],
        "position": None
    },
    {
        "name": "Aquarius",
        "star_rating": 5,
        "primary_dance": dance_patterns["Aquarius"][5],
        "alt_dance": dance_patterns["Aquarius"]["Alt"],
        "position": None
    },
    {
        "name": "Pisces",
        "star_rating": 4,
        "primary_dance": dance_patterns["Pisces"]["4A"],
        "alt_dance": dance_patterns["Pisces"]["Alt"],
        "position": None
    }
]

env = ArrowmancerEnv()

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # Replace with your agent's action selection
    obs, reward, done, info = env.step(action)
    print(f"Observation: {obs}, Reward: {reward}, Done: {done}")
    env.render()

env.close()