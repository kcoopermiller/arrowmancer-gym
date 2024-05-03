from src import ArrowmancerEnv
import pygame

units = [
    {'name': 'Capricorn', 'level': '4A'}, 
    {'name': 'Aquarius', 'level': '5'}, 
    {'name': 'Pisces', 'level': 'Alt'}, 
]

# env = ArrowmancerEnv(units)

# obs = env.reset()
# done = False

# while not done:
#     # TODO: Replace this with agent's action selection logic
#     action = env.action_space.sample()

#     obs, reward, done, info = env.step(action)

#     print(f"Observation: {obs}, Reward: {reward}, Done: {done}")

#     env.render()

# env.close()

# Create an instance of the ArrowmancerEnv
env = ArrowmancerEnv(units)

# Reset the environment
obs = env.reset()

# Game loop
running = True
pygame.init()
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Render the environment
    env.render()

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()