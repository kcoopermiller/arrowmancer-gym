<p align="center">
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/kcoopermiller/arrowmancer-gym/assets/44559144/4be53a84-8acd-49d9-a050-1da6af03fab5" width="600" >
  <img alt="Arrowmancer logo" src="https://github.com/kcoopermiller/arrowmancer-gym/assets/44559144/554c3ec7-defd-4fca-bdbb-993587d42d76" width="600"/>
  </picture> 
</p>

[Gymansium](https://github.com/Farama-Foundation/Gymnasium) RL environment for [Spellbrush](https://spellbrush.com/)'s [Arrowmancer](https://www.arrowmancer.com/) + a simple [Deep Q Learning](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm) agent

<img alt="Demo GIF" src="https://github.com/kcoopermiller/arrowmancer-gym/assets/44559144/adec638c-729d-4c72-9635-3cc7d346a5c4" width="40%" height="40%"/>

> [!WARNING]  
> Currently in the process of adding support for all standard banner witches. `agent.py` and `train.py` may not working as intended at the moment.

## Getting Started

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

```shell
# Clone the repository
git clone https://github.com/kcoopermiller/arrowmancer-gym.git
cd arrowmancer-gym

# Install dependencies
poetry install

# Run project
poetry run python main.py
```

## Roadmap
- [ ] Fix unit swapping
- [ ] Adjust reward for combos
- [ ] Add character abilities, passives, stats, etc.
- [ ] Beautify PyGame environment
- [ ] More accurate action space (ex: ability to move multiple units at once)


Emojis come from https://openmoji.org/