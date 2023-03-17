To get started, try running `reversi_game.py`:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python human_cpu_game.py
```

An AI can be implemented by creating a `ReversiAgent` subclass, implementing the `policy` method.

To play against a human, start a ReversiGame with a DualAgent, one of whose constituent agents is a HumanAgent. You should also set `headless=False`. `headless=True` would be appropriate for RL training.

To test ChatGPT, see the 'gpt_game.py' script. You will need two things:
- An API key, added to '.env'
- A replay file with "good" moves to draw from for few-shot learning
