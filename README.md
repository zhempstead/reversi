To get started, try running `reversi_game.py`:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python reversi_game.py
```

An AI can be implemented by creating a `ReversiAgent` subclass, implementing the `policy` method.

To play against a human, start a ReversiGame with a DualAgent, one of whose constituent agents is a HumanAgent. You should also set `headless=False`. `headless=True` would be appropriate for RL training.
