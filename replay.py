import sys

from lib.replay import Replay

if len(sys.argv) > 1:
    replay = sys.argv[1]
else:
    replay = 'replays/human.json'
r = Replay(replay)
r.play()
