import sys
from infer import EventScoring
scoring = EventScoring()
path = '/home/dsoft/Music/me/project/tf-pose-estimation/image_test_copy (copy)'
if __name__ == "__main__":
    scoring.score_infer(path)