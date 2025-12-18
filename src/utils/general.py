import subprocess
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES,NAMES
import random
import json

def get_git_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_modified_files():
    return (
        subprocess.check_output(["git", "diff", "--name-status"])
        .decode("ascii")
        .strip()
    )


def get_deterministic_name(hparams):
    state = random.getstate()

    string_seed = json.dumps(hparams)
    random.seed(string_seed)
    out = get_random_name(combo=[ADJECTIVES, ADJECTIVES, NAMES], separator="_", style="lowercase")

    random.setstate(state)
    return out

