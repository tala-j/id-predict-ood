"""
Create train, in-distribution test, and OOD test datasets.

Train: 100K balanced+matched + 100K unbalanced+unmatched = 200K unique strings,
       repeated 5 times with different orderings = 1M rows.
ID test: 500 balanced+matched + 500 unbalanced+unmatched, disjoint from train.
OOD test: 1000 unbalanced+matched, disjoint from train and ID test.

Run from the repo root:
    python execution/make_datasets/create_splits.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.dirname(__file__))

import random
import pandas as pd
import numpy as np
from enum import Enum
from tqdm import tqdm
from dyck_utils import all_opens, all_closes, get_close_of, DyckString

# Inject all_opens/all_closes into utils.data's namespace so matchedHeuristic.fn works
import utils.data as _data_module
_data_module.all_opens = all_opens
_data_module.all_closes = all_closes

from utils.data import Heuristic, get_heuristics

balancedHeuristic, matchedHeuristic = get_heuristics()

# ---------------------------------------------------------------------------
# Copied verbatim from dataset_maker.py to avoid triggering its module-level
# samp_strs() execution on import. Every method body is identical to the
# original — only comments and type annotations are cosmetic differences.
# ---------------------------------------------------------------------------

class DistType(Enum):
    Uniform = "uniform"
    Binomial = "binomial"
    def __eq__(self, other):
        return self.value == other.value


class StringGenerator():
    def __init__(self, heuristics:list[Heuristic], cdns:list[bool], n_brackets:int = 1):
        self.heuristics = heuristics
        self.cdns = cdns
        assert len(heuristics) == len(cdns), "heuristics and cdns must have the same length"
        assert type(cdns[0]) == bool, "cdn must be a boolean"

        self.CATALAN = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786, 208012, 742900, 2674440, 9694845, 35357670, 129644790, 477638700, 1767263190, 6564120420]
        # self.description describes what kind of strings are generated, e.g. balanced enclosed
        self.description = " ".join([h.positive if cdn else h.negative for h, cdn in zip(heuristics, cdns)])
        self.reps = 0
        self.get_ur_str = lambda len_str: "".join([random.choice(self.opens + self.closes) for _ in range(len_str)])

        self.opens = all_opens[:n_brackets]
        self.closes = all_closes[:n_brackets]

    def requires_heuristic(self, name:str):
        for heuristic, cdn in zip(self.heuristics, self.cdns):
            if (heuristic.name == name) and (cdn == True):
                return True
        return False

    def get_uniform_bal_str(self, length:int):
        assert length % 2 == 0, "length of balanced string must be even"

        s = DyckString()
        s.append_open(random.choice(self.opens))

        for i in range(1, length):
            r = s.elevation() # current elev
            k = length - i # number of characters to generate
            p_closed = r*(k+r+2)/(2*k*(r+1)) # probability of choosing an open_paren next
            # see https://dl.acm.org/doi/pdf/10.1145/357084.357091
            if random.random() < p_closed:
                s.append_close()
            else:
                s.append_open(random.choice(self.opens))
        return str(s)

    def get_uniform_matched_str(self, length:int):
        assert length % 2 == 0, "length of matched string must be even"
        open_half = [random.choice(self.opens) for _ in range(length//2)]
        close_half = [get_close_of(open_br) for open_br in open_half]
        bits = open_half + close_half
        random.shuffle(bits)
        gen_str = "".join(bits)

        return gen_str

    def get_unbalanced_enclosed_str(self, length: int):
        assert length >= 6, "length must be at least 6 to be unbalanced enclosed"
        while True:
            inner_str = self.get_ur_str(length - 2)
            gen_str = f"({inner_str})"
            if not balancedHeuristic.fn(gen_str):
                return gen_str

    def get_uniform_str(self, length:int):
        self.reps += 1
        if self.reps >= 100:
            raise ValueError(f"Too many failures when attempting to generate a length {length} {self.description} string. Are you sure this is possible?")

        if not self.requires_heuristic("balanced") and self.requires_heuristic("enclosure"):
            if length < 6:
                return self.get_uniform_str(6)  # Recursively call with minimum length of 6

        if self.requires_heuristic("balanced"):
            gen_str = self.get_uniform_bal_str(length)

        elif self.requires_heuristic("matched"):
            gen_str = self.get_uniform_matched_str(length)

        elif (not self.requires_heuristic("balanced")) and self.requires_heuristic("enclosure"):
            gen_str = self.get_unbalanced_enclosed_str(length)

        else:
            gen_str = self.get_ur_str(length)

        for heuristic, cdn in zip(self.heuristics, self.cdns):
            if heuristic.fn(gen_str) != cdn:
                # try again
                return self.get_uniform_str(length)

        # we have a string that satisfies all the constraints
        self.reps = 0
        return gen_str


def generate_uniform_random_string(string_generator: StringGenerator,
                                   length_dist:DistType, length_params:tuple):
    # aka sample_method = "uniform_random"
    if length_dist == DistType.Uniform:
        a, b = length_params
        a = 1 if a // 2 == 0 else a // 2
        b = b // 2
        str_len = 2*random.randint(a, b) # even number by length param
    elif length_dist == DistType.Binomial:
        n, p = length_params
        str_len = 2 + 2*np.random.binomial(n/2-1, p) # even number between 2 and n

    return string_generator.get_uniform_str(str_len)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HEURISTICS = [balancedHeuristic, matchedHeuristic]
LENGTH_DIST = DistType.Binomial
LENGTH_PARAMS = (40, 0.5)
N_BRACKETS = 1

TRAIN_POS_SIZE = 100_000   # balanced + matched
TRAIN_NEG_SIZE = 100_000   # unbalanced + unmatched
NUM_REPEATS = 5
ID_TEST_POS_SIZE = 500     # balanced + matched
ID_TEST_NEG_SIZE = 500     # unbalanced + unmatched
OOD_TEST_SIZE = 1_000      # unbalanced + matched

SHUFFLE_SEEDS = [161, 231, 220, 487, 953]


def generate_unique_strings(cdns, n, exclude=None):
    """Generate n unique strings satisfying heuristic conditions, excluding a set."""
    gen = StringGenerator(HEURISTICS, cdns, n_brackets=N_BRACKETS)
    result = set()
    if exclude is None:
        exclude = set()

    with tqdm(total=n, desc=gen.description) as pbar:
        while len(result) < n:
            s = generate_uniform_random_string(gen, LENGTH_DIST, LENGTH_PARAMS)
            if s not in result and s not in exclude:
                result.add(s)
                pbar.update(1)

    return result


def build_df(strings, balanced_val, matched_val):
    """Build a DataFrame from a set of strings with fixed heuristic labels."""
    df = pd.DataFrame({'string': list(strings)})
    df['balanced'] = balanced_val
    df['matched'] = matched_val
    return df


def create_splits(save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. Generate 200K unique training strings ---
    print("=== Generating training pool ===")
    train_pos = generate_unique_strings([True, True], TRAIN_POS_SIZE)
    train_neg = generate_unique_strings([False, False], TRAIN_NEG_SIZE,
                                        exclude=train_pos)
    train_pool = train_pos | train_neg

    # Build the 200K base DataFrame and repeat 5x with different orderings
    base_df = pd.concat([
        build_df(train_pos, True, True),
        build_df(train_neg, False, False),
    ], ignore_index=True)

    print("=== Shuffling into 1M training set ===")
    blocks = []
    for seed in SHUFFLE_SEEDS:
        blocks.append(base_df.sample(frac=1, random_state=seed).reset_index(drop=True))
    train_df = pd.concat(blocks, ignore_index=True)

    train_path = os.path.join(save_dir, "train_binomial(40,0.5).csv")
    train_df.to_csv(train_path, index=False)
    print(f"Saved training set: {len(train_df)} rows -> {train_path}")

    # --- 2. Generate 1K ID test strings (disjoint from train) ---
    print("=== Generating ID test set ===")
    id_test_pos = generate_unique_strings([True, True], ID_TEST_POS_SIZE,
                                          exclude=train_pool)
    id_test_neg = generate_unique_strings([False, False], ID_TEST_NEG_SIZE,
                                          exclude=train_pool | id_test_pos)

    id_test_df = pd.concat([
        build_df(id_test_pos, True, True),
        build_df(id_test_neg, False, False),
    ], ignore_index=True)

    id_test_path = os.path.join(save_dir, "in_dist_test_binomial(40,0.5).csv")
    id_test_df.to_csv(id_test_path, index=False)
    print(f"Saved ID test set: {len(id_test_df)} rows -> {id_test_path}")

    # --- 3. Generate 1K OOD test strings (disjoint from everything) ---
    print("=== Generating OOD test set ===")
    all_previous = train_pool | id_test_pos | id_test_neg
    ood_test = generate_unique_strings([False, True], OOD_TEST_SIZE,
                                       exclude=all_previous)

    ood_test_df = build_df(ood_test, False, True)

    ood_test_path = os.path.join(save_dir, "test_binomial(40,0.5).csv")
    ood_test_df.to_csv(ood_test_path, index=False)
    print(f"Saved OOD test set: {len(ood_test_df)} rows -> {ood_test_path}")

    # --- 4. Verification ---
    print("\n=== Verification ===")

    expected_train_rows = (TRAIN_POS_SIZE + TRAIN_NEG_SIZE) * NUM_REPEATS
    assert len(train_df) == expected_train_rows, \
        f"Train should have {expected_train_rows} rows, got {len(train_df)}"

    unique_train_pos = set(train_df[train_df['balanced'] == True]['string'])
    unique_train_neg = set(train_df[train_df['balanced'] == False]['string'])
    assert len(unique_train_pos) == TRAIN_POS_SIZE, \
        f"Expected {TRAIN_POS_SIZE} unique balanced+matched, got {len(unique_train_pos)}"
    assert len(unique_train_neg) == TRAIN_NEG_SIZE, \
        f"Expected {TRAIN_NEG_SIZE} unique unbalanced+unmatched, got {len(unique_train_neg)}"

    unique_train = set(train_df['string'])
    id_test_strings = set(id_test_df['string'])
    ood_test_strings = set(ood_test_df['string'])

    assert len(id_test_df) == ID_TEST_POS_SIZE + ID_TEST_NEG_SIZE
    assert len(id_test_strings & unique_train) == 0, "ID test overlaps with train"

    assert len(ood_test_df) == OOD_TEST_SIZE
    assert len(ood_test_strings & unique_train) == 0, "OOD test overlaps with train"
    assert len(ood_test_strings & id_test_strings) == 0, "OOD test overlaps with ID test"

    # Verify heuristic labels are correct for every string
    for _, row in id_test_df.iterrows():
        assert balancedHeuristic.fn(row['string']) == row['balanced']
        assert matchedHeuristic.fn(row['string']) == row['matched']
    for _, row in ood_test_df.iterrows():
        assert balancedHeuristic.fn(row['string']) == row['balanced']
        assert matchedHeuristic.fn(row['string']) == row['matched']

    print("All assertions passed.")


if __name__ == "__main__":
    create_splits(save_dir=".")
