import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "eva-lab"))
sys.path.insert(0, str(ROOT / "src" / "shared"))

from eva_lab.muzero.jax_mcts import JAXMuZeroMCTS, Node


def test_expand_node_accepts_vector_policy_without_batch_dimension():
    config = SimpleNamespace(action_space_size=5)
    mcts = JAXMuZeroMCTS(config)
    root = Node(0.0)

    mcts._expand_node(root, np.array([0.1, 0.2, 0.3, 0.15, 0.25], dtype=np.float32))

    assert len(root.children) == 5
    assert root.children[2].prior == 0.3
