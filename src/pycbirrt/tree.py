from dataclasses import dataclass, field
import numpy as np


@dataclass
class Node:
    """A node in the RRT tree."""

    config: np.ndarray
    parent: int | None = None  # Index of parent node, None for root


class RRTree:
    """Rapidly-exploring Random Tree data structure."""

    def __init__(self, root_config: np.ndarray | list[np.ndarray]):
        """Initialize tree with one or more root nodes.

        Args:
            root_config: Single configuration or list of configurations.
                        All provided configs become roots (parent=None).
        """
        # Normalize to list
        if isinstance(root_config, np.ndarray):
            configs = [root_config]
        else:
            configs = root_config

        self.nodes: list[Node] = []
        self._configs: list[np.ndarray] = []

        # Add all configs as roots
        for config in configs:
            self.nodes.append(Node(config=config.copy(), parent=None))
            self._configs.append(config.copy())

    def add_node(self, config: np.ndarray, parent_idx: int) -> int:
        """Add a new node to the tree.

        Args:
            config: Joint configuration for new node
            parent_idx: Index of parent node

        Returns:
            Index of newly added node
        """
        node = Node(config=config.copy(), parent=parent_idx)
        self.nodes.append(node)
        self._configs.append(config.copy())
        return len(self.nodes) - 1

    def nearest(self, config: np.ndarray) -> int:
        """Find the nearest node to a configuration.

        Args:
            config: Query configuration

        Returns:
            Index of nearest node
        """
        configs = np.array(self._configs)
        distances = np.linalg.norm(configs - config, axis=1)
        return int(np.argmin(distances))

    def get_path_to_root(self, node_idx: int) -> list[np.ndarray]:
        """Extract path from node back to root.

        Args:
            node_idx: Index of node to trace from

        Returns:
            List of configurations from root to node
        """
        path = []
        idx = node_idx
        while idx is not None:
            path.append(self.nodes[idx].config)
            idx = self.nodes[idx].parent
        return list(reversed(path))

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def num_roots(self) -> int:
        """Number of root nodes in the tree."""
        return sum(1 for node in self.nodes if node.parent is None)
