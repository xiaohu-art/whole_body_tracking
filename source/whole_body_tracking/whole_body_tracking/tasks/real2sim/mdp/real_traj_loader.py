import numpy as np
import os
import torch


class RealTrajLoader:
    """
    Helper class to load and sample motion data from NumPy-file format.
    """

    def __init__(self, traj_path: str, device: torch.device | str) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            traj_path: Motion file path to load.
            device: The device to which to load the data.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        assert os.path.isfile(traj_path), f"Invalid file path: {traj_path}"
        data = np.load(traj_path)

        self.device = device
        self._joint_names = data["joint_names"].tolist()
        self._q = torch.tensor(data["q"], dtype=torch.float32, device=self.device)
        self._v = torch.tensor(data["v"], dtype=torch.float32, device=self.device)
        self._a = torch.tensor(data["a"], dtype=torch.float32, device=self.device)

        self.num_frames = self._q.shape[0]
        print(f"Real trajectory loaded ({traj_path}):  frames: {self.num_frames}")

    @property
    def joint_names(self) -> list[str]:
        """Skeleton DOF names."""
        return self._joint_names

    @property
    def num_joints(self) -> int:
        """Number of skeleton's DOFs."""
        return len(self._joint_names)

    def sample_frame(self, frame_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._q[frame_idx], self._v[frame_idx], self._a[frame_idx]

    def sample_indexes(self, num_samples: int) -> torch.Tensor:
        return torch.randint(0, self.num_frames, (num_samples,), device=self.device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Real trajectory path")
    args = parser.parse_args()

    traj = RealTrajLoader(args.path, "cpu")

    print("- number of frames:", traj.num_frames)
    print("- number of DOFs:", traj.num_joints)
    print("- DOF names:", traj.joint_names)
