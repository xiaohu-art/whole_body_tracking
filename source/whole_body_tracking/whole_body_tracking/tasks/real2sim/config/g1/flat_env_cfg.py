from isaaclab.utils import configclass
from whole_body_tracking.assets import ASSET_DIR
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.real2sim.real2sim_env_cfg import Real2SimEnvCfg


@configclass
class G1FlatEnvCfg(Real2SimEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class G1FlatDanceEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.real_traj.traj_path = f"{ASSET_DIR}/g1/real_trajs"
