from isaaclab.utils import configclass
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.balancing.balancing_env_cfg import BalancingEnvCfg


@configclass
class G1FlatEnvCfg(BalancingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
