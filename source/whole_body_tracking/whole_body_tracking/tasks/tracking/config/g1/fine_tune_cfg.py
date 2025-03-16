from isaaclab.utils import configclass

from whole_body_tracking.assets import ASSET_DIR
from whole_body_tracking.tasks.real2sim.mdp.pre_trained_real2sim_action import PreTrainedReal2SimActionCfg
from whole_body_tracking.tasks.real2sim.real2sim_env_cfg import Real2SimEnvCfg
from .flat_env_cfg import G1FlatDanceEnvCfg

REAL2SIM_ENV_CFG = Real2SimEnvCfg()


@configclass
class G1FineTuneDanceEnvCfg(G1FlatDanceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.real2sim = PreTrainedReal2SimActionCfg(
            asset_name="robot",
            policy_path=f"{ASSET_DIR}/g1/real2sim.pt",
            real2sim_actions=REAL2SIM_ENV_CFG.actions.wrench,
            real2sim_observations=REAL2SIM_ENV_CFG.observations.policy,
        )
