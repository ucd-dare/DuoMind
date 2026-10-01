import numpy as np
import sapien

from robopoly import ASSET_DIR
from robopoly.agents.base_agent import BaseAgent
from robopoly.agents.controllers import *
from robopoly.agents.registration import register_agent
from robopoly.sensors.camera import CameraConfig


# TODO (stao) (xuanlin): Add mobile base, model it properly based on real2sim
@register_agent(asset_download_ids=["googlerobot"])
class GoogleRobot(BaseAgent):
    uid = "googlerobot"
    urdf_path = (
        f"{ASSET_DIR}/robots/googlerobot/google_robot_meta_sim_fix_fingertip.urdf"
    )
    urdf_config = dict()

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="overhead_camera",
                pose=sapien.Pose([0, 0, 0], [0.5, 0.5, -0.5, 0.5]),
                width=640,
                height=512,
                entity_uid="link_camera",
                intrinsic=np.array([[425.0, 0, 305.0], [0, 413.1, 233.0], [0, 0, 1]]),
            )
        ]
