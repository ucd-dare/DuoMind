import numpy as np
import sapien
from transforms3d.euler import euler2quat

from robopoly import PACKAGE_ASSET_DIR
from robopoly.agents.registration import register_agent
from robopoly.sensors.camera import CameraConfig
from robopoly.utils import sapien_utils

from .panda import Panda


@register_agent()
class PandaWristCam(Panda):
    """Panda arm robot with the real sense camera attached to gripper"""

    uid = "panda_wristcam"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="hand_camera",
                # camera_link is 20 mm off the Panda hand centerline in the
                # URDF. Cancel that lateral offset and pitch the optical axis
                # 25 degrees toward the gripper/work surface so the gripper is
                # horizontally centered and sits near the image midline.
                pose=sapien.Pose(
                    p=[0, -0.02, 0],
                    q=euler2quat(0, np.deg2rad(25.0), 0),
                ),
                width=128,
                height=128,
                # Keep the native 128x128 sensor, but expose more of the nearby
                # workspace than the original 90-degree wrist view.
                fov=np.deg2rad(110.0),
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            )
        ]
