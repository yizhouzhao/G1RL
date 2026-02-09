# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import omni.timeline
import omni.ui as ui
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import get_prim_object_type
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.gui.components.element_wrappers import CollapsableFrame, DropDown, FloatField, TextBlock
from isaacsim.gui.components.ui_utils import get_style


class UIBuilder:
    def __init__(self):
        # Frames are sub-windows that can contain multiple UI elements
        self.frames = []

        # UI elements created using a UIElementWrapper from isaacsim.gui.components.element_wrappers
        self.wrapped_ui_elements = []

        # Get access to the timeline to control stop/pause/play programmatically
        self._timeline = omni.timeline.get_timeline_interface()

        # Run initialization for the provided example
        self._on_init()

    ###################################################################################
    #           The Functions Below Are Called Automatically By extension.py
    ###################################################################################

    def on_timeline_event(self, event):
        """Callback for Timeline events (Play, Pause, Stop)

        Args:
            event (omni.timeline.TimelineEventType): Event Type
        """
        pass

    def on_physics_step(self, step):
        """Callback for Physics Step.
        Physics steps only occur when the timeline is playing

        Args:
            step (float): Size of physics step
        """
        # print("on_physics_step", step)
        if self.task is not None:
            if not self.task.initialized:
                self.task.initialize()
                print("[UIBuilder] task initialized")


    def on_stage_event(self, event):
        """Callback for Stage Events

        Args:
            event (omni.usd.StageEventType): Event Type
        """
        if event.type == int(omni.usd.StageEventType.ASSETS_LOADED):  # Any asset added or removed
            pass
        elif event.type == int(omni.usd.StageEventType.SIMULATION_START_PLAY):  # Timeline played
            # Treat a playing timeline as a trigger for selecting an Articulation
            pass
        elif event.type == int(omni.usd.StageEventType.SIMULATION_STOP_PLAY):  # Timeline stopped
            # Ignore pause events
            if self._timeline.is_stopped():
                pass

    def cleanup(self):
        """
        Called when the stage is closed or the extension is hot reloaded.
        Perform any necessary cleanup such as removing active callback functions
        Buttons imported from isaacsim.gui.components.element_wrappers implement a cleanup function that should be called
        """
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

    def build_ui(self):
        """
        Build a custom UI tool to run your extension.
        This function will be called any time the UI window is closed and reopened.
        """
        selection_panel_frame = CollapsableFrame("Selection Panel", collapsed=False)

        with selection_panel_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                ui.Spacer(height=5)
                ui.Button("Setup Stage", clicked_fn=self.setup_stage)
                ui.Button("Get Observation", clicked_fn=self.get_observation)
                ui.Button("Reset", clicked_fn=self.reset_task)


    ######################################################################################
    # Functions Below This Point Support The Provided Example And Can Be Replaced/Deleted
    ######################################################################################

    def _on_init(self):
        self.task = None

    def setup_stage(self):
        print("Setup Stage")
        # from .g1 import G1Robot
        # robot = G1Robot("/World/G1")

        from .locomotion_task import G1LocomotionTask
        self.task: G1LocomotionTask = G1LocomotionTask()
        self.task.set_up_scene()


    def get_observation(self):
        print("Get Observation")
        if self.task is not None and self.task.initialized:
            obs = self.task.get_observation()
            print("[UIBuilder] obs", obs)

    def reset_task(self):
        print("reset task")
        if self.task is not None and self.task.initialized:
            self.task.reset()
