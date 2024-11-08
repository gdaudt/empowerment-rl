from __future__ import annotations

# For relative imports to work in Python 3.6
# import os, sys

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(SCRIPT_DIR + '/..')

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        
        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        if max_steps is None:
            max_steps = 4 * size**2
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=256,
            **kwargs,
        )
        
    @staticmethod
    def _gen_mission():
        return "grand mission"
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        self.put_obj(Goal(), width - 2, height - 2)
        for i in range(0, height-3):
            self.grid.set(5, i, Wall())


def main():
    env = SimpleEnv(render_mode='human')
    mc = ManualControl(env, seed=42)
    mc.start()

if __name__ == '__main__':
    main()