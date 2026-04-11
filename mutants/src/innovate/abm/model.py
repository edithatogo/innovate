from mesa import Model
from mesa.space import MultiGrid

from .agent import InnovationAgent
from typing import Annotated
from typing import Callable
from typing import ClassVar

MutantDict = Annotated[dict[str, Callable], "Mutant"] # type: ignore


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg = None): # type: ignore
    """Forward call to original or mutated function, depending on the environment"""
    import os # type: ignore
    mutant_under_test = os.environ['MUTANT_UNDER_TEST'] # type: ignore
    if mutant_under_test == 'fail': # type: ignore
        from mutmut.__main__ import MutmutProgrammaticFailException # type: ignore
        raise MutmutProgrammaticFailException('Failed programmatically')       # type: ignore
    elif mutant_under_test == 'stats': # type: ignore
        from mutmut.__main__ import record_trampoline_hit # type: ignore
        record_trampoline_hit(orig.__module__ + '.' + orig.__name__) # type: ignore
        # (for class methods, orig is bound and thus does not need the explicit self argument)
        result = orig(*call_args, **call_kwargs) # type: ignore
        return result # type: ignore
    prefix = orig.__module__ + '.' + orig.__name__ + '__mutmut_' # type: ignore
    if not mutant_under_test.startswith(prefix): # type: ignore
        result = orig(*call_args, **call_kwargs) # type: ignore
        return result # type: ignore
    mutant_name = mutant_under_test.rpartition('.')[-1] # type: ignore
    if self_arg is not None: # type: ignore
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs) # type: ignore
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs) # type: ignore
    return result # type: ignore


class InnovationModel(Model):
    """A model for innovation diffusion."""

    def __init__(self, num_agents, width, height):
        args = [num_agents, width, height]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁInnovationModelǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁInnovationModelǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁInnovationModelǁ__init____mutmut_orig(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_1(self, num_agents, width, height):
        super().__init__()
        self.num_agents = None
        self.grid = MultiGrid(width, height, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_2(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = None
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_3(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(None, height, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_4(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, None, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_5(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, None)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_6(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(height, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_7(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_8(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, )
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_9(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, False)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_10(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = None  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_11(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = False  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_12(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(None):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_13(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = None
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_14(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(None, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_15(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, None)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_16(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_17(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, )
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_18(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = None
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_19(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(None)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_20(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = None
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_21(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(None)
            self.grid.place_agent(agent, (x, y))

    def xǁInnovationModelǁ__init____mutmut_22(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(None, (x, y))

    def xǁInnovationModelǁ__init____mutmut_23(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, None)

    def xǁInnovationModelǁ__init____mutmut_24(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent((x, y))

    def xǁInnovationModelǁ__init____mutmut_25(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True  # For visualization/interactive mode

        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, )
    
    xǁInnovationModelǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁInnovationModelǁ__init____mutmut_1': xǁInnovationModelǁ__init____mutmut_1, 
        'xǁInnovationModelǁ__init____mutmut_2': xǁInnovationModelǁ__init____mutmut_2, 
        'xǁInnovationModelǁ__init____mutmut_3': xǁInnovationModelǁ__init____mutmut_3, 
        'xǁInnovationModelǁ__init____mutmut_4': xǁInnovationModelǁ__init____mutmut_4, 
        'xǁInnovationModelǁ__init____mutmut_5': xǁInnovationModelǁ__init____mutmut_5, 
        'xǁInnovationModelǁ__init____mutmut_6': xǁInnovationModelǁ__init____mutmut_6, 
        'xǁInnovationModelǁ__init____mutmut_7': xǁInnovationModelǁ__init____mutmut_7, 
        'xǁInnovationModelǁ__init____mutmut_8': xǁInnovationModelǁ__init____mutmut_8, 
        'xǁInnovationModelǁ__init____mutmut_9': xǁInnovationModelǁ__init____mutmut_9, 
        'xǁInnovationModelǁ__init____mutmut_10': xǁInnovationModelǁ__init____mutmut_10, 
        'xǁInnovationModelǁ__init____mutmut_11': xǁInnovationModelǁ__init____mutmut_11, 
        'xǁInnovationModelǁ__init____mutmut_12': xǁInnovationModelǁ__init____mutmut_12, 
        'xǁInnovationModelǁ__init____mutmut_13': xǁInnovationModelǁ__init____mutmut_13, 
        'xǁInnovationModelǁ__init____mutmut_14': xǁInnovationModelǁ__init____mutmut_14, 
        'xǁInnovationModelǁ__init____mutmut_15': xǁInnovationModelǁ__init____mutmut_15, 
        'xǁInnovationModelǁ__init____mutmut_16': xǁInnovationModelǁ__init____mutmut_16, 
        'xǁInnovationModelǁ__init____mutmut_17': xǁInnovationModelǁ__init____mutmut_17, 
        'xǁInnovationModelǁ__init____mutmut_18': xǁInnovationModelǁ__init____mutmut_18, 
        'xǁInnovationModelǁ__init____mutmut_19': xǁInnovationModelǁ__init____mutmut_19, 
        'xǁInnovationModelǁ__init____mutmut_20': xǁInnovationModelǁ__init____mutmut_20, 
        'xǁInnovationModelǁ__init____mutmut_21': xǁInnovationModelǁ__init____mutmut_21, 
        'xǁInnovationModelǁ__init____mutmut_22': xǁInnovationModelǁ__init____mutmut_22, 
        'xǁInnovationModelǁ__init____mutmut_23': xǁInnovationModelǁ__init____mutmut_23, 
        'xǁInnovationModelǁ__init____mutmut_24': xǁInnovationModelǁ__init____mutmut_24, 
        'xǁInnovationModelǁ__init____mutmut_25': xǁInnovationModelǁ__init____mutmut_25
    }
    xǁInnovationModelǁ__init____mutmut_orig.__name__ = 'xǁInnovationModelǁ__init__'

    def step(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁInnovationModelǁstep__mutmut_orig'), object.__getattribute__(self, 'xǁInnovationModelǁstep__mutmut_mutants'), args, kwargs, self)

    def xǁInnovationModelǁstep__mutmut_orig(self):
        """Run one step of the model."""
        self.agents.do("step")

    def xǁInnovationModelǁstep__mutmut_1(self):
        """Run one step of the model."""
        self.agents.do(None)

    def xǁInnovationModelǁstep__mutmut_2(self):
        """Run one step of the model."""
        self.agents.do("XXstepXX")

    def xǁInnovationModelǁstep__mutmut_3(self):
        """Run one step of the model."""
        self.agents.do("STEP")
    
    xǁInnovationModelǁstep__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁInnovationModelǁstep__mutmut_1': xǁInnovationModelǁstep__mutmut_1, 
        'xǁInnovationModelǁstep__mutmut_2': xǁInnovationModelǁstep__mutmut_2, 
        'xǁInnovationModelǁstep__mutmut_3': xǁInnovationModelǁstep__mutmut_3
    }
    xǁInnovationModelǁstep__mutmut_orig.__name__ = 'xǁInnovationModelǁstep'
