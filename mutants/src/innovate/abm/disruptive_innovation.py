from mesa import Model
from mesa.datacollection import DataCollector
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


class DisruptiveInnovationAgent(InnovationAgent):
    """An agent in a disruptive innovation model.
    The agent can choose between an incumbent and a disruptive product.
    """

    def __init__(self, unique_id, model):
        args = [unique_id, model]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁDisruptiveInnovationAgentǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁDisruptiveInnovationAgentǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁDisruptiveInnovationAgentǁ__init____mutmut_orig(self, unique_id, model):
        super().__init__(unique_id, model)
        self.choice = None  # 'incumbent' or 'disruptive'

    def xǁDisruptiveInnovationAgentǁ__init____mutmut_1(self, unique_id, model):
        super().__init__(None, model)
        self.choice = None  # 'incumbent' or 'disruptive'

    def xǁDisruptiveInnovationAgentǁ__init____mutmut_2(self, unique_id, model):
        super().__init__(unique_id, None)
        self.choice = None  # 'incumbent' or 'disruptive'

    def xǁDisruptiveInnovationAgentǁ__init____mutmut_3(self, unique_id, model):
        super().__init__(model)
        self.choice = None  # 'incumbent' or 'disruptive'

    def xǁDisruptiveInnovationAgentǁ__init____mutmut_4(self, unique_id, model):
        super().__init__(unique_id, )
        self.choice = None  # 'incumbent' or 'disruptive'

    def xǁDisruptiveInnovationAgentǁ__init____mutmut_5(self, unique_id, model):
        super().__init__(unique_id, model)
        self.choice = ""  # 'incumbent' or 'disruptive'
    
    xǁDisruptiveInnovationAgentǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁDisruptiveInnovationAgentǁ__init____mutmut_1': xǁDisruptiveInnovationAgentǁ__init____mutmut_1, 
        'xǁDisruptiveInnovationAgentǁ__init____mutmut_2': xǁDisruptiveInnovationAgentǁ__init____mutmut_2, 
        'xǁDisruptiveInnovationAgentǁ__init____mutmut_3': xǁDisruptiveInnovationAgentǁ__init____mutmut_3, 
        'xǁDisruptiveInnovationAgentǁ__init____mutmut_4': xǁDisruptiveInnovationAgentǁ__init____mutmut_4, 
        'xǁDisruptiveInnovationAgentǁ__init____mutmut_5': xǁDisruptiveInnovationAgentǁ__init____mutmut_5
    }
    xǁDisruptiveInnovationAgentǁ__init____mutmut_orig.__name__ = 'xǁDisruptiveInnovationAgentǁ__init__'

    def step(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁDisruptiveInnovationAgentǁstep__mutmut_orig'), object.__getattribute__(self, 'xǁDisruptiveInnovationAgentǁstep__mutmut_mutants'), args, kwargs, self)

    def xǁDisruptiveInnovationAgentǁstep__mutmut_orig(self):
        """The agent's step function.
        The agent's choice is based on the perceived value of each product.
        """
        incumbent_value = self.model.incumbent_performance - self.model.incumbent_price
        disruptive_value = self.model.disruptive_performance - self.model.disruptive_price

        if disruptive_value > incumbent_value:
            self.choice = "disruptive"
        else:
            self.choice = "incumbent"

    def xǁDisruptiveInnovationAgentǁstep__mutmut_1(self):
        """The agent's step function.
        The agent's choice is based on the perceived value of each product.
        """
        incumbent_value = None
        disruptive_value = self.model.disruptive_performance - self.model.disruptive_price

        if disruptive_value > incumbent_value:
            self.choice = "disruptive"
        else:
            self.choice = "incumbent"

    def xǁDisruptiveInnovationAgentǁstep__mutmut_2(self):
        """The agent's step function.
        The agent's choice is based on the perceived value of each product.
        """
        incumbent_value = self.model.incumbent_performance + self.model.incumbent_price
        disruptive_value = self.model.disruptive_performance - self.model.disruptive_price

        if disruptive_value > incumbent_value:
            self.choice = "disruptive"
        else:
            self.choice = "incumbent"

    def xǁDisruptiveInnovationAgentǁstep__mutmut_3(self):
        """The agent's step function.
        The agent's choice is based on the perceived value of each product.
        """
        incumbent_value = self.model.incumbent_performance - self.model.incumbent_price
        disruptive_value = None

        if disruptive_value > incumbent_value:
            self.choice = "disruptive"
        else:
            self.choice = "incumbent"

    def xǁDisruptiveInnovationAgentǁstep__mutmut_4(self):
        """The agent's step function.
        The agent's choice is based on the perceived value of each product.
        """
        incumbent_value = self.model.incumbent_performance - self.model.incumbent_price
        disruptive_value = self.model.disruptive_performance + self.model.disruptive_price

        if disruptive_value > incumbent_value:
            self.choice = "disruptive"
        else:
            self.choice = "incumbent"

    def xǁDisruptiveInnovationAgentǁstep__mutmut_5(self):
        """The agent's step function.
        The agent's choice is based on the perceived value of each product.
        """
        incumbent_value = self.model.incumbent_performance - self.model.incumbent_price
        disruptive_value = self.model.disruptive_performance - self.model.disruptive_price

        if disruptive_value >= incumbent_value:
            self.choice = "disruptive"
        else:
            self.choice = "incumbent"

    def xǁDisruptiveInnovationAgentǁstep__mutmut_6(self):
        """The agent's step function.
        The agent's choice is based on the perceived value of each product.
        """
        incumbent_value = self.model.incumbent_performance - self.model.incumbent_price
        disruptive_value = self.model.disruptive_performance - self.model.disruptive_price

        if disruptive_value > incumbent_value:
            self.choice = None
        else:
            self.choice = "incumbent"

    def xǁDisruptiveInnovationAgentǁstep__mutmut_7(self):
        """The agent's step function.
        The agent's choice is based on the perceived value of each product.
        """
        incumbent_value = self.model.incumbent_performance - self.model.incumbent_price
        disruptive_value = self.model.disruptive_performance - self.model.disruptive_price

        if disruptive_value > incumbent_value:
            self.choice = "XXdisruptiveXX"
        else:
            self.choice = "incumbent"

    def xǁDisruptiveInnovationAgentǁstep__mutmut_8(self):
        """The agent's step function.
        The agent's choice is based on the perceived value of each product.
        """
        incumbent_value = self.model.incumbent_performance - self.model.incumbent_price
        disruptive_value = self.model.disruptive_performance - self.model.disruptive_price

        if disruptive_value > incumbent_value:
            self.choice = "DISRUPTIVE"
        else:
            self.choice = "incumbent"

    def xǁDisruptiveInnovationAgentǁstep__mutmut_9(self):
        """The agent's step function.
        The agent's choice is based on the perceived value of each product.
        """
        incumbent_value = self.model.incumbent_performance - self.model.incumbent_price
        disruptive_value = self.model.disruptive_performance - self.model.disruptive_price

        if disruptive_value > incumbent_value:
            self.choice = "disruptive"
        else:
            self.choice = None

    def xǁDisruptiveInnovationAgentǁstep__mutmut_10(self):
        """The agent's step function.
        The agent's choice is based on the perceived value of each product.
        """
        incumbent_value = self.model.incumbent_performance - self.model.incumbent_price
        disruptive_value = self.model.disruptive_performance - self.model.disruptive_price

        if disruptive_value > incumbent_value:
            self.choice = "disruptive"
        else:
            self.choice = "XXincumbentXX"

    def xǁDisruptiveInnovationAgentǁstep__mutmut_11(self):
        """The agent's step function.
        The agent's choice is based on the perceived value of each product.
        """
        incumbent_value = self.model.incumbent_performance - self.model.incumbent_price
        disruptive_value = self.model.disruptive_performance - self.model.disruptive_price

        if disruptive_value > incumbent_value:
            self.choice = "disruptive"
        else:
            self.choice = "INCUMBENT"
    
    xǁDisruptiveInnovationAgentǁstep__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁDisruptiveInnovationAgentǁstep__mutmut_1': xǁDisruptiveInnovationAgentǁstep__mutmut_1, 
        'xǁDisruptiveInnovationAgentǁstep__mutmut_2': xǁDisruptiveInnovationAgentǁstep__mutmut_2, 
        'xǁDisruptiveInnovationAgentǁstep__mutmut_3': xǁDisruptiveInnovationAgentǁstep__mutmut_3, 
        'xǁDisruptiveInnovationAgentǁstep__mutmut_4': xǁDisruptiveInnovationAgentǁstep__mutmut_4, 
        'xǁDisruptiveInnovationAgentǁstep__mutmut_5': xǁDisruptiveInnovationAgentǁstep__mutmut_5, 
        'xǁDisruptiveInnovationAgentǁstep__mutmut_6': xǁDisruptiveInnovationAgentǁstep__mutmut_6, 
        'xǁDisruptiveInnovationAgentǁstep__mutmut_7': xǁDisruptiveInnovationAgentǁstep__mutmut_7, 
        'xǁDisruptiveInnovationAgentǁstep__mutmut_8': xǁDisruptiveInnovationAgentǁstep__mutmut_8, 
        'xǁDisruptiveInnovationAgentǁstep__mutmut_9': xǁDisruptiveInnovationAgentǁstep__mutmut_9, 
        'xǁDisruptiveInnovationAgentǁstep__mutmut_10': xǁDisruptiveInnovationAgentǁstep__mutmut_10, 
        'xǁDisruptiveInnovationAgentǁstep__mutmut_11': xǁDisruptiveInnovationAgentǁstep__mutmut_11
    }
    xǁDisruptiveInnovationAgentǁstep__mutmut_orig.__name__ = 'xǁDisruptiveInnovationAgentǁstep'


class DisruptiveInnovationModel(Model):
    """A model for disruptive innovation."""

    def __init__(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        args = [num_agents, width, height, initial_disruptive_performance, disruptive_performance_improvement]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁDisruptiveInnovationModelǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁDisruptiveInnovationModelǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁDisruptiveInnovationModelǁ__init____mutmut_orig(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_1(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = None
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_2(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = None
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_3(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(None, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_4(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, None, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_5(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, None)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_6(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_7(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_8(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, )
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_9(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, False)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_10(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = None

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_11(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = False

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_12(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = None
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_13(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 2.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_14(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = None
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_15(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 1.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_16(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = None
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_17(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = None
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_18(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 1.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_19(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = None

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_20(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(None):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_21(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = None
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_22(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=None, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_23(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=None)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_24(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_25(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_26(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = None
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_27(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(None)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_28(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = None
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_29(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(None)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_30(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(None, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_31(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, None)

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_32(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent((x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_33(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, )

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_34(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = None

    def xǁDisruptiveInnovationModelǁ__init____mutmut_35(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters=None,
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_36(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "XXIncumbentAdoptersXX": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_37(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "incumbentadopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_38(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "INCUMBENTADOPTERS": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_39(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: None,
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_40(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    None,
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_41(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [2 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_42(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice != "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_43(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "XXincumbentXX"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_44(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "INCUMBENT"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_45(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "XXDisruptiveAdoptersXX": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_46(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "disruptiveadopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_47(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DISRUPTIVEADOPTERS": lambda m: sum(
                    [1 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_48(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: None,
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_49(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    None,
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_50(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [2 for a in m.agents if a.choice == "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_51(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice != "disruptive"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_52(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "XXdisruptiveXX"],
                ),
            },
        )

    def xǁDisruptiveInnovationModelǁ__init____mutmut_53(
        self,
        num_agents,
        width,
        height,
        initial_disruptive_performance,
        disruptive_performance_improvement,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        for i in range(self.num_agents):
            agent = DisruptiveInnovationAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "incumbent"],
                ),
                "DisruptiveAdopters": lambda m: sum(
                    [1 for a in m.agents if a.choice == "DISRUPTIVE"],
                ),
            },
        )
    
    xǁDisruptiveInnovationModelǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁDisruptiveInnovationModelǁ__init____mutmut_1': xǁDisruptiveInnovationModelǁ__init____mutmut_1, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_2': xǁDisruptiveInnovationModelǁ__init____mutmut_2, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_3': xǁDisruptiveInnovationModelǁ__init____mutmut_3, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_4': xǁDisruptiveInnovationModelǁ__init____mutmut_4, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_5': xǁDisruptiveInnovationModelǁ__init____mutmut_5, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_6': xǁDisruptiveInnovationModelǁ__init____mutmut_6, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_7': xǁDisruptiveInnovationModelǁ__init____mutmut_7, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_8': xǁDisruptiveInnovationModelǁ__init____mutmut_8, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_9': xǁDisruptiveInnovationModelǁ__init____mutmut_9, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_10': xǁDisruptiveInnovationModelǁ__init____mutmut_10, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_11': xǁDisruptiveInnovationModelǁ__init____mutmut_11, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_12': xǁDisruptiveInnovationModelǁ__init____mutmut_12, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_13': xǁDisruptiveInnovationModelǁ__init____mutmut_13, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_14': xǁDisruptiveInnovationModelǁ__init____mutmut_14, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_15': xǁDisruptiveInnovationModelǁ__init____mutmut_15, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_16': xǁDisruptiveInnovationModelǁ__init____mutmut_16, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_17': xǁDisruptiveInnovationModelǁ__init____mutmut_17, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_18': xǁDisruptiveInnovationModelǁ__init____mutmut_18, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_19': xǁDisruptiveInnovationModelǁ__init____mutmut_19, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_20': xǁDisruptiveInnovationModelǁ__init____mutmut_20, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_21': xǁDisruptiveInnovationModelǁ__init____mutmut_21, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_22': xǁDisruptiveInnovationModelǁ__init____mutmut_22, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_23': xǁDisruptiveInnovationModelǁ__init____mutmut_23, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_24': xǁDisruptiveInnovationModelǁ__init____mutmut_24, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_25': xǁDisruptiveInnovationModelǁ__init____mutmut_25, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_26': xǁDisruptiveInnovationModelǁ__init____mutmut_26, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_27': xǁDisruptiveInnovationModelǁ__init____mutmut_27, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_28': xǁDisruptiveInnovationModelǁ__init____mutmut_28, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_29': xǁDisruptiveInnovationModelǁ__init____mutmut_29, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_30': xǁDisruptiveInnovationModelǁ__init____mutmut_30, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_31': xǁDisruptiveInnovationModelǁ__init____mutmut_31, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_32': xǁDisruptiveInnovationModelǁ__init____mutmut_32, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_33': xǁDisruptiveInnovationModelǁ__init____mutmut_33, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_34': xǁDisruptiveInnovationModelǁ__init____mutmut_34, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_35': xǁDisruptiveInnovationModelǁ__init____mutmut_35, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_36': xǁDisruptiveInnovationModelǁ__init____mutmut_36, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_37': xǁDisruptiveInnovationModelǁ__init____mutmut_37, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_38': xǁDisruptiveInnovationModelǁ__init____mutmut_38, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_39': xǁDisruptiveInnovationModelǁ__init____mutmut_39, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_40': xǁDisruptiveInnovationModelǁ__init____mutmut_40, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_41': xǁDisruptiveInnovationModelǁ__init____mutmut_41, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_42': xǁDisruptiveInnovationModelǁ__init____mutmut_42, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_43': xǁDisruptiveInnovationModelǁ__init____mutmut_43, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_44': xǁDisruptiveInnovationModelǁ__init____mutmut_44, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_45': xǁDisruptiveInnovationModelǁ__init____mutmut_45, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_46': xǁDisruptiveInnovationModelǁ__init____mutmut_46, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_47': xǁDisruptiveInnovationModelǁ__init____mutmut_47, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_48': xǁDisruptiveInnovationModelǁ__init____mutmut_48, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_49': xǁDisruptiveInnovationModelǁ__init____mutmut_49, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_50': xǁDisruptiveInnovationModelǁ__init____mutmut_50, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_51': xǁDisruptiveInnovationModelǁ__init____mutmut_51, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_52': xǁDisruptiveInnovationModelǁ__init____mutmut_52, 
        'xǁDisruptiveInnovationModelǁ__init____mutmut_53': xǁDisruptiveInnovationModelǁ__init____mutmut_53
    }
    xǁDisruptiveInnovationModelǁ__init____mutmut_orig.__name__ = 'xǁDisruptiveInnovationModelǁ__init__'

    def step(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁDisruptiveInnovationModelǁstep__mutmut_orig'), object.__getattribute__(self, 'xǁDisruptiveInnovationModelǁstep__mutmut_mutants'), args, kwargs, self)

    def xǁDisruptiveInnovationModelǁstep__mutmut_orig(self):
        """Run one step of the model."""
        self.disruptive_performance += self.disruptive_performance_improvement
        self.datacollector.collect(self)
        self.agents.do("step")

    def xǁDisruptiveInnovationModelǁstep__mutmut_1(self):
        """Run one step of the model."""
        self.disruptive_performance = self.disruptive_performance_improvement
        self.datacollector.collect(self)
        self.agents.do("step")

    def xǁDisruptiveInnovationModelǁstep__mutmut_2(self):
        """Run one step of the model."""
        self.disruptive_performance -= self.disruptive_performance_improvement
        self.datacollector.collect(self)
        self.agents.do("step")

    def xǁDisruptiveInnovationModelǁstep__mutmut_3(self):
        """Run one step of the model."""
        self.disruptive_performance += self.disruptive_performance_improvement
        self.datacollector.collect(None)
        self.agents.do("step")

    def xǁDisruptiveInnovationModelǁstep__mutmut_4(self):
        """Run one step of the model."""
        self.disruptive_performance += self.disruptive_performance_improvement
        self.datacollector.collect(self)
        self.agents.do(None)

    def xǁDisruptiveInnovationModelǁstep__mutmut_5(self):
        """Run one step of the model."""
        self.disruptive_performance += self.disruptive_performance_improvement
        self.datacollector.collect(self)
        self.agents.do("XXstepXX")

    def xǁDisruptiveInnovationModelǁstep__mutmut_6(self):
        """Run one step of the model."""
        self.disruptive_performance += self.disruptive_performance_improvement
        self.datacollector.collect(self)
        self.agents.do("STEP")
    
    xǁDisruptiveInnovationModelǁstep__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁDisruptiveInnovationModelǁstep__mutmut_1': xǁDisruptiveInnovationModelǁstep__mutmut_1, 
        'xǁDisruptiveInnovationModelǁstep__mutmut_2': xǁDisruptiveInnovationModelǁstep__mutmut_2, 
        'xǁDisruptiveInnovationModelǁstep__mutmut_3': xǁDisruptiveInnovationModelǁstep__mutmut_3, 
        'xǁDisruptiveInnovationModelǁstep__mutmut_4': xǁDisruptiveInnovationModelǁstep__mutmut_4, 
        'xǁDisruptiveInnovationModelǁstep__mutmut_5': xǁDisruptiveInnovationModelǁstep__mutmut_5, 
        'xǁDisruptiveInnovationModelǁstep__mutmut_6': xǁDisruptiveInnovationModelǁstep__mutmut_6
    }
    xǁDisruptiveInnovationModelǁstep__mutmut_orig.__name__ = 'xǁDisruptiveInnovationModelǁstep'

    def run_model(self, n_steps):
        args = [n_steps]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁDisruptiveInnovationModelǁrun_model__mutmut_orig'), object.__getattribute__(self, 'xǁDisruptiveInnovationModelǁrun_model__mutmut_mutants'), args, kwargs, self)

    def xǁDisruptiveInnovationModelǁrun_model__mutmut_orig(self, n_steps):
        """Run the model for a specified number of steps."""
        for _ in range(n_steps):
            self.step()
        return self.datacollector.get_model_vars_dataframe()

    def xǁDisruptiveInnovationModelǁrun_model__mutmut_1(self, n_steps):
        """Run the model for a specified number of steps."""
        for _ in range(None):
            self.step()
        return self.datacollector.get_model_vars_dataframe()
    
    xǁDisruptiveInnovationModelǁrun_model__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁDisruptiveInnovationModelǁrun_model__mutmut_1': xǁDisruptiveInnovationModelǁrun_model__mutmut_1
    }
    xǁDisruptiveInnovationModelǁrun_model__mutmut_orig.__name__ = 'xǁDisruptiveInnovationModelǁrun_model'
