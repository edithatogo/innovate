from mesa import Agent
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


class InnovationAgent(Agent):
    """An agent in the innovation diffusion model."""

    def __init__(self, unique_id, model):
        args = [unique_id, model]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁInnovationAgentǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁInnovationAgentǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁInnovationAgentǁ__init____mutmut_orig(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id
        # Add agent-specific attributes here, e.g.,
        self.adopted = False
        self.awareness = 0.0  # 0.0 to 1.0
        self.influence = 0.0  # How much this agent influences others
        self.susceptibility = 0.0  # How susceptible this agent is to influence

    def xǁInnovationAgentǁ__init____mutmut_1(self, unique_id, model):
        super().__init__(None)
        self.unique_id = unique_id
        # Add agent-specific attributes here, e.g.,
        self.adopted = False
        self.awareness = 0.0  # 0.0 to 1.0
        self.influence = 0.0  # How much this agent influences others
        self.susceptibility = 0.0  # How susceptible this agent is to influence

    def xǁInnovationAgentǁ__init____mutmut_2(self, unique_id, model):
        super().__init__(model)
        self.unique_id = None
        # Add agent-specific attributes here, e.g.,
        self.adopted = False
        self.awareness = 0.0  # 0.0 to 1.0
        self.influence = 0.0  # How much this agent influences others
        self.susceptibility = 0.0  # How susceptible this agent is to influence

    def xǁInnovationAgentǁ__init____mutmut_3(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id
        # Add agent-specific attributes here, e.g.,
        self.adopted = None
        self.awareness = 0.0  # 0.0 to 1.0
        self.influence = 0.0  # How much this agent influences others
        self.susceptibility = 0.0  # How susceptible this agent is to influence

    def xǁInnovationAgentǁ__init____mutmut_4(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id
        # Add agent-specific attributes here, e.g.,
        self.adopted = True
        self.awareness = 0.0  # 0.0 to 1.0
        self.influence = 0.0  # How much this agent influences others
        self.susceptibility = 0.0  # How susceptible this agent is to influence

    def xǁInnovationAgentǁ__init____mutmut_5(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id
        # Add agent-specific attributes here, e.g.,
        self.adopted = False
        self.awareness = None  # 0.0 to 1.0
        self.influence = 0.0  # How much this agent influences others
        self.susceptibility = 0.0  # How susceptible this agent is to influence

    def xǁInnovationAgentǁ__init____mutmut_6(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id
        # Add agent-specific attributes here, e.g.,
        self.adopted = False
        self.awareness = 1.0  # 0.0 to 1.0
        self.influence = 0.0  # How much this agent influences others
        self.susceptibility = 0.0  # How susceptible this agent is to influence

    def xǁInnovationAgentǁ__init____mutmut_7(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id
        # Add agent-specific attributes here, e.g.,
        self.adopted = False
        self.awareness = 0.0  # 0.0 to 1.0
        self.influence = None  # How much this agent influences others
        self.susceptibility = 0.0  # How susceptible this agent is to influence

    def xǁInnovationAgentǁ__init____mutmut_8(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id
        # Add agent-specific attributes here, e.g.,
        self.adopted = False
        self.awareness = 0.0  # 0.0 to 1.0
        self.influence = 1.0  # How much this agent influences others
        self.susceptibility = 0.0  # How susceptible this agent is to influence

    def xǁInnovationAgentǁ__init____mutmut_9(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id
        # Add agent-specific attributes here, e.g.,
        self.adopted = False
        self.awareness = 0.0  # 0.0 to 1.0
        self.influence = 0.0  # How much this agent influences others
        self.susceptibility = None  # How susceptible this agent is to influence

    def xǁInnovationAgentǁ__init____mutmut_10(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id
        # Add agent-specific attributes here, e.g.,
        self.adopted = False
        self.awareness = 0.0  # 0.0 to 1.0
        self.influence = 0.0  # How much this agent influences others
        self.susceptibility = 1.0  # How susceptible this agent is to influence
    
    xǁInnovationAgentǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁInnovationAgentǁ__init____mutmut_1': xǁInnovationAgentǁ__init____mutmut_1, 
        'xǁInnovationAgentǁ__init____mutmut_2': xǁInnovationAgentǁ__init____mutmut_2, 
        'xǁInnovationAgentǁ__init____mutmut_3': xǁInnovationAgentǁ__init____mutmut_3, 
        'xǁInnovationAgentǁ__init____mutmut_4': xǁInnovationAgentǁ__init____mutmut_4, 
        'xǁInnovationAgentǁ__init____mutmut_5': xǁInnovationAgentǁ__init____mutmut_5, 
        'xǁInnovationAgentǁ__init____mutmut_6': xǁInnovationAgentǁ__init____mutmut_6, 
        'xǁInnovationAgentǁ__init____mutmut_7': xǁInnovationAgentǁ__init____mutmut_7, 
        'xǁInnovationAgentǁ__init____mutmut_8': xǁInnovationAgentǁ__init____mutmut_8, 
        'xǁInnovationAgentǁ__init____mutmut_9': xǁInnovationAgentǁ__init____mutmut_9, 
        'xǁInnovationAgentǁ__init____mutmut_10': xǁInnovationAgentǁ__init____mutmut_10
    }
    xǁInnovationAgentǁ__init____mutmut_orig.__name__ = 'xǁInnovationAgentǁ__init__'

    def step(self):
        """Agent's behavior at each step."""
        # Implement agent's decision-making process here
