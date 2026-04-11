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


class CompetitiveDiffusionAgent(InnovationAgent):
    """An agent in a competitive diffusion model.
    The agent can adopt one of several competing innovations.
    """

    def __init__(self, unique_id, model, num_innovations):
        args = [unique_id, model, num_innovations]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁCompetitiveDiffusionAgentǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁCompetitiveDiffusionAgentǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁCompetitiveDiffusionAgentǁ__init____mutmut_orig(self, unique_id, model, num_innovations):
        super().__init__(unique_id, model)
        self.adopted_innovation = -1  # -1 means no adoption, 0, 1, ... for innovations

    def xǁCompetitiveDiffusionAgentǁ__init____mutmut_1(self, unique_id, model, num_innovations):
        super().__init__(None, model)
        self.adopted_innovation = -1  # -1 means no adoption, 0, 1, ... for innovations

    def xǁCompetitiveDiffusionAgentǁ__init____mutmut_2(self, unique_id, model, num_innovations):
        super().__init__(unique_id, None)
        self.adopted_innovation = -1  # -1 means no adoption, 0, 1, ... for innovations

    def xǁCompetitiveDiffusionAgentǁ__init____mutmut_3(self, unique_id, model, num_innovations):
        super().__init__(model)
        self.adopted_innovation = -1  # -1 means no adoption, 0, 1, ... for innovations

    def xǁCompetitiveDiffusionAgentǁ__init____mutmut_4(self, unique_id, model, num_innovations):
        super().__init__(unique_id, )
        self.adopted_innovation = -1  # -1 means no adoption, 0, 1, ... for innovations

    def xǁCompetitiveDiffusionAgentǁ__init____mutmut_5(self, unique_id, model, num_innovations):
        super().__init__(unique_id, model)
        self.adopted_innovation = None  # -1 means no adoption, 0, 1, ... for innovations

    def xǁCompetitiveDiffusionAgentǁ__init____mutmut_6(self, unique_id, model, num_innovations):
        super().__init__(unique_id, model)
        self.adopted_innovation = +1  # -1 means no adoption, 0, 1, ... for innovations

    def xǁCompetitiveDiffusionAgentǁ__init____mutmut_7(self, unique_id, model, num_innovations):
        super().__init__(unique_id, model)
        self.adopted_innovation = -2  # -1 means no adoption, 0, 1, ... for innovations
    
    xǁCompetitiveDiffusionAgentǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁCompetitiveDiffusionAgentǁ__init____mutmut_1': xǁCompetitiveDiffusionAgentǁ__init____mutmut_1, 
        'xǁCompetitiveDiffusionAgentǁ__init____mutmut_2': xǁCompetitiveDiffusionAgentǁ__init____mutmut_2, 
        'xǁCompetitiveDiffusionAgentǁ__init____mutmut_3': xǁCompetitiveDiffusionAgentǁ__init____mutmut_3, 
        'xǁCompetitiveDiffusionAgentǁ__init____mutmut_4': xǁCompetitiveDiffusionAgentǁ__init____mutmut_4, 
        'xǁCompetitiveDiffusionAgentǁ__init____mutmut_5': xǁCompetitiveDiffusionAgentǁ__init____mutmut_5, 
        'xǁCompetitiveDiffusionAgentǁ__init____mutmut_6': xǁCompetitiveDiffusionAgentǁ__init____mutmut_6, 
        'xǁCompetitiveDiffusionAgentǁ__init____mutmut_7': xǁCompetitiveDiffusionAgentǁ__init____mutmut_7
    }
    xǁCompetitiveDiffusionAgentǁ__init____mutmut_orig.__name__ = 'xǁCompetitiveDiffusionAgentǁ__init__'

    def step(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁCompetitiveDiffusionAgentǁstep__mutmut_orig'), object.__getattribute__(self, 'xǁCompetitiveDiffusionAgentǁstep__mutmut_mutants'), args, kwargs, self)

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_orig(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_1(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation == -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_2(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != +1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_3(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -2:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_4(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = None
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_5(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            None,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_6(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=None,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_7(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=None,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_8(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_9(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_10(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_11(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=False,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_12(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=True,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_13(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_14(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = None
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_15(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation == -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_16(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != +1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_17(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -2]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_18(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_19(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = None
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_20(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(None, key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_21(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=None)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_22(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_23(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), )
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_24(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(None), key=adoptions.count)
        self.adopted_innovation = most_popular

    def xǁCompetitiveDiffusionAgentǁstep__mutmut_25(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = None
    
    xǁCompetitiveDiffusionAgentǁstep__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁCompetitiveDiffusionAgentǁstep__mutmut_1': xǁCompetitiveDiffusionAgentǁstep__mutmut_1, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_2': xǁCompetitiveDiffusionAgentǁstep__mutmut_2, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_3': xǁCompetitiveDiffusionAgentǁstep__mutmut_3, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_4': xǁCompetitiveDiffusionAgentǁstep__mutmut_4, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_5': xǁCompetitiveDiffusionAgentǁstep__mutmut_5, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_6': xǁCompetitiveDiffusionAgentǁstep__mutmut_6, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_7': xǁCompetitiveDiffusionAgentǁstep__mutmut_7, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_8': xǁCompetitiveDiffusionAgentǁstep__mutmut_8, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_9': xǁCompetitiveDiffusionAgentǁstep__mutmut_9, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_10': xǁCompetitiveDiffusionAgentǁstep__mutmut_10, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_11': xǁCompetitiveDiffusionAgentǁstep__mutmut_11, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_12': xǁCompetitiveDiffusionAgentǁstep__mutmut_12, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_13': xǁCompetitiveDiffusionAgentǁstep__mutmut_13, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_14': xǁCompetitiveDiffusionAgentǁstep__mutmut_14, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_15': xǁCompetitiveDiffusionAgentǁstep__mutmut_15, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_16': xǁCompetitiveDiffusionAgentǁstep__mutmut_16, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_17': xǁCompetitiveDiffusionAgentǁstep__mutmut_17, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_18': xǁCompetitiveDiffusionAgentǁstep__mutmut_18, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_19': xǁCompetitiveDiffusionAgentǁstep__mutmut_19, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_20': xǁCompetitiveDiffusionAgentǁstep__mutmut_20, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_21': xǁCompetitiveDiffusionAgentǁstep__mutmut_21, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_22': xǁCompetitiveDiffusionAgentǁstep__mutmut_22, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_23': xǁCompetitiveDiffusionAgentǁstep__mutmut_23, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_24': xǁCompetitiveDiffusionAgentǁstep__mutmut_24, 
        'xǁCompetitiveDiffusionAgentǁstep__mutmut_25': xǁCompetitiveDiffusionAgentǁstep__mutmut_25
    }
    xǁCompetitiveDiffusionAgentǁstep__mutmut_orig.__name__ = 'xǁCompetitiveDiffusionAgentǁstep'


class CompetitiveDiffusionModel(Model):
    """A model for competitive diffusion of multiple innovations."""

    def __init__(self, num_agents, width, height, num_innovations):
        args = [num_agents, width, height, num_innovations]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁCompetitiveDiffusionModelǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁCompetitiveDiffusionModelǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_orig(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_1(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = None
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_2(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = None
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_3(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = None
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_4(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(None, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_5(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, None, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_6(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, None)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_7(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_8(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_9(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, )
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_10(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, False)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_11(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = None

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_12(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = False

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_13(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(None):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_14(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = None
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_15(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=None,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_16(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=None,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_17(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=None,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_18(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_19(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_20(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_21(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = None
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_22(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(None)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_23(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = None
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_24(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(None)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_25(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(None, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_26(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, None)

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_27(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent((x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_28(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, )

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_29(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(None):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_30(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = None
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_31(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(None)
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_32(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(None))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_33(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = None

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_34(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = None
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_35(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] / m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_36(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [1] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_37(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation == -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_38(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != +1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_39(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -2:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_40(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] = 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_41(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] -= 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_42(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 2
            return counts

        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_43(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = None

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_44(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters=None,
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_45(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"XXAdoptionCountsXX": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_46(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"adoptioncounts": adoption_counts},
        )

    def xǁCompetitiveDiffusionModelǁ__init____mutmut_47(self, num_agents, width, height, num_innovations):
        super().__init__()
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = CompetitiveDiffusionAgent(
                unique_id=i,
                model=self,
                num_innovations=num_innovations,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector - track count of adopters for each innovation
        def adoption_counts(m):
            counts = [0] * m.num_innovations
            for a in m.agents:
                if a.adopted_innovation != -1:
                    counts[a.adopted_innovation] += 1
            return counts

        self.datacollector = DataCollector(
            model_reporters={"ADOPTIONCOUNTS": adoption_counts},
        )
    
    xǁCompetitiveDiffusionModelǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁCompetitiveDiffusionModelǁ__init____mutmut_1': xǁCompetitiveDiffusionModelǁ__init____mutmut_1, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_2': xǁCompetitiveDiffusionModelǁ__init____mutmut_2, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_3': xǁCompetitiveDiffusionModelǁ__init____mutmut_3, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_4': xǁCompetitiveDiffusionModelǁ__init____mutmut_4, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_5': xǁCompetitiveDiffusionModelǁ__init____mutmut_5, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_6': xǁCompetitiveDiffusionModelǁ__init____mutmut_6, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_7': xǁCompetitiveDiffusionModelǁ__init____mutmut_7, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_8': xǁCompetitiveDiffusionModelǁ__init____mutmut_8, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_9': xǁCompetitiveDiffusionModelǁ__init____mutmut_9, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_10': xǁCompetitiveDiffusionModelǁ__init____mutmut_10, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_11': xǁCompetitiveDiffusionModelǁ__init____mutmut_11, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_12': xǁCompetitiveDiffusionModelǁ__init____mutmut_12, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_13': xǁCompetitiveDiffusionModelǁ__init____mutmut_13, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_14': xǁCompetitiveDiffusionModelǁ__init____mutmut_14, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_15': xǁCompetitiveDiffusionModelǁ__init____mutmut_15, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_16': xǁCompetitiveDiffusionModelǁ__init____mutmut_16, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_17': xǁCompetitiveDiffusionModelǁ__init____mutmut_17, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_18': xǁCompetitiveDiffusionModelǁ__init____mutmut_18, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_19': xǁCompetitiveDiffusionModelǁ__init____mutmut_19, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_20': xǁCompetitiveDiffusionModelǁ__init____mutmut_20, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_21': xǁCompetitiveDiffusionModelǁ__init____mutmut_21, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_22': xǁCompetitiveDiffusionModelǁ__init____mutmut_22, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_23': xǁCompetitiveDiffusionModelǁ__init____mutmut_23, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_24': xǁCompetitiveDiffusionModelǁ__init____mutmut_24, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_25': xǁCompetitiveDiffusionModelǁ__init____mutmut_25, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_26': xǁCompetitiveDiffusionModelǁ__init____mutmut_26, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_27': xǁCompetitiveDiffusionModelǁ__init____mutmut_27, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_28': xǁCompetitiveDiffusionModelǁ__init____mutmut_28, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_29': xǁCompetitiveDiffusionModelǁ__init____mutmut_29, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_30': xǁCompetitiveDiffusionModelǁ__init____mutmut_30, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_31': xǁCompetitiveDiffusionModelǁ__init____mutmut_31, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_32': xǁCompetitiveDiffusionModelǁ__init____mutmut_32, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_33': xǁCompetitiveDiffusionModelǁ__init____mutmut_33, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_34': xǁCompetitiveDiffusionModelǁ__init____mutmut_34, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_35': xǁCompetitiveDiffusionModelǁ__init____mutmut_35, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_36': xǁCompetitiveDiffusionModelǁ__init____mutmut_36, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_37': xǁCompetitiveDiffusionModelǁ__init____mutmut_37, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_38': xǁCompetitiveDiffusionModelǁ__init____mutmut_38, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_39': xǁCompetitiveDiffusionModelǁ__init____mutmut_39, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_40': xǁCompetitiveDiffusionModelǁ__init____mutmut_40, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_41': xǁCompetitiveDiffusionModelǁ__init____mutmut_41, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_42': xǁCompetitiveDiffusionModelǁ__init____mutmut_42, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_43': xǁCompetitiveDiffusionModelǁ__init____mutmut_43, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_44': xǁCompetitiveDiffusionModelǁ__init____mutmut_44, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_45': xǁCompetitiveDiffusionModelǁ__init____mutmut_45, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_46': xǁCompetitiveDiffusionModelǁ__init____mutmut_46, 
        'xǁCompetitiveDiffusionModelǁ__init____mutmut_47': xǁCompetitiveDiffusionModelǁ__init____mutmut_47
    }
    xǁCompetitiveDiffusionModelǁ__init____mutmut_orig.__name__ = 'xǁCompetitiveDiffusionModelǁ__init__'

    def step(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁCompetitiveDiffusionModelǁstep__mutmut_orig'), object.__getattribute__(self, 'xǁCompetitiveDiffusionModelǁstep__mutmut_mutants'), args, kwargs, self)

    def xǁCompetitiveDiffusionModelǁstep__mutmut_orig(self):
        """Run one step of the model."""
        self.datacollector.collect(self)
        self.agents.do("step")

    def xǁCompetitiveDiffusionModelǁstep__mutmut_1(self):
        """Run one step of the model."""
        self.datacollector.collect(None)
        self.agents.do("step")

    def xǁCompetitiveDiffusionModelǁstep__mutmut_2(self):
        """Run one step of the model."""
        self.datacollector.collect(self)
        self.agents.do(None)

    def xǁCompetitiveDiffusionModelǁstep__mutmut_3(self):
        """Run one step of the model."""
        self.datacollector.collect(self)
        self.agents.do("XXstepXX")

    def xǁCompetitiveDiffusionModelǁstep__mutmut_4(self):
        """Run one step of the model."""
        self.datacollector.collect(self)
        self.agents.do("STEP")
    
    xǁCompetitiveDiffusionModelǁstep__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁCompetitiveDiffusionModelǁstep__mutmut_1': xǁCompetitiveDiffusionModelǁstep__mutmut_1, 
        'xǁCompetitiveDiffusionModelǁstep__mutmut_2': xǁCompetitiveDiffusionModelǁstep__mutmut_2, 
        'xǁCompetitiveDiffusionModelǁstep__mutmut_3': xǁCompetitiveDiffusionModelǁstep__mutmut_3, 
        'xǁCompetitiveDiffusionModelǁstep__mutmut_4': xǁCompetitiveDiffusionModelǁstep__mutmut_4
    }
    xǁCompetitiveDiffusionModelǁstep__mutmut_orig.__name__ = 'xǁCompetitiveDiffusionModelǁstep'

    def run_model(self, n_steps):
        args = [n_steps]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁCompetitiveDiffusionModelǁrun_model__mutmut_orig'), object.__getattribute__(self, 'xǁCompetitiveDiffusionModelǁrun_model__mutmut_mutants'), args, kwargs, self)

    def xǁCompetitiveDiffusionModelǁrun_model__mutmut_orig(self, n_steps):
        """Run the model for a specified number of steps."""
        for _ in range(n_steps):
            self.step()
        return self.datacollector.get_model_vars_dataframe()

    def xǁCompetitiveDiffusionModelǁrun_model__mutmut_1(self, n_steps):
        """Run the model for a specified number of steps."""
        for _ in range(None):
            self.step()
        return self.datacollector.get_model_vars_dataframe()
    
    xǁCompetitiveDiffusionModelǁrun_model__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁCompetitiveDiffusionModelǁrun_model__mutmut_1': xǁCompetitiveDiffusionModelǁrun_model__mutmut_1
    }
    xǁCompetitiveDiffusionModelǁrun_model__mutmut_orig.__name__ = 'xǁCompetitiveDiffusionModelǁrun_model'
