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


class SentimentHypeAgent(InnovationAgent):
    """An agent in a sentiment-driven hype cycle model.
    The agent's adoption decision is influenced by sentiment.
    """

    def __init__(self, unique_id, model):
        args = [unique_id, model]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSentimentHypeAgentǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁSentimentHypeAgentǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁSentimentHypeAgentǁ__init____mutmut_orig(self, unique_id, model):
        super().__init__(unique_id, model)
        self.sentiment = 0  # -1 for negative, 0 for neutral, 1 for positive

    def xǁSentimentHypeAgentǁ__init____mutmut_1(self, unique_id, model):
        super().__init__(None, model)
        self.sentiment = 0  # -1 for negative, 0 for neutral, 1 for positive

    def xǁSentimentHypeAgentǁ__init____mutmut_2(self, unique_id, model):
        super().__init__(unique_id, None)
        self.sentiment = 0  # -1 for negative, 0 for neutral, 1 for positive

    def xǁSentimentHypeAgentǁ__init____mutmut_3(self, unique_id, model):
        super().__init__(model)
        self.sentiment = 0  # -1 for negative, 0 for neutral, 1 for positive

    def xǁSentimentHypeAgentǁ__init____mutmut_4(self, unique_id, model):
        super().__init__(unique_id, )
        self.sentiment = 0  # -1 for negative, 0 for neutral, 1 for positive

    def xǁSentimentHypeAgentǁ__init____mutmut_5(self, unique_id, model):
        super().__init__(unique_id, model)
        self.sentiment = None  # -1 for negative, 0 for neutral, 1 for positive

    def xǁSentimentHypeAgentǁ__init____mutmut_6(self, unique_id, model):
        super().__init__(unique_id, model)
        self.sentiment = 1  # -1 for negative, 0 for neutral, 1 for positive
    
    xǁSentimentHypeAgentǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSentimentHypeAgentǁ__init____mutmut_1': xǁSentimentHypeAgentǁ__init____mutmut_1, 
        'xǁSentimentHypeAgentǁ__init____mutmut_2': xǁSentimentHypeAgentǁ__init____mutmut_2, 
        'xǁSentimentHypeAgentǁ__init____mutmut_3': xǁSentimentHypeAgentǁ__init____mutmut_3, 
        'xǁSentimentHypeAgentǁ__init____mutmut_4': xǁSentimentHypeAgentǁ__init____mutmut_4, 
        'xǁSentimentHypeAgentǁ__init____mutmut_5': xǁSentimentHypeAgentǁ__init____mutmut_5, 
        'xǁSentimentHypeAgentǁ__init____mutmut_6': xǁSentimentHypeAgentǁ__init____mutmut_6
    }
    xǁSentimentHypeAgentǁ__init____mutmut_orig.__name__ = 'xǁSentimentHypeAgentǁ__init__'

    def step(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSentimentHypeAgentǁstep__mutmut_orig'), object.__getattribute__(self, 'xǁSentimentHypeAgentǁstep__mutmut_mutants'), args, kwargs, self)

    def xǁSentimentHypeAgentǁstep__mutmut_orig(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = [n for n in neighbors if n.adopted]
        positive_sentiment_neighbors = [n for n in neighbors if n.sentiment > 0]

        if (
            len(adopting_neighbors) > self.model.adoption_threshold
            and len(positive_sentiment_neighbors) > self.model.sentiment_threshold
        ):
            self.adopted = True

    def xǁSentimentHypeAgentǁstep__mutmut_1(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = None
        if not neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = [n for n in neighbors if n.adopted]
        positive_sentiment_neighbors = [n for n in neighbors if n.sentiment > 0]

        if (
            len(adopting_neighbors) > self.model.adoption_threshold
            and len(positive_sentiment_neighbors) > self.model.sentiment_threshold
        ):
            self.adopted = True

    def xǁSentimentHypeAgentǁstep__mutmut_2(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = self.model.grid.get_neighbors(
            None,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = [n for n in neighbors if n.adopted]
        positive_sentiment_neighbors = [n for n in neighbors if n.sentiment > 0]

        if (
            len(adopting_neighbors) > self.model.adoption_threshold
            and len(positive_sentiment_neighbors) > self.model.sentiment_threshold
        ):
            self.adopted = True

    def xǁSentimentHypeAgentǁstep__mutmut_3(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=None,
            include_center=False,
        )
        if not neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = [n for n in neighbors if n.adopted]
        positive_sentiment_neighbors = [n for n in neighbors if n.sentiment > 0]

        if (
            len(adopting_neighbors) > self.model.adoption_threshold
            and len(positive_sentiment_neighbors) > self.model.sentiment_threshold
        ):
            self.adopted = True

    def xǁSentimentHypeAgentǁstep__mutmut_4(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=None,
        )
        if not neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = [n for n in neighbors if n.adopted]
        positive_sentiment_neighbors = [n for n in neighbors if n.sentiment > 0]

        if (
            len(adopting_neighbors) > self.model.adoption_threshold
            and len(positive_sentiment_neighbors) > self.model.sentiment_threshold
        ):
            self.adopted = True

    def xǁSentimentHypeAgentǁstep__mutmut_5(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = self.model.grid.get_neighbors(
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = [n for n in neighbors if n.adopted]
        positive_sentiment_neighbors = [n for n in neighbors if n.sentiment > 0]

        if (
            len(adopting_neighbors) > self.model.adoption_threshold
            and len(positive_sentiment_neighbors) > self.model.sentiment_threshold
        ):
            self.adopted = True

    def xǁSentimentHypeAgentǁstep__mutmut_6(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            include_center=False,
        )
        if not neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = [n for n in neighbors if n.adopted]
        positive_sentiment_neighbors = [n for n in neighbors if n.sentiment > 0]

        if (
            len(adopting_neighbors) > self.model.adoption_threshold
            and len(positive_sentiment_neighbors) > self.model.sentiment_threshold
        ):
            self.adopted = True

    def xǁSentimentHypeAgentǁstep__mutmut_7(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            )
        if not neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = [n for n in neighbors if n.adopted]
        positive_sentiment_neighbors = [n for n in neighbors if n.sentiment > 0]

        if (
            len(adopting_neighbors) > self.model.adoption_threshold
            and len(positive_sentiment_neighbors) > self.model.sentiment_threshold
        ):
            self.adopted = True

    def xǁSentimentHypeAgentǁstep__mutmut_8(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=False,
            include_center=False,
        )
        if not neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = [n for n in neighbors if n.adopted]
        positive_sentiment_neighbors = [n for n in neighbors if n.sentiment > 0]

        if (
            len(adopting_neighbors) > self.model.adoption_threshold
            and len(positive_sentiment_neighbors) > self.model.sentiment_threshold
        ):
            self.adopted = True

    def xǁSentimentHypeAgentǁstep__mutmut_9(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=True,
        )
        if not neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = [n for n in neighbors if n.adopted]
        positive_sentiment_neighbors = [n for n in neighbors if n.sentiment > 0]

        if (
            len(adopting_neighbors) > self.model.adoption_threshold
            and len(positive_sentiment_neighbors) > self.model.sentiment_threshold
        ):
            self.adopted = True

    def xǁSentimentHypeAgentǁstep__mutmut_10(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = [n for n in neighbors if n.adopted]
        positive_sentiment_neighbors = [n for n in neighbors if n.sentiment > 0]

        if (
            len(adopting_neighbors) > self.model.adoption_threshold
            and len(positive_sentiment_neighbors) > self.model.sentiment_threshold
        ):
            self.adopted = True

    def xǁSentimentHypeAgentǁstep__mutmut_11(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = None
        positive_sentiment_neighbors = [n for n in neighbors if n.sentiment > 0]

        if (
            len(adopting_neighbors) > self.model.adoption_threshold
            and len(positive_sentiment_neighbors) > self.model.sentiment_threshold
        ):
            self.adopted = True

    def xǁSentimentHypeAgentǁstep__mutmut_12(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = [n for n in neighbors if n.adopted]
        positive_sentiment_neighbors = None

        if (
            len(adopting_neighbors) > self.model.adoption_threshold
            and len(positive_sentiment_neighbors) > self.model.sentiment_threshold
        ):
            self.adopted = True

    def xǁSentimentHypeAgentǁstep__mutmut_13(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = [n for n in neighbors if n.adopted]
        positive_sentiment_neighbors = [n for n in neighbors if n.sentiment >= 0]

        if (
            len(adopting_neighbors) > self.model.adoption_threshold
            and len(positive_sentiment_neighbors) > self.model.sentiment_threshold
        ):
            self.adopted = True

    def xǁSentimentHypeAgentǁstep__mutmut_14(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = [n for n in neighbors if n.adopted]
        positive_sentiment_neighbors = [n for n in neighbors if n.sentiment > 1]

        if (
            len(adopting_neighbors) > self.model.adoption_threshold
            and len(positive_sentiment_neighbors) > self.model.sentiment_threshold
        ):
            self.adopted = True

    def xǁSentimentHypeAgentǁstep__mutmut_15(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = [n for n in neighbors if n.adopted]
        positive_sentiment_neighbors = [n for n in neighbors if n.sentiment > 0]

        if (
            len(adopting_neighbors) > self.model.adoption_threshold or len(positive_sentiment_neighbors) > self.model.sentiment_threshold
        ):
            self.adopted = True

    def xǁSentimentHypeAgentǁstep__mutmut_16(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = [n for n in neighbors if n.adopted]
        positive_sentiment_neighbors = [n for n in neighbors if n.sentiment > 0]

        if (
            len(adopting_neighbors) >= self.model.adoption_threshold
            and len(positive_sentiment_neighbors) > self.model.sentiment_threshold
        ):
            self.adopted = True

    def xǁSentimentHypeAgentǁstep__mutmut_17(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = [n for n in neighbors if n.adopted]
        positive_sentiment_neighbors = [n for n in neighbors if n.sentiment > 0]

        if (
            len(adopting_neighbors) > self.model.adoption_threshold
            and len(positive_sentiment_neighbors) >= self.model.sentiment_threshold
        ):
            self.adopted = True

    def xǁSentimentHypeAgentǁstep__mutmut_18(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = [n for n in neighbors if n.adopted]
        positive_sentiment_neighbors = [n for n in neighbors if n.sentiment > 0]

        if (
            len(adopting_neighbors) > self.model.adoption_threshold
            and len(positive_sentiment_neighbors) > self.model.sentiment_threshold
        ):
            self.adopted = None

    def xǁSentimentHypeAgentǁstep__mutmut_19(self):
        """The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions and sentiment.
        """
        if self.adopted:
            return

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )
        if not neighbors:
            return

        # Example logic: adopt if enough neighbors have adopted and sentiment is positive
        adopting_neighbors = [n for n in neighbors if n.adopted]
        positive_sentiment_neighbors = [n for n in neighbors if n.sentiment > 0]

        if (
            len(adopting_neighbors) > self.model.adoption_threshold
            and len(positive_sentiment_neighbors) > self.model.sentiment_threshold
        ):
            self.adopted = False
    
    xǁSentimentHypeAgentǁstep__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSentimentHypeAgentǁstep__mutmut_1': xǁSentimentHypeAgentǁstep__mutmut_1, 
        'xǁSentimentHypeAgentǁstep__mutmut_2': xǁSentimentHypeAgentǁstep__mutmut_2, 
        'xǁSentimentHypeAgentǁstep__mutmut_3': xǁSentimentHypeAgentǁstep__mutmut_3, 
        'xǁSentimentHypeAgentǁstep__mutmut_4': xǁSentimentHypeAgentǁstep__mutmut_4, 
        'xǁSentimentHypeAgentǁstep__mutmut_5': xǁSentimentHypeAgentǁstep__mutmut_5, 
        'xǁSentimentHypeAgentǁstep__mutmut_6': xǁSentimentHypeAgentǁstep__mutmut_6, 
        'xǁSentimentHypeAgentǁstep__mutmut_7': xǁSentimentHypeAgentǁstep__mutmut_7, 
        'xǁSentimentHypeAgentǁstep__mutmut_8': xǁSentimentHypeAgentǁstep__mutmut_8, 
        'xǁSentimentHypeAgentǁstep__mutmut_9': xǁSentimentHypeAgentǁstep__mutmut_9, 
        'xǁSentimentHypeAgentǁstep__mutmut_10': xǁSentimentHypeAgentǁstep__mutmut_10, 
        'xǁSentimentHypeAgentǁstep__mutmut_11': xǁSentimentHypeAgentǁstep__mutmut_11, 
        'xǁSentimentHypeAgentǁstep__mutmut_12': xǁSentimentHypeAgentǁstep__mutmut_12, 
        'xǁSentimentHypeAgentǁstep__mutmut_13': xǁSentimentHypeAgentǁstep__mutmut_13, 
        'xǁSentimentHypeAgentǁstep__mutmut_14': xǁSentimentHypeAgentǁstep__mutmut_14, 
        'xǁSentimentHypeAgentǁstep__mutmut_15': xǁSentimentHypeAgentǁstep__mutmut_15, 
        'xǁSentimentHypeAgentǁstep__mutmut_16': xǁSentimentHypeAgentǁstep__mutmut_16, 
        'xǁSentimentHypeAgentǁstep__mutmut_17': xǁSentimentHypeAgentǁstep__mutmut_17, 
        'xǁSentimentHypeAgentǁstep__mutmut_18': xǁSentimentHypeAgentǁstep__mutmut_18, 
        'xǁSentimentHypeAgentǁstep__mutmut_19': xǁSentimentHypeAgentǁstep__mutmut_19
    }
    xǁSentimentHypeAgentǁstep__mutmut_orig.__name__ = 'xǁSentimentHypeAgentǁstep'


class SentimentHypeModel(Model):
    """A model for a sentiment-driven hype cycle."""

    def __init__(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        args = [num_agents, width, height, adoption_threshold, sentiment_threshold]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSentimentHypeModelǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁSentimentHypeModelǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁSentimentHypeModelǁ__init____mutmut_orig(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_1(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = None
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_2(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = None
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_3(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = None
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_4(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = None
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_5(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(None, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_6(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, None, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_7(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, None)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_8(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_9(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_10(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, )
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_11(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, False)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_12(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = None

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_13(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = False

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_14(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(None):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_15(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = None
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_16(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=None, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_17(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=None)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_18(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_19(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_20(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = None
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_21(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(None)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_22(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = None
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_23(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(None)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_24(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(None, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_25(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, None)

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_26(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent((x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_27(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, )

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_28(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(None):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_29(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(6):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_30(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = None
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_31(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(None)
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_32(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(None))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_33(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = None
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_34(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = False
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_35(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = None

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_36(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 2

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_37(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = None

    def xǁSentimentHypeModelǁ__init____mutmut_38(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters=None,
        )

    def xǁSentimentHypeModelǁ__init____mutmut_39(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "XXAdoptersXX": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_40(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "adopters": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_41(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "ADOPTERS": lambda m: sum([1 for a in m.agents if a.adopted]),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_42(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: None,
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_43(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum(None),
            },
        )

    def xǁSentimentHypeModelǁ__init____mutmut_44(
        self,
        num_agents,
        width,
        height,
        adoption_threshold,
        sentiment_threshold,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.adoption_threshold = adoption_threshold
        self.sentiment_threshold = sentiment_threshold
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            agent = SentimentHypeAgent(unique_id=i, model=self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters and sentiment
        for _ in range(5):  # Seed 5 initial adopters
            agent = self.random.choice(list(self.agents))
            agent.adopted = True
            agent.sentiment = 1

        self.datacollector = DataCollector(
            model_reporters={
                "Adopters": lambda m: sum([2 for a in m.agents if a.adopted]),
            },
        )
    
    xǁSentimentHypeModelǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSentimentHypeModelǁ__init____mutmut_1': xǁSentimentHypeModelǁ__init____mutmut_1, 
        'xǁSentimentHypeModelǁ__init____mutmut_2': xǁSentimentHypeModelǁ__init____mutmut_2, 
        'xǁSentimentHypeModelǁ__init____mutmut_3': xǁSentimentHypeModelǁ__init____mutmut_3, 
        'xǁSentimentHypeModelǁ__init____mutmut_4': xǁSentimentHypeModelǁ__init____mutmut_4, 
        'xǁSentimentHypeModelǁ__init____mutmut_5': xǁSentimentHypeModelǁ__init____mutmut_5, 
        'xǁSentimentHypeModelǁ__init____mutmut_6': xǁSentimentHypeModelǁ__init____mutmut_6, 
        'xǁSentimentHypeModelǁ__init____mutmut_7': xǁSentimentHypeModelǁ__init____mutmut_7, 
        'xǁSentimentHypeModelǁ__init____mutmut_8': xǁSentimentHypeModelǁ__init____mutmut_8, 
        'xǁSentimentHypeModelǁ__init____mutmut_9': xǁSentimentHypeModelǁ__init____mutmut_9, 
        'xǁSentimentHypeModelǁ__init____mutmut_10': xǁSentimentHypeModelǁ__init____mutmut_10, 
        'xǁSentimentHypeModelǁ__init____mutmut_11': xǁSentimentHypeModelǁ__init____mutmut_11, 
        'xǁSentimentHypeModelǁ__init____mutmut_12': xǁSentimentHypeModelǁ__init____mutmut_12, 
        'xǁSentimentHypeModelǁ__init____mutmut_13': xǁSentimentHypeModelǁ__init____mutmut_13, 
        'xǁSentimentHypeModelǁ__init____mutmut_14': xǁSentimentHypeModelǁ__init____mutmut_14, 
        'xǁSentimentHypeModelǁ__init____mutmut_15': xǁSentimentHypeModelǁ__init____mutmut_15, 
        'xǁSentimentHypeModelǁ__init____mutmut_16': xǁSentimentHypeModelǁ__init____mutmut_16, 
        'xǁSentimentHypeModelǁ__init____mutmut_17': xǁSentimentHypeModelǁ__init____mutmut_17, 
        'xǁSentimentHypeModelǁ__init____mutmut_18': xǁSentimentHypeModelǁ__init____mutmut_18, 
        'xǁSentimentHypeModelǁ__init____mutmut_19': xǁSentimentHypeModelǁ__init____mutmut_19, 
        'xǁSentimentHypeModelǁ__init____mutmut_20': xǁSentimentHypeModelǁ__init____mutmut_20, 
        'xǁSentimentHypeModelǁ__init____mutmut_21': xǁSentimentHypeModelǁ__init____mutmut_21, 
        'xǁSentimentHypeModelǁ__init____mutmut_22': xǁSentimentHypeModelǁ__init____mutmut_22, 
        'xǁSentimentHypeModelǁ__init____mutmut_23': xǁSentimentHypeModelǁ__init____mutmut_23, 
        'xǁSentimentHypeModelǁ__init____mutmut_24': xǁSentimentHypeModelǁ__init____mutmut_24, 
        'xǁSentimentHypeModelǁ__init____mutmut_25': xǁSentimentHypeModelǁ__init____mutmut_25, 
        'xǁSentimentHypeModelǁ__init____mutmut_26': xǁSentimentHypeModelǁ__init____mutmut_26, 
        'xǁSentimentHypeModelǁ__init____mutmut_27': xǁSentimentHypeModelǁ__init____mutmut_27, 
        'xǁSentimentHypeModelǁ__init____mutmut_28': xǁSentimentHypeModelǁ__init____mutmut_28, 
        'xǁSentimentHypeModelǁ__init____mutmut_29': xǁSentimentHypeModelǁ__init____mutmut_29, 
        'xǁSentimentHypeModelǁ__init____mutmut_30': xǁSentimentHypeModelǁ__init____mutmut_30, 
        'xǁSentimentHypeModelǁ__init____mutmut_31': xǁSentimentHypeModelǁ__init____mutmut_31, 
        'xǁSentimentHypeModelǁ__init____mutmut_32': xǁSentimentHypeModelǁ__init____mutmut_32, 
        'xǁSentimentHypeModelǁ__init____mutmut_33': xǁSentimentHypeModelǁ__init____mutmut_33, 
        'xǁSentimentHypeModelǁ__init____mutmut_34': xǁSentimentHypeModelǁ__init____mutmut_34, 
        'xǁSentimentHypeModelǁ__init____mutmut_35': xǁSentimentHypeModelǁ__init____mutmut_35, 
        'xǁSentimentHypeModelǁ__init____mutmut_36': xǁSentimentHypeModelǁ__init____mutmut_36, 
        'xǁSentimentHypeModelǁ__init____mutmut_37': xǁSentimentHypeModelǁ__init____mutmut_37, 
        'xǁSentimentHypeModelǁ__init____mutmut_38': xǁSentimentHypeModelǁ__init____mutmut_38, 
        'xǁSentimentHypeModelǁ__init____mutmut_39': xǁSentimentHypeModelǁ__init____mutmut_39, 
        'xǁSentimentHypeModelǁ__init____mutmut_40': xǁSentimentHypeModelǁ__init____mutmut_40, 
        'xǁSentimentHypeModelǁ__init____mutmut_41': xǁSentimentHypeModelǁ__init____mutmut_41, 
        'xǁSentimentHypeModelǁ__init____mutmut_42': xǁSentimentHypeModelǁ__init____mutmut_42, 
        'xǁSentimentHypeModelǁ__init____mutmut_43': xǁSentimentHypeModelǁ__init____mutmut_43, 
        'xǁSentimentHypeModelǁ__init____mutmut_44': xǁSentimentHypeModelǁ__init____mutmut_44
    }
    xǁSentimentHypeModelǁ__init____mutmut_orig.__name__ = 'xǁSentimentHypeModelǁ__init__'

    def step(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSentimentHypeModelǁstep__mutmut_orig'), object.__getattribute__(self, 'xǁSentimentHypeModelǁstep__mutmut_mutants'), args, kwargs, self)

    def xǁSentimentHypeModelǁstep__mutmut_orig(self):
        """Run one step of the model."""
        self.datacollector.collect(self)
        self.agents.do("step")

    def xǁSentimentHypeModelǁstep__mutmut_1(self):
        """Run one step of the model."""
        self.datacollector.collect(None)
        self.agents.do("step")

    def xǁSentimentHypeModelǁstep__mutmut_2(self):
        """Run one step of the model."""
        self.datacollector.collect(self)
        self.agents.do(None)

    def xǁSentimentHypeModelǁstep__mutmut_3(self):
        """Run one step of the model."""
        self.datacollector.collect(self)
        self.agents.do("XXstepXX")

    def xǁSentimentHypeModelǁstep__mutmut_4(self):
        """Run one step of the model."""
        self.datacollector.collect(self)
        self.agents.do("STEP")
    
    xǁSentimentHypeModelǁstep__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSentimentHypeModelǁstep__mutmut_1': xǁSentimentHypeModelǁstep__mutmut_1, 
        'xǁSentimentHypeModelǁstep__mutmut_2': xǁSentimentHypeModelǁstep__mutmut_2, 
        'xǁSentimentHypeModelǁstep__mutmut_3': xǁSentimentHypeModelǁstep__mutmut_3, 
        'xǁSentimentHypeModelǁstep__mutmut_4': xǁSentimentHypeModelǁstep__mutmut_4
    }
    xǁSentimentHypeModelǁstep__mutmut_orig.__name__ = 'xǁSentimentHypeModelǁstep'

    def run_model(self, n_steps):
        args = [n_steps]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSentimentHypeModelǁrun_model__mutmut_orig'), object.__getattribute__(self, 'xǁSentimentHypeModelǁrun_model__mutmut_mutants'), args, kwargs, self)

    def xǁSentimentHypeModelǁrun_model__mutmut_orig(self, n_steps):
        """Run the model for a specified number of steps."""
        for _ in range(n_steps):
            self.step()
        return self.datacollector.get_model_vars_dataframe()

    def xǁSentimentHypeModelǁrun_model__mutmut_1(self, n_steps):
        """Run the model for a specified number of steps."""
        for _ in range(None):
            self.step()
        return self.datacollector.get_model_vars_dataframe()
    
    xǁSentimentHypeModelǁrun_model__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSentimentHypeModelǁrun_model__mutmut_1': xǁSentimentHypeModelǁrun_model__mutmut_1
    }
    xǁSentimentHypeModelǁrun_model__mutmut_orig.__name__ = 'xǁSentimentHypeModelǁrun_model'
