import pluggy

hookspec = pluggy.HookspecMarker("innovate")
hookimpl = pluggy.HookimplMarker("innovate")

class InnovatePluginManager:
    """
    A simple plugin manager for the innovate library.
    """
    def __init__(self):
        self.pm = pluggy.PluginManager("innovate")
        self.pm.add_hookspecs(self)

    @hookspec
    def register_models(self):
        """
        A hook for plugins to register new models.
        """
        pass

def get_plugin_manager():
    """
    Returns a new instance of the plugin manager.
    """
    return InnovatePluginManager()
