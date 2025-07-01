import yaml

class Config:
    def __init__(self, path: str):
        with open(path, 'r') as f:
            self._cfg = yaml.safe_load(f)

    def get(self, selection: str, default = None):
        return self._cfg.get(selection, default)
    
    def __getitem__(self, key):
        return self._cfg[key]
    
    def __repr__(self):
        return f"Config({self._cfg!r})"