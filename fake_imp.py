import sys
import types
import importlib.util
import os

imp_mock = types.ModuleType('imp')

def find_module(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise ImportError("No module named " + name)
    return (None, os.path.dirname(spec.origin), None)

imp_mock.find_module = find_module
sys.modules['imp'] = imp_mock
