from .scheduler import Scheduler

# import all subclasses of Scheduler automatically from each file in the scheduler directory
from pathlib import Path
import os
import importlib

# get the path to the scheduler directory
scheduler_dir = Path(__file__).parent
# get the list of files in the scheduler directory
scheduler_files = os.listdir(scheduler_dir)
# get the list of python files in the scheduler directory
scheduler_files = [file for file in scheduler_files if file.endswith('.py') and file != '__init__.py']
# import each python file in the scheduler directory
for file in scheduler_files:
    module_name = f'.{file[:-3]}'
    module = importlib.import_module(module_name, 'pmpd.modules.scheduler')
    # get the subclass of Scheduler in the module
    for obj in module.__dict__.values():
        if isinstance(obj, type) and issubclass(obj, Scheduler) and obj is not Scheduler:
            scheduler_subclass = obj
            # import the subclass module
            print(f'Registering {scheduler_subclass.__name__} from {module_name}')
            exec(f'from .{file[:-3]} import {scheduler_subclass.__name__}')