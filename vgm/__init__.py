"""Shared library for the vision-guided manipulation pipeline.

Stage scripts under ``stages/`` are thin command-line wrappers around this package.
Modules that do not talk to CoppeliaSim (``geometry``, ``config``, ``metrics``,
``perception``, ``perturb``, ``servo``) are unit-tested without the simulator.
"""
