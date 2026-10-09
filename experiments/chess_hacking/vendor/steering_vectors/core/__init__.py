"""Facts, formats and small mechanisms that depend on nothing.

Every module here uses the standard library only, and no module here imports
anything outside it. That is what makes it safe for any process to import,
including one that has no numpy.

Nothing is re-exported: importing a name from this package must not drag in the
other modules.
"""
