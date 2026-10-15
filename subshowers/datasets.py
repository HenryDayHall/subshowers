"""
The subshower dataset a config describes: which of the subshower sets in its
subshowers section have been made, and DatasetSources for those that have.

>>> checks = check_subshower_sets(config)
>>> sources = [check.source() for check in checks if check.ready]
>>> missing = [(check.label, check.problem) for check in checks if not check.ready]
"""

import os as _os
from dataclasses import dataclass as _dataclass

from subshowers import generation as _generation
from subshowers.sampling import DatasetSource as _DatasetSource


@_dataclass
class SetCheck:
    """Whether one subshower set the config asks for can be used."""

    subshower_set: _generation.SubshowerSet
    problem: str = None  # why the set can't be used; None if it can

    @property
    def ready(self):
        return self.problem is None

    @property
    def label(self):
        return (
            f"{self.subshower_set.shower.subfolder},"
            f" energy cut {self.subshower_set.energy_cut} GeV"
        )

    def source(self):
        """A DatasetSource reading the set's saved subshowers."""
        subshower_set = self.subshower_set
        return _DatasetSource(
            subshower_set.shower.folder,
            output_path=subshower_set.output,
            leaves_only=subshower_set.leaves_only,
            name=subshower_set.shower.subfolder,
        )


def check_subshower_sets(config, queued=None):
    """
    A SetCheck for each subshower set in the config's subshowers section,
    in the order the section lists them. queued is {output path: JOB_TASK}
    for jobs in the queue, as from generation.queued_outputs, so that sets
    still being made can say so.
    """
    queued = queued or {}
    return [
        SetCheck(subshower_set, _problem(subshower_set, queued))
        for subshower_set in _generation.wanted_subshower_sets(config)
    ]


def _problem(subshower_set, queued):
    """Why a subshower set can't be used yet, or None if it can."""
    shower = subshower_set.shower
    settings = _generation.saved_settings(subshower_set.output)
    if settings == (subshower_set.energy_cut, subshower_set.leaves_only):
        return None
    if settings is not None:
        return (
            f"made with other settings (energy cut {settings[0]},"
            f" leaves only {settings[1]})"
        )
    if subshower_set.output in queued:
        return f"queued as job {queued[subshower_set.output]}"
    status = _generation.shower_status(shower.folder)
    if shower.folder in queued and status != _generation.DONE:
        return f"the shower is queued as job {queued[shower.folder]}"
    if status == _generation.MISSING:
        return "the shower isn't made"
    if status != _generation.DONE:
        return f"the shower is unfinished, as {status}"
    if not _os.path.isfile(_generation.history_path(shower.folder)):
        return "the shower has no history/history.parquet"
    if _os.path.exists(subshower_set.output):
        return "unfinished: the .h5 is there, but its .yaml isn't complete"
    return "not made yet"
