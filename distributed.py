"""Discrete voltage dispatch using station-local min-sum and traceback.

The boundary indices of each sparse transition tensor are cumulative signed
voltage ticks at the monitors and cumulative nonnegative resource usage.
An action adds a shift and incurs a local cost. A station combines histories
that reach the same boundary, retaining its cheapest predecessor locally.
The terminal adds the conditional expected voltage loss supplied by the caller.
"""
from dataclasses import dataclass
import json
import multiprocessing
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Station:
    bus: int
    actions: Sequence          # (P,Q) per action
    voltage_shifts: Sequence   # (action, monitor), in p.u.
    costs: Sequence            # compatible with the terminal loss units
    available: Sequence | None = None
    resources: Sequence | None = None  # (action, shared resource), in discrete units


@dataclass(frozen=True)
class DispatchResult:
    action_indices: tuple
    actions: tuple
    voltage_shift: tuple
    objective: float
    diagnostics: dict


def _encode(payload):
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _decode(data):
    payload = json.loads(data)
    if isinstance(payload, dict) and "error" in payload:
        raise ValueError(payload["error"])
    return payload


def _resource_units(values):
    data = np.asarray(values)
    if data.size and (data.dtype.kind not in "iu" or np.any(data < 0)):
        raise ValueError("shared resource units must be nonnegative integers")
    return data


class StationAgent:
    """Owns one station's transitions, minimum-cost predecessors and action."""

    def __init__(self, station, tick, limits, monitors):
        actions = np.asarray(station.actions, float)
        shifts = np.asarray(station.voltage_shifts, float)
        costs = np.asarray(station.costs, float)
        n = len(actions)
        if (not n or actions.shape != (n, 2) or shifts.shape != (n, monitors)
                or costs.shape != (n,) or not np.all(np.isfinite(actions))
                or not np.all(np.isfinite(shifts)) or not np.all(np.isfinite(costs))):
            raise ValueError("invalid action table at bus %s" % station.bus)
        available = np.ones(n, dtype=bool) if station.available is None else np.asarray(station.available)
        if available.shape != (n,) or available.dtype.kind != "b":
            raise ValueError("available must contain one boolean per action")
        resources = (np.zeros((n, len(limits)), dtype=int) if station.resources is None
                     else _resource_units(station.resources))
        if resources.shape != (n, len(limits)):
            raise ValueError("invalid resource table at bus %s" % station.bus)
        self.moves = []
        for i in np.flatnonzero(available):
            increments = tuple(map(int, np.rint(shifts[i] / tick))) + tuple(map(int, resources[i]))
            self.moves.append((int(i), increments, float(costs[i])))
        self.tick = tick
        self.monitors = monitors
        self.limits = limits
        self.parents = {}
        self.costs = {}

    def advance(self, message=None):
        """Combine this station with its predecessor's forward wire message."""
        origin = (0,) * (self.monitors + len(self.limits))
        previous = [(origin, 0.0)] if message is None else _decode(message)
        self.parents.clear()
        self.costs.clear()
        for boundary, cost in previous:
            for action, shift, local_cost in self.moves:
                state = tuple(x + dx for x, dx in zip(boundary, shift, strict=True))
                if any(used > cap for used, cap in zip(state[self.monitors:], self.limits)):
                    continue
                candidate = cost + local_cost
                if candidate < self.costs.get(state, float("inf")):
                    self.costs[state] = candidate
                    self.parents[state] = (tuple(boundary), action)
        if not self.costs:
            raise ValueError("no feasible sequence of available actions fits shared resource limits")

    def offer(self):
        """Only boundary values and their minimum costs leave the station."""
        return _encode(list(self.costs.items()))

    def finish(self, risk):
        # After enforcing the last budget limit, only the least-cost history
        # for a given voltage can win. Different budget usage needs no rescore.
        if self.limits:
            by_voltage = {}
            for state, cost in self.costs.items():
                voltage = state[:self.monitors]
                old = by_voltage.get(voltage)
                if old is None or cost < self.costs[old]:
                    by_voltage[voltage] = state
            states = list(by_voltage.values())
        else:
            states = list(self.costs)
        shifts = np.asarray([state[:self.monitors] for state in states], float) * self.tick
        losses = np.asarray(risk(shifts), float)
        if losses.shape != (len(states),) or np.any(np.isnan(losses)) or np.any(np.isneginf(losses)):
            raise ValueError("terminal risk must return one finite or +inf loss per boundary")
        scores = losses + np.fromiter((self.costs[state] for state in states), dtype=float)
        choice = int(np.argmin(scores))
        if not np.isfinite(scores[choice]):
            raise ValueError("no feasible terminal boundary")
        return states[choice], float(scores[choice])

    def traceback(self, state):
        return self.parents[tuple(state)]


def _local_sweeps(stations, risk, tick, caps, monitors):
    agents = []
    message = None
    sent = 0
    counts = []
    for i, station in enumerate(stations):
        agent = StationAgent(station, tick, caps, monitors)
        agent.advance(message)
        agents.append(agent)
        counts.append(len(agent.costs))
        if i + 1 < len(stations):
            message = agent.offer()
            sent += len(message)
    final, objective = agents[-1].finish(risk)
    state = final
    indices = []
    for i in range(len(agents)-1, -1, -1):
        state, index = agents[i].traceback(state)
        indices.append(index)
        if i:
            message = _encode(state)
            sent += len(message)
            state = _decode(message)
    return tuple(reversed(indices)), final, objective, counts, sent


def _receive(link, timeout):
    if not link.poll(timeout):
        raise TimeoutError("station timed out waiting for its neighbor")
    return link.recv_bytes()


def _worker(station, tick, caps, monitors, upstream, downstream, report, timeout):
    """One process, one private action table; both sweeps use direct neighbor pipes."""
    try:
        agent = StationAgent(station, tick, caps, monitors)
        agent.advance(None if upstream is None else _receive(upstream, timeout))
        offer = agent.offer()
        downstream.send_bytes(offer)
        state, index = agent.traceback(_decode(_receive(downstream, timeout)))
        sent = len(offer)
        if upstream is not None:
            response = _encode(state)
            upstream.send_bytes(response)
            sent += len(response)
        report.send((index, len(agent.costs), sent))
    except Exception as exc:
        downstream.send_bytes(_encode({"error": str(exc)}))
    finally:
        if upstream is not None:
            upstream.close()
        downstream.close()
        report.close()


def _process_sweeps(stations, risk, tick, caps, monitors, timeout):
    # The terminal stays in the caller so its risk model need not be pickled.
    ctx = multiprocessing.get_context("spawn")
    workers, reports = [], []
    incoming = None
    try:
        for station in stations[:-1]:
            outgoing, next_incoming = ctx.Pipe()
            receiver, sender = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_worker, args=(station, tick, caps, monitors, incoming,
                                                    outgoing, sender, timeout))
            try:
                proc.start()
            except BaseException:
                receiver.close()
                next_incoming.close()
                raise
            finally:
                outgoing.close()
                sender.close()
            workers.append(proc)
            reports.append(receiver)
            if incoming is not None:
                incoming.close()
            incoming = next_incoming
        terminal = StationAgent(stations[-1], tick, caps, monitors)
        terminal.advance(_receive(incoming, timeout) if incoming is not None else None)
        final, objective = terminal.finish(risk)
        previous, last_index = terminal.traceback(final)
        sent = 0
        if incoming is not None:
            response = _encode(previous)
            incoming.send_bytes(response)
            sent += len(response)
        # Supervisor reads the chosen actions solely to collect the simulation output.
        choices, counts = [], []
        for receiver in reports:
            if not receiver.poll(timeout):
                raise TimeoutError("station timed out reporting its selected action")
            index, size, payload = receiver.recv()
            choices.append(index)
            counts.append(size)
            sent += payload
        return tuple(choices + [last_index]), final, objective, counts + [len(terminal.costs)], sent
    finally:
        if incoming is not None:
            incoming.close()
        for receiver in reports:
            receiver.close()
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
            proc.join()
            proc.close()


def solve_dispatch(stations, risk, *, tick=0.0001, resource_limits=(), transport="local",
                   timeout=60.0):
    """Minimize sum(local costs) + expected voltage risk for one control period.

    The risk callback accepts an (n_boundaries, n_monitors) array of *quantized*
    voltage increments and returns one scalar per row. Local action increments
    use nearest ticks (ties to even). An optional resource ledger enforces exact
    integer budgets. Both transports use identical JSON neighbor messages. In
    process mode call from an importable module with a guarded main on Windows.
    """
    stations = tuple(stations)
    if not stations or len({s.bus for s in stations}) != len(stations):
        raise ValueError("at least one station with unique bus IDs is required")
    if not np.isfinite(tick) or tick <= 0 or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("tick and timeout must be finite and positive")
    limits = _resource_units(resource_limits)
    if limits.ndim != 1:
        raise ValueError("resource_limits must be a vector")
    caps = tuple(map(int, limits))
    first = np.asarray(stations[0].voltage_shifts)
    if first.ndim != 2 or first.shape[1] < 1:
        raise ValueError("expected a nonempty monitor dimension")
    monitors = first.shape[1]
    if transport == "local":
        result = _local_sweeps(stations, risk, tick, caps, monitors)
    elif transport == "process":
        result = _process_sweeps(stations, risk, tick, caps, monitors, timeout)
    else:
        raise ValueError("transport must be 'local' or 'process'")
    indices, final, objective, counts, wire_bytes = result
    actions = tuple(tuple(map(float, s.actions[i])) for s, i in zip(stations, indices))
    diagnostics = dict(messages=2*(len(stations)-1), forward_messages=len(stations)-1,
                       backward_messages=len(stations)-1, forward_entries=counts[:-1],
                       terminal_entries=counts[-1], wire_bytes=wire_bytes, tick=tick,
                       max_voltage_rounding_error=len(stations)*tick/2,
                       resources_used=final[monitors:], transport=transport)
    return DispatchResult(indices, actions, tuple(x*tick for x in final[:monitors]),
                          objective, diagnostics)
