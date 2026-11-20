"""Distributed feasibility of a finite archive of previously qualified schedules."""
import multiprocessing

import numpy as np

from distributed import _receive


def _local_menu(station, prescriptions):
    # An archive ID has one pre-positioned action at this station. Only the
    # intersection of admissible IDs crosses an edge during forward agreement.
    menu = {}
    for i, action in enumerate(station.actions):
        if station.available is None or station.available[i]:
            menu.setdefault(tuple(action), i)
    indices = [menu.get(tuple(action)) for action in prescriptions]
    mask = sum(1 << k for k, index in enumerate(indices) if index is not None)
    return indices, mask


def _winner(scores, mask):
    valid = [i for i, score in enumerate(scores) if mask & (1 << i) and np.isfinite(score)]
    if not valid:
        raise ValueError("no qualified and locally available archived policy")
    return min(valid, key=lambda i: scores[i])


def _archive_worker(station, prescriptions, upstream, downstream, report, width, timeout):
    try:
        indices, mask = _local_menu(station, prescriptions)
        if upstream is not None:
            mask &= int.from_bytes(_receive(upstream, timeout), 'little')
        downstream.send_bytes(mask.to_bytes(width, 'little'))
        chosen = _receive(downstream, timeout)
        if upstream is not None:
            upstream.send_bytes(chosen)
        report.send(indices[int.from_bytes(chosen, 'little')])
    finally:
        if upstream is not None:
            upstream.close()
        downstream.close()
        report.close()


def _process(stations, columns, scores, width, id_width, timeout):
    ctx = multiprocessing.get_context("spawn")
    incoming = None
    workers, receipts = [], []
    try:
        for station, prescriptions in zip(stations[:-1], columns[:-1], strict=True):
            outgoing, next_incoming = ctx.Pipe()
            receiver, sender = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_archive_worker,
                               args=(station, prescriptions, incoming, outgoing, sender, width, timeout))
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
            receipts.append(receiver)
            if incoming is not None:
                incoming.close()
            incoming = next_incoming
        indices, mask = _local_menu(stations[-1], columns[-1])
        if incoming is not None:
            mask &= int.from_bytes(_receive(incoming, timeout), 'little')
        chosen = _winner(scores, mask)
        if incoming is not None:
            incoming.send_bytes(chosen.to_bytes(id_width, 'little'))
        local_indices = []
        for receipt in receipts:
            if not receipt.poll(timeout):
                raise TimeoutError("no archived action acknowledgement from station")
            local_indices.append(receipt.recv())
        return chosen, tuple(local_indices + [indices[chosen]])
    finally:
        if incoming is not None:
            incoming.close()
        for receipt in receipts:
            receipt.close()
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
            proc.join()
            proc.close()


def agree_archive(stations, schedules, scores, *, transport, timeout=60.):
    """Select the lowest terminal cost admitted by every station, then return its ID.

    Archive construction enforces the common resource budget. Each station checks
    its *current* local capability. The terminal is the only holder of scores.
    Ties follow input order. Payload counts exclude process supervision.
    """
    stations, schedules = tuple(stations), tuple(schedules)
    scores = np.asarray(scores, float)
    if (not stations or not schedules or scores.shape != (len(schedules),)
            or np.isnan(scores).any() or np.isneginf(scores).any()
            or any(len(s) != len(stations) for s in schedules)):
        raise ValueError("expected one terminal score and one prescription per station for each policy")
    columns = tuple(zip(*schedules, strict=True))
    width = (len(schedules) + 7)//8
    id_width = max(1, ((len(schedules)-1).bit_length() + 7)//8)
    if transport == "process":
        chosen, indices = _process(stations, columns, scores, width, id_width, timeout)
    elif transport == "local":
        message = None
        station_indices = []
        for station, prescriptions in zip(stations, columns, strict=True):
            indices, mask = _local_menu(station, prescriptions)
            if message is not None:
                mask &= int.from_bytes(message, 'little')
            message = mask.to_bytes(width, 'little')
            station_indices.append(indices)
        chosen = _winner(scores, mask)
        reply = chosen.to_bytes(id_width, 'little')
        indices = tuple(menu[int.from_bytes(reply, 'little')] for menu in station_indices)
    else:
        raise ValueError("transport must be 'local' or 'process'")
    edges = len(stations)-1
    byte_count = edges*(width+id_width)
    return chosen, indices, dict(messages=2*edges, forward_messages=edges, backward_messages=edges,
                                 wire_bytes=byte_count, raw_wire_bytes=byte_count,
                                 archive_mask_bytes=width, policy_id_bytes=id_width)
