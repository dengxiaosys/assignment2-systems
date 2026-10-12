"""Apply reproducible, separately selectable sm_61 FFMA scheduling ablations.

Yield/reuse policies only change their control fields. Reordering additionally
moves mutually independent register FFMA words within dependency-free segments.
Stalls, dependency barriers, branches and ELF metadata always remain unchanged.
"""

import argparse
import hashlib
import json
import re
import struct
import subprocess
import tempfile
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def text_sections(cubin):
    if cubin[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected a little-endian ELF64 cubin")
    if struct.unpack_from("<I", cubin, 48)[0] & 255 != 61:
        raise ValueError("Only sm_61 cubins are supported")
    offset = struct.unpack_from("<Q", cubin, 40)[0]
    size, count, strings_index = struct.unpack_from("<HHH", cubin, 58)
    sections = [struct.unpack_from("<IIQQQQIIQQ", cubin, offset + i * size) for i in range(count)]
    strings = sections[strings_index]
    names = cubin[strings[4]:strings[4] + strings[5]]
    for section in sections:
        name = names[section[0]:].split(b"\0", 1)[0].decode()
        if (name.startswith(".text._ZN11maxwell_d647forward")
                or name.startswith(".text._ZN10fixed_wide7forward")):
            yield name[6:], section[4], section[5]


def apply_yield_policy(cubin, disassembly, interval):
    output = bytearray(cubin)
    records = []
    for symbol, offset, size in text_sections(cubin):
        marker = "Function : " + symbol
        if marker not in disassembly:
            raise ValueError(f"Missing disassembly for {symbol}")
        assembly = disassembly.split(marker, 1)[1].split("Function :", 1)[0]
        count = changed = 0
        for line in assembly.splitlines():
            match = re.match(r"\s*/\*([0-9a-f]+)\*/\s+(FFMA\s+.*?)\s+;"
                             r"\s*/\* (0x[0-9a-f]+) \*/", line)
            if match is None:
                continue
            address = int(match[1], 16)
            if address >= size or address % 32 == 0:
                raise ValueError("Invalid FFMA instruction address")
            instruction = struct.unpack_from("<Q", cubin, offset + address)[0]
            if instruction != int(match[3], 16):
                raise ValueError("Disassembly does not match cubin instruction")
            control_offset = offset + (address // 32) * 32
            slot = address % 32 // 8 - 1
            bit = 1 << (slot * 21 + 4)
            control = struct.unpack_from("<Q", output, control_offset)[0]
            count += 1
            # Sparse yields distribute ILP work across ready warps; interval=0
            # removes all FFMA yield hints for the corresponding ablation.
            updated = control & ~bit if interval and count % interval == 0 else control | bit
            changed += updated != control
            struct.pack_into("<Q", output, control_offset, updated)
        records.append({"symbol": symbol, "ffma_instructions": count, "changed_yield_bits": changed})
    if len(records) != 2 or any(record["ffma_instructions"] < 500 for record in records):
        raise ValueError("Expected both d64 causal templates and their complete FFMA bodies")
    return bytes(output), records


def apply_reuse_policy(cubin, disassembly):
    output = bytearray(cubin)
    records = []
    for symbol, offset, _ in text_sections(cubin):
        assembly = disassembly.split("Function : " + symbol, 1)[1].split("Function :", 1)[0]
        instructions = []
        for line in assembly.splitlines():
            match = re.match(r"\s*/\*([0-9a-f]+)\*/\s+(.*?)\s+;"
                             r"\s*/\* (0x[0-9a-f]+) \*/", line)
            if match:
                address = int(match[1], 16)
                instruction = struct.unpack_from("<Q", cubin, offset + address)[0]
                if instruction != int(match[3], 16):
                    raise ValueError("Disassembly does not match cubin instruction")
                instructions.append((address, match[2]))
        # Match only register FFMA instructions. Conservative basic blocks end
        # at every other instruction, so no memory write or branch can cross
        # an operand-cache lifetime.
        cache = [dict(), dict(), dict()]
        changed = count = 0
        for address, text in instructions:
            match = re.fullmatch(
                r"FFMA R(\d+), R(\d+)(?:\.reuse)?, R(\d+)(?:\.reuse)?, R(\d+)(?:\.reuse)?",
                text,
            )
            if match is None:
                cache = [dict(), dict(), dict()]
                continue
            count += 1
            registers = list(map(int, match.groups()))
            destination, sources = registers[0], registers[1:]
            control_offset = offset + address // 32 * 32
            slot = address % 32 // 8 - 1
            control = struct.unpack_from("<Q", output, control_offset)[0]
            cleared = control & ~(7 << (slot * 21 + 17))
            struct.pack_into("<Q", output, control_offset, cleared)
            for history in cache:
                history.pop(destination, None)
            for source_slot, register in enumerate(sources):
                if register == destination:
                    continue
                history = cache[source_slot]
                if register in history:
                    previous = history[register]
                    previous_offset = offset + previous // 32 * 32
                    previous_slot = previous % 32 // 8 - 1
                    previous_control = struct.unpack_from("<Q", output, previous_offset)[0]
                    previous_control |= 1 << (previous_slot * 21 + 17 + source_slot)
                    struct.pack_into("<Q", output, previous_offset, previous_control)
                elif len(history) == 2:
                    history.pop(min(history, key=history.get))
                history[register] = address
        for address, text in instructions:
            if text.startswith("FFMA"):
                control_offset = offset + address // 32 * 32
                slot = address % 32 // 8 - 1
                old = struct.unpack_from("<Q", cubin, control_offset)[0] >> (slot * 21 + 17) & 7
                new = struct.unpack_from("<Q", output, control_offset)[0] >> (slot * 21 + 17) & 7
                changed += old != new
        records.append({"symbol": symbol, "register_ffma_instructions": count,
                        "changed_reuse_fields": changed})
    return bytes(output), records


def register_ffma(text):
    match = re.fullmatch(
        r"FFMA R(\d+), R(\d+)(?:\.reuse)?, R(\d+)(?:\.reuse)?, R(\d+)(?:\.reuse)?",
        text,
    )
    return tuple(map(int, match.groups())) if match else None


def bank_step(registers, cache):
    """Community four-bank/two-entries-per-source-slot performance model."""
    destination, *sources = registers
    updated = [[value for value in history if value != destination] for history in cache]
    reads = []
    for slot, register in enumerate(sources):
        history = updated[slot]
        if register not in history:
            reads.append(register % 4)
        if register in history:
            history.remove(register)
        history.append(register)
        del history[:-2]
    conflicts = len(reads) - len(set(reads))
    return conflicts, len(reads), updated


def order_cost(order):
    cache = [[], [], []]
    conflicts = reads = 0
    for instruction in order:
        extra, loaded, cache = bank_step(instruction[3], cache)
        conflicts += extra
        reads += loaded
    return conflicts, reads


def independent_order(segment):
    best = segment
    best_cost = order_cost(best)
    # Seeding each possible first instruction avoids a random search and makes
    # the result deterministic for a given ptxas output.
    for seed in range(len(segment)):
        remaining = list(segment)
        first = remaining.pop(seed)
        order = [first]
        _, _, cache = bank_step(first[3], [[], [], []])
        while remaining:
            scored = []
            for index, instruction in enumerate(remaining):
                conflicts, reads, next_cache = bank_step(instruction[3], cache)
                following = [bank_step(other[3], next_cache)[:2]
                             for j, other in enumerate(remaining) if j != index]
                lookahead = min(following, default=(0, 0))
                scored.append(((conflicts, lookahead[0], reads, lookahead[1], index),
                               index, next_cache))
            _, index, cache = min(scored, key=lambda item: item[0])
            order.append(remaining.pop(index))
        cost = order_cost(order)
        if cost < best_cost:
            best, best_cost = order, cost
    return best, order_cost(segment), best_cost


def apply_reorder_policy(cubin, disassembly):
    output = bytearray(cubin)
    records = []
    for symbol, offset, size in text_sections(cubin):
        assembly = disassembly.split("Function : " + symbol, 1)[1].split("Function :", 1)[0]
        targets = {int(value, 16) for value in re.findall(
            r"\b(?:BRA|SSY|PBK|PCNT|CAL)\s+(0x[0-9a-f]+)", assembly)}
        instructions = []
        for line in assembly.splitlines():
            match = re.match(r"\s*/\*([0-9a-f]+)\*/\s+(.*?)\s+;"
                             r"\s*/\* (0x[0-9a-f]+) \*/", line)
            if match:
                address, word = int(match[1], 16), int(match[3], 16)
                if address >= size or address % 32 == 0:
                    raise ValueError("Invalid instruction address")
                if struct.unpack_from("<Q", cubin, offset + address)[0] != word:
                    raise ValueError("Disassembly does not match cubin instruction")
                instructions.append((address, word, match[2], register_ffma(match[2])))
        segment = []
        changed = segments = before_conflicts = after_conflicts = 0

        def flush():
            nonlocal segment, changed, segments, before_conflicts, after_conflicts
            # Preserve a six-instruction suffix so a moved FFMA still finishes
            # before any use beyond the segment (Pascal FP32 latency: 6 cycles).
            movable = segment[:-6]
            if len(movable) >= 3:
                order, before, after = independent_order(movable)
                for destination, source in zip(movable, order):
                    struct.pack_into("<Q", output, offset + destination[0], source[1])
                    changed += destination[1] != source[1]
                segments += 1
                before_conflicts += before[0]
                after_conflicts += after[0]
            segment = []

        for instruction in instructions:
            address, _, _, registers = instruction
            control = struct.unpack_from("<Q", cubin, offset + address // 32 * 32)[0]
            slot_control = control >> ((address % 32 // 8 - 1) * 21) & ((1 << 21) - 1)
            # Only fixed one-cycle, no-wait/no-barrier FFMA can move. A target
            # may refer to the preceding control word, so check both addresses.
            safe = (registers is not None and slot_control & 15 == 1
                    and slot_control >> 5 & 63 == 63 and slot_control >> 11 & 63 == 0)
            if address in targets or address - 8 in targets:
                flush()
            if not safe:
                flush()
                continue
            destination, *sources = registers
            if any(destination == other[3][0] or destination in other[3][1:]
                   or other[3][0] in sources for other in segment):
                flush()
            segment.append(instruction)
        flush()
        records.append({"symbol": symbol, "independent_segments": segments,
                        "moved_instruction_words": changed,
                        "modeled_bank_conflicts_before": before_conflicts,
                        "modeled_bank_conflicts_after": after_conflicts})
    return bytes(output), records


def apply_dual_issue_policy(cubin, disassembly):
    """Pair independent FP32 and LSU instructions without moving either."""
    output = bytearray(cubin)
    records = []
    for symbol, offset, size in text_sections(cubin):
        assembly = disassembly.split("Function : " + symbol, 1)[1].split("Function :", 1)[0]
        instructions = []
        for match in re.finditer(
            r"/\*([0-9a-f]+)\*/\s+(.*?)\s+;\s*/\* (0x[0-9a-f]+) \*/", assembly,
        ):
            address, word = int(match[1], 16), int(match[3], 16)
            if address >= size or struct.unpack_from("<Q", cubin, offset + address)[0] != word:
                raise ValueError("Disassembly does not match cubin")
            instructions.append((address, match[2]))
        changed = []
        for index, ((address, text), (next_address, following)) in enumerate(
            zip(instructions, instructions[1:]),
        ):
            registers = register_ffma(text)
            if registers is None:
                continue
            load = re.fullmatch(
                r"(?:LDS\.U\.128|LDG\.E\.CI\.128) R(\d+), \[R(\d+)(?:\+0x[0-9a-f]+)?\]",
                following,
            )
            if load is None:
                continue
            destination, base = map(int, load.groups())
            if any(register in range(destination, destination + 4) for register in registers):
                continue
            if registers[0] == base:
                continue
            control_offset = offset + address // 32 * 32
            slot = (address % 32 // 8 - 1) * 21
            control = struct.unpack_from("<Q", output, control_offset)[0]
            field = control >> slot & ((1 << 21) - 1)
            # Only no-wait/no-scoreboard one-cycle FFMA can dual issue.
            if field & 15 != 1 or field >> 5 & 63 != 63 or field >> 11 & 63:
                continue
            next_control = struct.unpack_from("<Q", cubin, offset + next_address // 32 * 32)[0]
            next_field = next_control >> ((next_address % 32 // 8 - 1) * 21)
            if next_field >> 11 & 63:
                continue
            # Preserve the documented six-cycle FP32 latency after shortening
            # the first gap by one. Stop conservatively at control flow.
            cycles = 0
            safe_latency = False
            for later_address, later_text in instructions[index + 1:]:
                if cycles >= 6:
                    safe_latency = True
                    break
                if re.search(r"\b(?:BRA|SSY|EXIT|CAL|RET)\b", later_text):
                    break
                if registers[0] in map(int, re.findall(r"\bR(\d+)", later_text)):
                    break
                later_control = struct.unpack_from(
                    "<Q", output, offset + later_address // 32 * 32,
                )[0]
                cycles += later_control >> ((later_address % 32 // 8 - 1) * 21) & 15
            if not safe_latency:
                continue
            struct.pack_into("<Q", output, control_offset, control & ~(15 << slot))
            changed.append({"address": address, "ffma": text, "lsu": following})
        records.append({"symbol": symbol, "dual_issue_pairs": changed})
    if len(records) != 2:
        raise ValueError("Expected two specialized causal templates")
    return bytes(output), records


def minimum_assignment(cost):
    """Hungarian minimum assignment, O(n^3) time and O(n^2) storage."""
    n = len(cost)
    u, v, p, way = [0] * (n + 1), [0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minimum, used = [float("inf")] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), 0
            for j in range(1, n + 1):
                if not used[j]:
                    current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < minimum[j]:
                        minimum[j], way[j] = current, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]], v[j] = u[p[j]] + delta, v[j] - delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if not p[j0]:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if not j0:
                break
    return {p[j] - 1: j - 1 for j in range(1, n + 1)}


def arithmetic_fields(text, word):
    """Decode only scalar arithmetic fields independently checked against SASS."""
    bare = re.sub(r"^@!?P\d+\s+", "", text)
    registers = [int(x) for x in re.findall(r"\bR(\d+)", bare)]
    op = bare.split()[0]
    if op == "FFMA":
        positions = [0, 8, 20, 39] if word >> 48 == 0x5980 else [0, 8, 39]
    elif op in ("RRO.EX2",):
        positions = [0, 20]
    elif op in ("MUFU.EX2",):
        positions = [0, 8]
    elif op in ("FADD", "FMUL"):
        positions = [0, 8, 20]
    elif op == "MOV":
        positions = [0, 20] if word >> 48 == 0x5c98 else [0]
    else:
        return None
    # RZ appears in a field but is intentionally excluded from remapping.
    operands = re.findall(r"\bR(?:\d+|Z)", bare)
    if len(operands) != len(positions):
        raise ValueError(f"Unsupported arithmetic form: {text}")
    for operand, position in zip(operands, positions):
        expected = 255 if operand == "RZ" else int(operand[1:])
        if word >> position & 255 != expected:
            raise ValueError(f"Register encoding mismatch: {text}")
    return positions


def apply_bank_assignment(cubin, disassembly):
    output = bytearray(cubin)
    records = []
    for symbol, offset, size in text_sections(cubin):
        assembly = disassembly.split("Function : " + symbol, 1)[1].split("Function :", 1)[0]
        instructions, restricted, all_registers = [], set(), set()
        for match in re.finditer(
            r"/\*([0-9a-f]+)\*/\s+(.*?)\s+;\s*/\* (0x[0-9a-f]+) \*/", assembly,
        ):
            address, text, word = int(match[1], 16), match[2], int(match[3], 16)
            if address >= size or struct.unpack_from("<Q", cubin, offset + address)[0] != word:
                raise ValueError("Disassembly does not match cubin")
            registers = set(map(int, re.findall(r"\bR(\d+)", text)))
            all_registers |= registers
            fields = arithmetic_fields(text, word)
            if fields is None:
                # Preserve complete quads around all non-arithmetic operands,
                # including implicit .128 destinations and 64-bit address pairs.
                for register in registers:
                    restricted.update(range(register // 4 * 4, register // 4 * 4 + 4))
            instructions.append((address, text, word, fields))
        free = sorted(all_registers - restricted)
        weights = {register: [0] * 4 for register in free}
        cache = [[], [], []]
        for address, text, word, fields in instructions:
            registers = register_ffma(text)
            if registers is None:
                cache = [[], [], []]
                continue
            destination, *sources = registers
            for history in cache:
                if destination in history:
                    history.remove(destination)
            uncached = [(slot, source) for slot, source in enumerate(sources)
                        if source not in cache[slot]]
            candidates = {source for _, source in uncached if source in weights}
            if len(candidates) == 1:
                candidate = next(iter(candidates))
                fixed_banks = [source % 4 for _, source in uncached if source != candidate]
                for bank in range(4):
                    weights[candidate][bank] += fixed_banks.count(bank)
            control = struct.unpack_from("<Q", cubin, offset + address // 32 * 32)[0]
            flags = control >> ((address % 32 // 8 - 1) * 21 + 17) & 7
            for slot, source in enumerate(sources):
                if flags >> slot & 1:
                    if source in cache[slot]:
                        cache[slot].remove(source)
                    cache[slot].append(source)
                    del cache[slot][:-2]
        assignment = minimum_assignment([
            [weights[source][target % 4] * 100 + (source != target) for target in free]
            for source in free
        ])
        mapping = {source: free[assignment[i]] for i, source in enumerate(free)}
        changed = 0
        for address, text, word, fields in instructions:
            if fields is None:
                continue
            updated = word
            for position in fields:
                register = word >> position & 255
                if register in mapping:
                    updated = (updated & ~(255 << position)) | (mapping[register] << position)
            if updated != word:
                struct.pack_into("<Q", output, offset + address, updated)
                changed += 1
        records.append({
            "symbol": symbol, "scalar_register_permutation": mapping,
            "changed_instructions": changed,
            "modeled_collisions_before": sum(weights[r][r % 4] for r in free),
            "modeled_collisions_after": sum(weights[r][mapping[r] % 4] for r in free),
        })
    return bytes(output), records


def patch_extension(extension, interval, report_path, reuse=False, reorder=False,
                    dual_issue=False, bank_assignment=False):
    original = extension.read_bytes()
    with tempfile.TemporaryDirectory(prefix="fa2-v5-schedule-") as temporary:
        directory = Path(temporary)
        subprocess.run(
            ["/usr/local/cuda/bin/cuobjdump", "--extract-elf", "all", str(extension.resolve())],
            cwd=directory, check=True, capture_output=True,
        )
        cubins = list(directory.glob("*.sm_61.cubin"))
        if len(cubins) != 1:
            raise ValueError("Expected exactly one sm_61 cubin")
        cubin = cubins[0].read_bytes()
        assembly = subprocess.check_output(
            ["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(cubins[0])], text=True,
        )
        if bank_assignment:
            updated, records = apply_bank_assignment(cubin, assembly)
        elif dual_issue:
            updated, records = apply_dual_issue_policy(cubin, assembly)
        elif reorder:
            updated, records = apply_reorder_policy(cubin, assembly)
            cubins[0].write_bytes(updated)
            assembly = subprocess.check_output(
                ["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(cubins[0])], text=True,
            )
            updated, reuse_records = apply_reuse_policy(updated, assembly)
            for record, reuse_record in zip(records, reuse_records):
                record.update({key: value for key, value in reuse_record.items() if key != "symbol"})
        else:
            updated, records = (apply_reuse_policy(cubin, assembly) if reuse
                                else apply_yield_policy(cubin, assembly, interval))
        position = original.find(cubin)
        if position < 0 or original.find(cubin, position + 1) >= 0:
            raise ValueError("Expected one uncompressed cubin in extension")
        result = original[:position] + updated + original[position + len(cubin):]
        extension.write_bytes(result)
    report = {
        "policy": ("scalar_register_bank_assignment" if bank_assignment else
                   "ffma_lsu_dual_issue" if dual_issue else
                   "ffma_independent_reorder" if reorder else
                   "ffma_local_reuse" if reuse else "ffma_yield_interval"), "interval": interval,
        "original_extension_sha256": sha256(original), "extension_sha256": sha256(result),
        "original_cubin_sha256": sha256(cubin), "cubin_sha256": sha256(updated),
        "changed_bytes": sum(a != b for a, b in zip(cubin, updated)),
        "unchanged_instruction_words": not (reorder or bank_assignment),
        "unchanged_stalls_and_barriers": not dual_issue,
        "unchanged_dependency_barriers": True,
        "kernels": records,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension", type=Path)
    parser.add_argument("--yield-interval", type=int, default=32)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--reorder", action="store_true")
    parser.add_argument("--dual-issue", action="store_true")
    parser.add_argument("--bank-assignment", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.yield_interval < 0:
        parser.error("--yield-interval must be nonnegative")
    patch_extension(args.extension, args.yield_interval, args.output_json,
                    args.reuse, args.reorder, args.dual_issue, args.bank_assignment)
