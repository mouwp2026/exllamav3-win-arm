"""CPU-only regression tests; execute real budgeting functions, not torch imports.

EXL3_TEST_MEMORY_SOURCE selects the captured original to demonstrate the old cap.
The CUDA stub and fake procfs are scoped to the extracted module's globals.
"""
import ast
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

GIB = 1 << 30
HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("EXL3_TEST_MEMORY_SOURCE", HERE.parent / "exllamav3/util/memory.py"))


def load_budget_functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    names = {"touch_device", "set_memory_fraction_use", "set_memory_fraction_reserve",
             "uma_memory_headroom", "unset_memory_fraction", "device_mem_info", "is_integrated_device"}
    nodes = [node for node in tree.body if
             isinstance(node, ast.FunctionDef) and (node.name in names or node.name.startswith("_uma_"))
             or isinstance(node, (ast.Import, ast.ImportFrom)) and
             (node.names[0].name if isinstance(node, ast.Import) else node.module)
             in {"os", "sys", "re", "math", "pathlib"}]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_budget_functions()
        self.env = {"EXL3_UMA": "1"}
        self.props = SimpleNamespace(name="NVIDIA GB10", total_memory=128 * GIB,
                                     major=12, minor=1, integrated=True, is_integrated=False)
        self.current = 0
        self.free = 20 * GIB
        self.device_count = 1
        self.fractions = []
        self.touches = []
        self.reads = []
        self.cuda_reads = []
        self.logs = []
        self.ns["print"] = lambda message, **kwargs: self.logs.append(message)
        self.files = {
            "/proc/meminfo": f"MemTotal: {128 * GIB // 1024} kB\nMemAvailable: {118 * GIB // 1024} kB\nSwapFree: {512 * GIB // 1024} kB\n",
            "/proc/self/cgroup": "0::/\n",
            "/proc/self/mountinfo": "30 20 0:28 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        }
        self.cgroup("/sys/fs/cgroup", limit="max", current=10 * GIB)
        self.ns.update(
            os=SimpleNamespace(environ=self.env),
            sys=SimpleNamespace(platform="linux"),
            torch=SimpleNamespace(cuda=SimpleNamespace(
                get_device_properties=lambda device: self.props,
                device_count=lambda: self.device_count,
                memory_reserved=lambda device: self.current,
                mem_get_info=self.mem_get_info,
                set_per_process_memory_fraction=lambda fraction, device: self.fractions.append((fraction, device)),
            )),
            touch_device=lambda device: self.touches.append(device),
            host_memory_available=lambda: 118 * GIB,
            open=self.read,
        )

    def mem_get_info(self, device):
        self.cuda_reads.append(device)
        return self.free, self.props.total_memory

    def read(self, path, *args, **kwargs):
        path = str(path)
        self.reads.append(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        return io.StringIO(self.files[path])

    def cgroup(self, path, limit, current, inactive=0, dirty=0, writeback=0, shmem=0):
        self.files[path + "/memory.max"] = str(limit) + "\n"
        self.files[path + "/memory.current"] = str(current) + "\n"
        self.files[path + "/memory.stat"] = (
            f"file {inactive}\ninactive_file {inactive}\nfile_dirty {dirty}\n"
            f"file_writeback {writeback}\nshmem {shmem}\n"
        )

    def use(self, gib=106):
        return self.ns["set_memory_fraction_use"](gib * GIB, 0)

    def reserve(self, gib=8):
        return self.ns["set_memory_fraction_reserve"](gib * GIB, 0)

    def physical_check(self, transient=30 * GIB, allocated=0):
        path = Path(os.environ.get("EXL3_TEST_MODEL_SOURCE", HERE.parent / "exllamav3/model/model_ls.py"))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        block = None
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not isinstance(body, list):
                continue
            starts = [i for i, stmt in enumerate(body) if isinstance(stmt, ast.Assign)
                      and any(isinstance(n, ast.Name) and n.id == "free_now"
                              for target in stmt.targets for n in ast.walk(target))]
            if starts:
                start = starts[0]
                end = next(i for i in range(start, len(body)) if isinstance(body[i], ast.If)
                           and any(isinstance(n, ast.Name) and n.id == "reusable" for n in ast.walk(body[i].test)))
                block = body[start:end + 1]
                break
        self.assertIsNotNone(block, "could not locate real autosplit physical-headroom check")
        self.ns["torch"].cuda.memory_allocated = lambda device: allocated
        self.ns["torch"].cuda.OutOfMemoryError = MemoryError
        self.ns.update(load_device=0, i=0, max_transient={0: transient}, autosplit_margin=256 << 20)
        exec(compile(ast.Module(body=block, type_ignores=[]), str(path), "exec"), self.ns)
        return self.ns["reusable"]

    def test_default_use_and_reserve_remain_cuda_free_limited(self):
        self.env.clear()
        self.current = 4 * GIB
        self.props.name = "NVIDIA RTX 4090"
        self.assertEqual(self.use(), 24 * GIB)
        self.assertEqual(self.reserve(), 16 * GIB)
        self.assertEqual(self.reads, [])
        self.assertEqual(self.logs, [])

    def test_explicit_off_ignores_uma_reserve_and_telemetry(self):
        self.env.update(EXL3_UMA="0", EXL3_UMA_RESERVE_MB="nonsense")
        self.files.clear()
        self.assertEqual(self.use(), 20 * GIB)
        self.assertEqual(self.reserve(), 12 * GIB)
        self.ns["unset_memory_fraction"]([0])
        self.assertEqual(self.fractions[-1], (1.0, 0))

    def test_default_reserve_retains_legacy_one_percent_floor(self):
        self.env.clear()
        self.free = 0
        self.assertEqual(self.reserve(), int(0.01 * self.props.total_memory))

    def test_default_physical_check_still_uses_cuda_free(self):
        self.env.clear()
        with self.assertRaises(MemoryError):
            self.physical_check()
        self.assertEqual(self.cuda_reads, [0])
        self.assertEqual(self.reads, [])

    def test_second_component_counts_current_reservation_once(self):
        self.current = 10 * GIB
        self.assertEqual(self.use(), 116 * GIB)
        self.assertEqual(self.fractions[-1], (116 / 128, 0))

    def test_device_remaining_capacity_bounds_second_component(self):
        self.current = 100 * GIB
        self.assertEqual(self.use(), 120 * GIB)

    def test_requested_use_is_a_ceiling_not_a_floor(self):
        self.assertEqual(self.use(3), 3 * GIB)
        self.assertEqual(self.use(0), 0)

    def test_host_available_minus_os_reserve_bounds_large_request(self):
        self.assertEqual(self.use(1000), 110 * GIB)

    def test_low_cgroup_limit_wins(self):
        self.cgroup("/sys/fs/cgroup", limit=24 * GIB, current=10 * GIB)
        self.assertEqual(self.use(), 6 * GIB)
        self.assertEqual(self.reserve(), 6 * GIB)

    def test_only_clean_inactive_cgroup_file_cache_is_reclaimable(self):
        self.cgroup("/sys/fs/cgroup", limit=64 * GIB, current=60 * GIB,
                    inactive=50 * GIB, dirty=5 * GIB, writeback=3 * GIB, shmem=2 * GIB)
        self.assertEqual(self.use(), 36 * GIB)

    def test_dirty_writeback_and_shmem_never_create_negative_reclaim(self):
        self.cgroup("/sys/fs/cgroup", limit=32 * GIB, current=16 * GIB,
                    inactive=5 * GIB, dirty=3 * GIB, writeback=3 * GIB, shmem=3 * GIB)
        self.assertEqual(self.use(), 8 * GIB)

    def test_reclaim_is_bounded_by_file_bytes_and_current_charge(self):
        self.cgroup("/sys/fs/cgroup", limit=64 * GIB, current=60 * GIB, inactive=200 * GIB)
        self.assertEqual(self.use(), 56 * GIB)
        self.files["/sys/fs/cgroup/memory.stat"] += "active_file 999999999999\n"
        self.assertEqual(self.use(), 56 * GIB)

    def test_no_headroom_fails_without_one_percent_floor(self):
        self.cgroup("/sys/fs/cgroup", limit=8 * GIB, current=0)
        with self.assertRaisesRegex(RuntimeError, "no headroom"):
            self.reserve(0)
        self.assertEqual(self.fractions, [])

    def test_over_limit_cgroup_has_no_positive_budget_without_reclaim(self):
        self.cgroup("/sys/fs/cgroup", limit=8 * GIB, current=10 * GIB)
        with self.assertRaisesRegex(RuntimeError, "no headroom"):
            self.use()
        self.assertEqual(self.ns["uma_memory_headroom"](0), 0)
        self.assertEqual(self.fractions, [])

    def test_os_reserve_environment_is_in_mib(self):
        self.env["EXL3_UMA_RESERVE_MB"] = "16384"
        self.assertEqual(self.use(), 102 * GIB)

    def test_explicit_zero_os_reserve_is_respected(self):
        self.env["EXL3_UMA_RESERVE_MB"] = "0"
        self.assertEqual(self.use(1000), 118 * GIB)

    def test_invalid_os_reserve_fails_closed(self):
        for value in ("", "-1", "nan", "inf", "1.5", "8GiB", "+8", " 8192", "8e3"):
            with self.subTest(value=value):
                self.env["EXL3_UMA_RESERVE_MB"] = value
                with self.assertRaisesRegex(RuntimeError, "EXL3_UMA_RESERVE_MB"):
                    self.use()
        self.assertEqual(self.fractions, [])

    def test_invalid_opt_in_value_fails_closed(self):
        for value in ("", "true", "yes", "2", "1 "):
            with self.subTest(value=value):
                self.env["EXL3_UMA"] = value
                with self.assertRaisesRegex(RuntimeError, "EXL3_UMA"):
                    self.use()
        self.assertEqual(self.fractions, [])

    def test_invalid_request_bytes_fail_closed(self):
        for value in (-1, float("inf"), float("nan"), 1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "nonnegative integer"):
                    self.ns["set_memory_fraction_use"](value, 0)
        self.assertEqual(self.fractions, [])

    def test_non_gb10_is_rejected_not_given_host_ram(self):
        for name in ("NVIDIA RTX 4090", "NVIDIA H100", "NVIDIA GB100", "GB10-ish"):
            with self.subTest(name=name):
                self.props.name = name
                with self.assertRaisesRegex(RuntimeError, "GB10"):
                    self.use()
        self.assertEqual(self.fractions, [])

    def test_non_linux_multi_gpu_discrete_property_and_wrong_cc_rejected(self):
        for attr, value in (("platform", "win32"), ("count", 2), ("integrated", False), ("major", 9)):
            with self.subTest(attr=attr):
                self.ns["sys"].platform = value if attr == "platform" else "linux"
                self.device_count = value if attr == "count" else 1
                self.props.integrated = value if attr == "integrated" else True
                self.props.major = value if attr == "major" else 12
                with self.assertRaisesRegex(RuntimeError, "GB10"):
                    self.use()
        self.assertEqual(self.fractions, [])

    def test_known_gb10_without_torch_integrated_attribute_is_supported(self):
        del self.props.integrated
        self.assertEqual(self.use(), 106 * GIB)

    def test_missing_required_telemetry_fails_closed_even_if_unlimited(self):
        for path in tuple(self.files):
            with self.subTest(path=path):
                saved = self.files.pop(path)
                try:
                    with self.assertRaisesRegex(RuntimeError, "telemetry unavailable"):
                        self.use()
                finally:
                    self.files[path] = saved
        self.assertEqual(self.fractions, [])

    def test_invalid_cgroup_numbers_fail_closed(self):
        for field in ("memory.current", "memory.max"):
            for value in ("-1", "nan", "", "1.2"):
                path = "/sys/fs/cgroup/" + field
                with self.subTest(path=path, value=value):
                    saved = self.files[path]
                    self.files[path] = value
                    try:
                        with self.assertRaisesRegex(RuntimeError, "invalid"):
                            self.use()
                    finally:
                        self.files[path] = saved
        self.assertEqual(self.fractions, [])

    def test_missing_dirty_writeback_shmem_or_inactive_stats_fail_closed(self):
        path = "/sys/fs/cgroup/memory.stat"
        saved = self.files[path]
        for field in ("file", "inactive_file", "file_dirty", "file_writeback", "shmem"):
            with self.subTest(field=field):
                self.files[path] = "\n".join(line for line in saved.splitlines() if not line.startswith(field + " "))
                with self.assertRaisesRegex(RuntimeError, "incomplete memory.stat"):
                    self.use()
        self.assertEqual(self.fractions, [])

    def test_malformed_or_duplicate_stat_counters_fail_closed(self):
        path = "/sys/fs/cgroup/memory.stat"
        saved = self.files[path]
        for suffix in ("shmem 0\n", "other -1\n", "broken\n", "other 1 extra\n"):
            with self.subTest(suffix=suffix):
                self.files[path] = saved + suffix
                with self.assertRaisesRegex(RuntimeError, "invalid"):
                    self.use()
        self.assertEqual(self.fractions, [])

    def test_invalid_or_missing_memavailable_never_uses_swap(self):
        for content in ("MemTotal: 128 kB\nSwapFree: 999999999 kB\n",
                        "MemTotal: 128 kB\nMemAvailable: -1 kB\n",
                        "MemTotal: 128 kB\nMemAvailable: 129 kB\n",
                        "MemTotal: 128 kB\nMemAvailable: 120 MB\n",
                        "MemTotal: 0 kB\nMemAvailable: 0 kB\n"):
            with self.subTest(content=content):
                self.files["/proc/meminfo"] = content
                with self.assertRaises(RuntimeError):
                    self.use()
        self.assertEqual(self.fractions, [])

    def test_invalid_cuda_telemetry_fails_closed(self):
        for total, current in ((0, 0), (128 * GIB, -1), (128 * GIB, 129 * GIB),
                               (float("inf"), 0), (128 * GIB, float("nan"))):
            with self.subTest(total=total, current=current):
                self.props.total_memory, self.current = total, current
                with self.assertRaisesRegex(RuntimeError, "CUDA total/reserved"):
                    self.use()
        self.assertEqual(self.fractions, [])

    def test_nonroot_mount_and_escaped_mountpoint_are_resolved(self):
        self.files["/proc/self/cgroup"] = "0::/tenant/job\n"
        self.files["/proc/self/mountinfo"] = "30 20 0:28 /tenant /cg\\040space rw - cgroup2 cgroup rw\n"
        self.cgroup("/cg space", limit=32 * GIB, current=16 * GIB)
        self.cgroup("/cg space/job", limit="max", current=4 * GIB)
        self.assertEqual(self.use(), 8 * GIB)

    def test_broadest_mount_does_not_hide_a_stricter_parent(self):
        self.files["/proc/self/cgroup"] = "0::/tenant/job\n"
        self.files["/proc/self/mountinfo"] = (
            "31 20 0:28 /tenant/job /bind rw - cgroup2 cgroup rw\n" + self.files["/proc/self/mountinfo"])
        self.cgroup("/bind", limit="max", current=4 * GIB)
        self.cgroup("/sys/fs/cgroup/tenant", limit=32 * GIB, current=16 * GIB)
        self.cgroup("/sys/fs/cgroup/tenant/job", limit="max", current=4 * GIB)
        self.assertEqual(self.use(), 8 * GIB)

    def test_missing_visible_parent_telemetry_fails_closed(self):
        self.files["/proc/self/cgroup"] = "0::/tenant/job\n"
        self.cgroup("/sys/fs/cgroup/tenant/job", limit="max", current=4 * GIB)
        with self.assertRaisesRegex(RuntimeError, "telemetry unavailable"):
            self.use()
        self.assertEqual(self.fractions, [])

    def test_unsupported_cgroup_membership_or_mount_fails_closed(self):
        for content in ("2:memory:/\n", "0::/../outside\n", "0::relative\n", "0::/\n0::/again\n"):
            with self.subTest(content=content):
                self.files["/proc/self/cgroup"] = content
                with self.assertRaises(RuntimeError):
                    self.use()
        self.files["/proc/self/cgroup"] = "0::/other\n"
        self.files["/proc/self/mountinfo"] = "30 20 0:28 /tenant /cg rw - cgroup2 cgroup rw\n"
        with self.assertRaisesRegex(RuntimeError, "cannot map"):
            self.use()
        self.assertEqual(self.fractions, [])

    def test_physical_helper_refreshes_telemetry(self):
        self.assertEqual(self.ns["uma_memory_headroom"](0), 110 * GIB)
        self.cgroup("/sys/fs/cgroup", limit=24 * GIB, current=10 * GIB)
        self.assertEqual(self.ns["uma_memory_headroom"](0), 6 * GIB)
        self.assertEqual(self.fractions, [])

    def test_physical_check_still_rejects_low_cgroup_headroom(self):
        self.cgroup("/sys/fs/cgroup", limit=24 * GIB, current=10 * GIB)
        with self.assertRaises(MemoryError):
            self.physical_check()

    def test_physical_check_counts_only_reusable_allocator_cache(self):
        self.current = 12 * GIB
        self.cgroup("/sys/fs/cgroup", limit=24 * GIB, current=10 * GIB)
        self.assertEqual(self.physical_check(transient=7 * GIB, allocated=10 * GIB), 8 * GIB)

    def test_unset_fails_closed_if_telemetry_is_lost_after_load(self):
        del self.files["/sys/fs/cgroup/memory.max"]
        with self.assertRaises(RuntimeError):
            self.ns["unset_memory_fraction"]([0])
        self.assertEqual(self.fractions, [])

    def test_uma_does_not_replace_global_mem_get_info(self):
        original = self.ns["torch"].cuda.mem_get_info
        self.use()
        self.reserve()
        self.assertIs(self.ns["torch"].cuda.mem_get_info, original)
        self.assertEqual(self.cuda_reads, [])

    def test_budget_never_exceeds_request_or_any_safety_bound(self):
        import random
        rng = random.Random(2026)
        for _ in range(100):
            total = rng.randrange(32 * GIB, 128 * GIB)
            current = rng.randrange(total)
            available = rng.randrange(128 * GIB)
            limit = rng.randrange(16 * GIB, 128 * GIB)
            charged = rng.randrange(limit + 1)
            inactive = rng.randrange(charged + 1)
            dirty = rng.randrange(inactive + 1)
            requested = rng.randrange(200 * GIB)
            self.props.total_memory, self.current = total, current
            self.files["/proc/meminfo"] = f"MemTotal: {128 * GIB // 1024} kB\nMemAvailable: {available // 1024} kB\n"
            self.cgroup("/sys/fs/cgroup", limit=limit, current=charged, inactive=inactive, dirty=dirty)
            headroom = min((available // 1024) * 1024, total - current,
                           limit - charged + inactive - dirty) - 8 * GIB
            with self.subTest(total=total, current=current, headroom=headroom):
                if headroom <= 0:
                    with self.assertRaisesRegex(RuntimeError, "no headroom"):
                        self.ns["set_memory_fraction_use"](requested, 0)
                else:
                    result = self.ns["set_memory_fraction_use"](requested, 0)
                    expected = current + min(requested, headroom)
                    self.assertLessEqual(result, expected)
                    self.assertLessEqual(expected - result, 1)
                    self.assertGreaterEqual(result, 0)
                    self.assertLessEqual(result, total)

    def test_cap_logs_exact_byte_budget_and_telemetry(self):
        result = self.use()
        text = "\n".join(self.logs)
        self.assertIn(f"cap_bytes={result}", text)
        self.assertIn(f"host_available={118 * GIB}", text)
        self.assertIn(f"os_reserve={8 * GIB}", text)
        self.assertIn("cgroup_headroom=unlimited", text)
        self.assertIn(f"device_headroom={128 * GIB}", text)

    def test_unset_preserves_uma_os_reserve_instead_of_resetting_one(self):
        self.current = 40 * GIB
        self.cgroup("/sys/fs/cgroup", limit=64 * GIB, current=48 * GIB)
        self.ns["unset_memory_fraction"]([0])
        self.assertEqual(self.fractions, [(48 / 128, 0)])

    def test_autosplit_physical_check_accepts_reclaimable_uma(self):
        self.current = 4 * GIB
        self.assertEqual(self.physical_check(allocated=2 * GIB), 112 * GIB)
        self.assertEqual(self.cuda_reads, [])

    def test_nested_cgroup_parent_limit_wins(self):
        self.files["/proc/self/cgroup"] = "0::/tenant/job\n"
        self.cgroup("/sys/fs/cgroup/tenant", limit=32 * GIB, current=16 * GIB)
        self.cgroup("/sys/fs/cgroup/tenant/job", limit="max", current=4 * GIB)
        self.assertEqual(self.use(), 8 * GIB)
        self.assertIn("/sys/fs/cgroup/tenant/job/memory.stat", self.reads)

    def test_uma_reserve_uses_the_larger_of_os_and_caller_reserves(self):
        self.current = 4 * GIB
        self.assertEqual(self.reserve(12), (4 + 118 - 12) * GIB)
        self.assertEqual(self.reserve(2), (4 + 118 - 8) * GIB)

    def test_uma_use_reclaims_host_cache_instead_of_freezing_cuda_free(self):
        self.assertEqual(self.use(), 106 * GIB)
        self.assertEqual(self.fractions, [(106 / 128, 0)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
