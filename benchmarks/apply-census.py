#!/usr/bin/env python3
"""Apply (or revert) the TESS_CENSUS probe to the mlx checkout.

Attributes every GPU dispatch and every hazard barrier to the primitive that
issued it, and tracks the ASAP critical-path depth of the dispatch DAG (the
minimum wave count any schedule could reach). Probe only — never committed.

usage: apply-census.py [apply|revert]
"""
import os
import subprocess
import sys

MLX = os.path.expanduser(
    "~/Library/Developer/Xcode/DerivedData/tesseract-buwysfpnwmzyucelgewutuddcvgv"
    "/SourcePackages/checkouts/mlx-swift/Source/Cmlx/mlx"
)
DEV = f"{MLX}/mlx/backend/metal/device.cpp"
EVAL = f"{MLX}/mlx/backend/metal/eval.cpp"

PREAMBLE = '''// ---- TESS_CENSUS (probe, not for commit) --------------------------------
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx::core::gpu {
thread_local const char* tess_census_prim = "(none)";
}

namespace {
struct TessCensusRow {
  long dispatches = 0;
  long barriers = 0;
};
std::mutex tess_census_mu;
std::map<std::string, TessCensusRow> tess_census;
long tess_census_n = 0;
bool tess_census_on = std::getenv("TESS_CENSUS") != nullptr;

std::unordered_map<const void*, long> tess_level;
long tess_cur_in_level = 0;
long tess_max_level = 0;
std::vector<const void*> tess_pending_outputs;

void tess_census_dump() {
  const char* path = std::getenv("TESS_CENSUS_OUT");
  FILE* f = std::fopen(path ? path : "/tmp/barrier-census.txt", "w");
  if (!f) {
    return;
  }
  long td = 0, tb = 0;
  for (auto& [k, v] : tess_census) {
    td += v.dispatches;
    tb += v.barriers;
  }
  std::fprintf(
      f,
      "# dispatches=%ld barriers=%ld criticalPathDepth=%ld\\n",
      td,
      tb,
      tess_max_level);
  std::fprintf(f, "# primitive\\tdispatches\\tbarriers\\n");
  for (auto& [k, v] : tess_census) {
    std::fprintf(f, "%s\\t%ld\\t%ld\\n", k.c_str(), v.dispatches, v.barriers);
  }
  std::fclose(f);
}

void tess_census_note(bool barrier) {
  std::lock_guard<std::mutex> lk(tess_census_mu);
  auto& row = tess_census[mlx::core::gpu::tess_census_prim];
  row.dispatches++;
  if (barrier) {
    row.barriers++;
  }
  long lvl = tess_cur_in_level + 1;
  if (lvl > tess_max_level) {
    tess_max_level = lvl;
  }
  for (auto* p : tess_pending_outputs) {
    tess_level[p] = lvl;
  }
  tess_pending_outputs.clear();
  tess_cur_in_level = 0;
  if (++tess_census_n % 200000 == 0) {
    tess_census_dump();
  }
}

void tess_census_input(const void* p) {
  std::lock_guard<std::mutex> lk(tess_census_mu);
  auto it = tess_level.find(p);
  if (it != tess_level.end() && it->second > tess_cur_in_level) {
    tess_cur_in_level = it->second;
  }
}

void tess_census_output(const void* p) {
  std::lock_guard<std::mutex> lk(tess_census_mu);
  tess_pending_outputs.push_back(p);
}
} // namespace
// ---- end TESS_CENSUS ----------------------------------------------------

namespace mlx::core::metal {

namespace {'''

EDITS_DEV = [
    ("namespace mlx::core::metal {\n\nnamespace {", PREAMBLE),
    (
        "  auto r_buf = static_cast<MTL::Resource*>(const_cast<void*>(a.buffer().ptr()));\n"
        "  needs_barrier_ =",
        "  auto r_buf = static_cast<MTL::Resource*>(const_cast<void*>(a.buffer().ptr()));\n"
        "  if (tess_census_on) {\n    tess_census_input(a.buffer().ptr());\n  }\n"
        "  needs_barrier_ =",
    ),
    (
        "void CommandEncoder::register_output_array(const array& a) {\n",
        "void CommandEncoder::register_output_array(const array& a) {\n"
        "  if (tess_census_on) {\n    tess_census_output(a.buffer().ptr());\n  }\n",
    ),
    (
        "void CommandEncoder::maybeInsertBarrier() {\n  if (needs_barrier_) {",
        "void CommandEncoder::maybeInsertBarrier() {\n"
        "  if (tess_census_on) {\n    tess_census_note(needs_barrier_);\n  }\n"
        "  if (needs_barrier_) {",
    ),
]

EDITS_EVAL = [
    (
        "namespace mlx::core::gpu {",
        "namespace mlx::core::gpu {\n\nextern thread_local const char* tess_census_prim;",
    ),
    (
        "    debug_set_primitive_buffer_label(command_buffer, arr.primitive());\n"
        "    arr.primitive().eval_gpu(arr.inputs(), outputs);",
        "    debug_set_primitive_buffer_label(command_buffer, arr.primitive());\n"
        "    tess_census_prim = arr.primitive().name();\n"
        "    arr.primitive().eval_gpu(arr.inputs(), outputs);\n"
        '    tess_census_prim = "(none)";',
    ),
]


def apply():
    for path, edits in ((DEV, EDITS_DEV), (EVAL, EDITS_EVAL)):
        os.chmod(path, 0o644)
        s = open(path).read()
        for old, new in edits:
            assert s.count(old) == 1, f"anchor not unique in {path}: {old[:60]!r}"
            s = s.replace(old, new, 1)
        open(path, "w").write(s)
    print("census applied")


def revert():
    subprocess.run(
        ["git", "checkout", "--", "mlx/backend/metal/device.cpp", "mlx/backend/metal/eval.cpp"],
        cwd=MLX,
        check=True,
    )
    print("census reverted")


{"apply": apply, "revert": revert}[sys.argv[1] if len(sys.argv) > 1 else "apply"]()
