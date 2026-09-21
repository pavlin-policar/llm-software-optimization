"""Benchmark graphlet counters over predefined input graphs.

The default run compares these three commands on each input graph:

	bruteforce.exe graph.in graphlets-bf
	optimized.exe graph.in graphlets-opt
	orca.exe node 5 graph.in graphlets-orca.ndump2

Results are appended to a JSONL file as they finish. If the file already
contains a successful record for a given tool/graph pair, that run is skipped
on the next invocation so interrupted benchmark runs can resume.
"""

import argparse
import datetime
import json
import os
import subprocess
import time
from os.path import abspath, dirname, exists, join

HERE = dirname(abspath(__file__))
ROOT = dirname(HERE)
GRAPHLETS = join(ROOT, "graphlets")

DEFAULT_GRAPHS = [
	("graph", join(GRAPHLETS, "graph.in")),
	("graph_10k_40k", join(GRAPHLETS, "data", "graph_10k_40k.in")),
	("graph_10k_50k", join(GRAPHLETS, "data", "graph_10k_50k.in")),
	("graph_10k_60k", join(GRAPHLETS, "data", "graph_10k_60k.in")),
	("graph_10k_80k", join(GRAPHLETS, "data", "graph_10k_80k.in")),
	("graph_10k_100k", join(GRAPHLETS, "data", "graph_10k_100k.in")),
	("graph_10k_120k", join(GRAPHLETS, "data", "graph_10k_120k.in")),
	("graph_10k_140k", join(GRAPHLETS, "data", "graph_10k_140k.in")),
	("graph_10k_200k", join(GRAPHLETS, "data", "graph_10k_200k.in")),
	("graph_10k_300k", join(GRAPHLETS, "data", "graph_10k_300k.in")),
	("graph_10k_400k", join(GRAPHLETS, "data", "graph_10k_400k.in")),
	("graph_10k_500k", join(GRAPHLETS, "data", "graph_10k_500k.in")),
	("graph_10k_600k", join(GRAPHLETS, "data", "graph_10k_600k.in")),
	("graph_10k_700k", join(GRAPHLETS, "data", "graph_10k_700k.in")),
	("graph_10k_800k", join(GRAPHLETS, "data", "graph_10k_800k.in")),
	("human", join(GRAPHLETS, "data", "human.in")),
]

DEFAULT_EXECUTABLES = [
	("bruteforce", [join(GRAPHLETS, "bruteforce.exe"), "{graph}", "{output}"]),
	("optimized", [join(GRAPHLETS, "optimized.exe"), "{graph}", "{output}"]),
	("orca", [join(GRAPHLETS, "orca.exe"), "node", "5", "{graph}", "{output}.ndump2"]),
]

DEFAULT_COMPILE_COMMANDS = [
	"g++ -O2 -o bruteforce.exe bruteforce.cpp",
	"g++ -O2 -o orca.exe orca.cpp",
	"g++ -O2 -o optimized.exe optimized.cpp",
]


def log(msg):
	print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def resolve_executables(selected):
	available = {name: template for name, template in DEFAULT_EXECUTABLES}

	if selected is None:
		return [(name, abspath(template[0]), list(template)) for name, template in DEFAULT_EXECUTABLES]

	commands = []
	for item in selected:
		if ":" in item:
			name, path = item.split(":", 1)
			name = name.strip()
			path = path.strip()
			if not name or not path:
				raise SystemExit(f"invalid executable override: {item}")
			if name not in available:
				raise SystemExit(f"unknown executable name: {name}")
			template = list(available[name])
			template[0] = abspath(path)
			commands.append((name, template[0], template))
			continue

		name = item.strip()
		if name not in available:
			raise SystemExit(f"unknown executable name: {name}")
		template = list(available[name])
		template[0] = abspath(template[0])
		commands.append((name, template[0], template))

	return commands


def resolve_graphs(selected):
	available = {name: path for name, path in DEFAULT_GRAPHS}
	if selected is None:
		return list(DEFAULT_GRAPHS)

	graphs = []
	for item in selected:
		if ":" in item:
			name, path = item.split(":", 1)
			graphs.append((name.strip(), abspath(path.strip())))
			continue
		if item not in available:
			raise SystemExit(f"unknown graph name: {item}")
		graphs.append((item, available[item]))
	return graphs


def load_results(path):
	results = {}
	if exists(path):
		with open(path, encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				record = json.loads(line)
				results[(record["tool"], record["graph"])] = record
	return results


def run_child(cmd, cwd, log_path):
	start = time.perf_counter()
	with open(log_path, "w", encoding="utf-8") as logf:
		proc = subprocess.run(cmd, cwd=cwd, stdout=logf, stderr=logf, text=True)
	return proc.returncode, time.perf_counter() - start


def cmd_compile(args):
	os.makedirs(args.log_dir, exist_ok=True)
	commands = args.commands or list(DEFAULT_COMPILE_COMMANDS)

	for i, command in enumerate(commands, start=1):
		log_name = f"compile_{i}.log"
		log_path = join(args.log_dir, log_name)
		log(f"compile {i}/{len(commands)}: start")
		log(f"cmd={command}")
		start = time.perf_counter()
		with open(log_path, "w", encoding="utf-8") as logf:
			proc = subprocess.run(command, cwd=args.cwd, stdout=logf, stderr=logf, text=True, shell=True)
		wall_s = time.perf_counter() - start
		if proc.returncode != 0:
			log(f"compile {i}/{len(commands)}: failed in {wall_s:.1f} s (see {log_path})")
			raise SystemExit(proc.returncode)
		log(f"compile {i}/{len(commands)}: ok in {wall_s:.1f} s")

	log("compile finished")


def cmd_run(args):
	commands = resolve_executables(args.executables)
	graphs = resolve_graphs(args.graphs)

	for _, path, _ in commands:
		if not exists(path):
			raise SystemExit(f"missing executable: {path}")
	for _, path in graphs:
		if not exists(path):
			raise SystemExit(f"missing graph input: {path}")

	os.makedirs(dirname(abspath(args.out)), exist_ok=True)
	os.makedirs(args.log_dir, exist_ok=True)
	os.makedirs(args.work_dir, exist_ok=True)

	done = load_results(args.out)

	for graph_name, graph_path in graphs:
		for tool_name, exe_path, template in commands:
			key = (tool_name, graph_name)
			if key in done and done[key].get("status") == "ok":
				continue

			output_base = join(args.work_dir, f"{tool_name}_{graph_name}")
			cmd = [part.format(graph=graph_path, output=output_base) for part in template]
			log_name = f"{tool_name}_{graph_name}.log"

			log(f"{tool_name} on {graph_name}: start")
			log(f"cmd={cmd}")
			log(f"cwd={args.cwd}")
			returncode, wall_s = run_child(cmd, cwd=args.cwd, log_path=join(args.log_dir, log_name))

			record = {
				"tool": tool_name,
				"graph": graph_name,
				"graph_path": graph_path,
				"executable": exe_path,
				"command": cmd,
				"status": "ok" if returncode == 0 else "failed",
				"returncode": returncode,
				"wall_s": wall_s,
				"finished": datetime.datetime.now().isoformat(timespec="seconds"),
			}
			with open(args.out, "a", encoding="utf-8") as f:
				f.write(json.dumps(record) + "\n")
			log(f"{tool_name} on {graph_name}: {record['status']} in {wall_s:.1f} s")

	log("benchmark finished")


def main():
	ap = argparse.ArgumentParser(description="Run graphlet-counting benchmarks.")
	sub = ap.add_subparsers(dest="command", required=True)

	run = sub.add_parser("run", help="run the benchmark")
	run.add_argument(
		"--executables",
		nargs="*",
		default=None,
		help="optional subset of predefined executable names, or NAME:PATH overrides",
	)
	run.add_argument(
		"--graphs",
		nargs="*",
		default=None,
		help="optional subset of predefined graph names, or NAME:PATH overrides",
	)
	run.add_argument("--out", default=join(HERE, "results-graphlets.jsonl"))
	run.add_argument("--log-dir", default=join(HERE, "tmp", "logs"))
	run.add_argument("--work-dir", default=join(HERE, "tmp"))
	run.add_argument("--cwd", default=GRAPHLETS, help="working directory used when launching executables")

	compile_parser = sub.add_parser("compile", help="compile graphlet executables")
	compile_parser.add_argument(
		"--commands",
		nargs="*",
		default=None,
		help="optional list of full shell compile commands",
	)
	compile_parser.add_argument("--log-dir", default=join(HERE, "tmp", "logs"))
	compile_parser.add_argument("--cwd", default=GRAPHLETS, help="working directory used for compile commands")

	args = ap.parse_args()
	if args.command == "run":
		cmd_run(args)
	elif args.command == "compile":
		cmd_compile(args)


if __name__ == "__main__":
	main()
