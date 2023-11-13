import torch
import warnings
import sqlite3

from moe import MoENetwork
from fff import FF
from fff import FFF
from fff import FFF_Sparse

import time

################################################################
# ARGUMENT PARSING											   #
################################################################

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--job-id", type=int)
parser.add_argument("--dataset", type=str)
parser.add_argument("--model", type=str, default='ff')
parser.add_argument("--input-width", type=int)
parser.add_argument("--hidden-width", type=int, default=3072)
parser.add_argument("--output-width", type=int)
parser.add_argument("--depth", type=int, default=11)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--n-iters", type=int, default=10000)
parser.add_argument("--results-db-file", type=str, default='results.db')
parser.add_argument("--device", type=str, default='cuda')

args = parser.parse_args()
cpu = args.device == 'cpu'

################################################################
# PREABMLE									   				   #
################################################################

def generate_data():
	global cpu
	if cpu:
		return (
			torch.randn(args.batch_size, args.input_width).to(torch.float32),
			torch.randint(args.output_width, (args.batch_size,))
		)
	else:
		return (
			torch.randn(args.batch_size, args.input_width).to(torch.float32).cuda(),
			torch.randint(args.output_width, (args.batch_size,)).cuda(),
		)

def init_model():
	global cpu
	
	# MOE
	if args.model == 'moe':
		model = MoENetwork(args.input_width, args.hidden_width, args.output_width, 2**args.depth, 1)
	
	# FF
	elif args.model == 'ff_native':
		model = torch.nn.Sequential(
			torch.nn.Linear(args.input_width, args.hidden_width, bias=False),
			torch.nn.GELU(),
			torch.nn.Linear(args.hidden_width, args.output_width, bias=False)
		)
	elif args.model == 'ff_bmm':
		model = FF(args.input_width, args.hidden_width, args.output_width)
	elif args.model == 'ff_sparse':
		raise NotImplementedError("Sparse FF is not implemented yet.")
	
	# FFF
	elif args.model == 'fff_native':
		raise NotImplementedError("Native FFF cannot be implemented yet -- could YOU be the hero of the PyTorch folk?")
	elif args.model == 'fff_bmm':
		model = FFF(args.input_width, args.depth, args.output_width)
	elif args.model == 'fff_sparse':
		model = FFF_Sparse(args.input_width, args.output_width, args.depth)
	
	if not cpu:
		return model.to(torch.float32).cuda()
	else:
		return model.to(torch.float32)

# A QUICK EAGER FOR DEBUGGING
model = init_model()

# EAGER
input_data = generate_data()[0]
model(input_data)

if not cpu:
	gpu_ok = False
	if torch.cuda.is_available():
		device_cap = torch.cuda.get_device_capability()
		if device_cap in ((7, 0), (8, 0), (9, 0)):
			gpu_ok = True
	else:
		raise RuntimeError("CUDA is not available.")

	if not gpu_ok:
		warnings.warn(
			"GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
			"than expected."
		)


################################################################
# INFERENCE BENCHMARK										   #
################################################################
# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
	if not cpu:
		start = torch.cuda.Event(enable_timing=True)
		end = torch.cuda.Event(enable_timing=True)
		start.record()
		result = fn()
		end.record()
		torch.cuda.synchronize()
		return result, start.elapsed_time(end) / 1000
	else:
		# measure process time between start and end
		start = time.process_time()
		result = fn()
		end = time.process_time()
		return result, end - start

def evaluate(mod, inp):
	with torch.no_grad():
		return mod(inp)


# LETS GO
model = init_model()

# EAGER
input_data = generate_data()[0]
model(input_data)

# COMPILED
import torch._dynamo # Reset since we are using a different mode.
torch._dynamo.reset()
evaluate_compiled = torch.compile(evaluate, mode="reduce-overhead") # otherwise "max-autotune"
print("eager:", timed(lambda: evaluate(model, input_data))[1])
evaluate_compiled = evaluate
print("compile:", timed(lambda: evaluate_compiled(model, input_data))[1])

compile_times = []
for i in range(args.n_iters):
	inp = generate_data()[0]
	_, compile_time = timed(lambda: evaluate_compiled(model, inp))
	compile_times.append(compile_time)

import numpy as np
compile_mean = np.mean(compile_times)
compile_stddev = np.std(compile_times)
print(f"(eval) compiled: {compile_mean} ± {compile_stddev}")
print("~" * 10)

################################################################
# SQL LOGGING												   #
################################################################

connection = sqlite3.connect(args.results_db_file)
cursor = connection.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS times (
	job_id INTEGER PRIMARY KEY,
	dataset TEXT,
	model TEXT,
	width INTEGER,
	depth INTEGER,
	batch_size INTEGER,
	n_iters INTEGER,
	inference_eager_mean REAL,
	inference_eager_stddev REAL,
	inference_compiled_mean REAL,
	inference_compiled_stddev REAL,
	training_eager_mean REAL,
	training_eager_stddev REAL,
	training_compiled_mean REAL,
	training_compiled_stddev REAL
)''')

results_dict = {
	"job_id": args.job_id,
	"dataset": args.dataset,
	"model": args.model,
	"width": args.hidden_width,
	"depth": args.depth,
	"batch_size": args.batch_size,
	"n_iters": args.n_iters,
	"inference_eager_mean": -1,
	"inference_eager_stddev": -1,
	"inference_compiled_mean": compile_mean,
	"inference_compiled_stddev": compile_stddev,
	"training_eager_mean": -1,
	"training_eager_stddev": -1,
	"training_compiled_mean": -1,
	"training_compiled_stddev": -1
}
cursor.executemany('''INSERT INTO times (
	job_id, dataset, model,
	width, depth, batch_size, n_iters,
	inference_eager_mean, inference_eager_stddev, inference_compiled_mean, inference_compiled_stddev,
	training_eager_mean, training_eager_stddev, training_compiled_mean, training_compiled_stddev
) VALUES (
	:job_id, :dataset, :model,
	:width, :depth, :batch_size, :n_iters,
	:inference_eager_mean, :inference_eager_stddev, :inference_compiled_mean, :inference_compiled_stddev,
	:training_eager_mean, :training_eager_stddev, :training_compiled_mean, :training_compiled_stddev
)''', [ results_dict ])

connection.commit()
connection.close()