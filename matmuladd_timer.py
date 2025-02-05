import os
import sys
import numpy as np
import tvm
from tvm import te, auto_scheduler
import time
import shutil

@auto_scheduler.register_workload
def matmul_add(M, N, K, dtype):
    A = te.placeholder((M, K), name = "A", dtype = dtype)
    B = te.placeholder((K, N), name = "B", dtype = dtype)
    C = te.placeholder((M, N), name = "C", dtype = dtype)

    k = te.reduce_axis((0, K), name = "k")
    matmul = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis = k),
        name = "matmul",
        attrs = {"layout_free_placeholders": [B]},
    )
    out = te.compute((M, N), lambda i, j: matmul[i, j] + C[i, j], name = "out")
    return [A, B, C, out]


def main():
    argv = sys.argv
    #print(len(argv))
    if len(argv) != 4:
        print("Invalid input!")
        print("python matmuladd_timer.py M N K!")
        exit(1)
    
    M = int(argv[1])
    N = int(argv[2])
    K = int(argv[3])
    print(f"mm_gencode: M={M}, N={N} K={K}")
    foldername = "M" + str(M) + "_N" + str(N) + "_K" + str(K) + "/"
    if os.path.exists(foldername):
        shutil.rmtree(foldername)
    os.mkdir(foldername)
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(M, N, K, "float32"), target=target)
    log_file = foldername + "matmul_add.json"

    tune_option = auto_scheduler.TuningOptions(
        #num_measure_trials=1000,
        num_measure_trials=10,
        #runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
        runner=auto_scheduler.LocalRunner(repeat=2, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=0,
    )
    # Step 1: generate llvm code file
    # Run auto-tuning (search)
    task.tune(tune_option)
    # Apply the best schedule
    sch, args = task.apply_best(log_file)
    func = tvm.build(sch, args, target, name="matmul_add")
    ll_code_str = func.get_source()
    ll_code_filename = foldername + "matmul_add.ll"
    ll_code_file = open(ll_code_filename, "w")
    ll_code_file.write(ll_code_str)
    ll_code_file.close()
    time.sleep(2)
    print("[Step 1: generate llvm code file] COMPLETED!")

    # Step 2: insert timestamp
    ll_code_file = open(ll_code_filename, 'r')
    lines, linecounter = ll_code_file.readlines(), 0
    # start
    for line in lines:
        if (line.find('define dllexport i32 @matmul_add(')!=-1 and lines[linecounter+1].strip()=='entry:'):
            lines.insert(linecounter+2, '  call void @get_time(i32 1)\n')
            break
        linecounter += 1

    # end
    linecounter = 0
    for line in lines:
        if (line.find('tail call fastcc i32 @matmul_add_compute_(')!=-1 and lines[linecounter+1].find('br label %common.ret')!=-1):
            lines[linecounter] = lines[linecounter].replace('tail', '')
            lines.insert(linecounter+1, '  call void @get_time(i32 0)\n')
            break
        linecounter += 1
    
    # add get_time(i32) declaration
    newLines, linecounter = [], 0
    for line in lines:
        if (line.find('attributes #0 = {')!=-1 and lines[linecounter+1].find('attributes #1 = {')!=-1):
            newLines.extend(lines[0:linecounter-1])
            newLines.insert(linecounter, 'declare void @get_time(i32)\n\n')
            newLines.extend(lines[linecounter:])
            break
        linecounter += 1
    ll_code_file.close()

    ll_code_newfilename = foldername + "matmul_add_new.ll"
    ll_code_newfile = open(ll_code_newfilename, 'w')
    ll_code_newfile.writelines(newLines)
    ll_code_newfile.close()
    time.sleep(2)
    print("[Step 2: insert timestamp] COMPLETED!")

    # Step 3: generate so file
    # clang -shared -fPIC -mavx2 -march=native -O3 M1024_N1024_K1024/matmul_add_new.ll -o M1024_N1024_K1024/matmul_add.so gettime.c
    sofile_path = f"{foldername}matmul_add.so"
    gen_so_cmd = f"clang -shared -fPIC -mavx2 -march=native -O3 {ll_code_newfilename} -o {sofile_path} gettime.c"
    os.system(gen_so_cmd)
    print("[Step 3 : generate so file] COMPLETED!")


    # Step 4: timer evaluation
    if not os.path.exists(sofile_path):
        print(f"Error: sofile <{sofile_path}> does not exist!")
        exit(1)
    
    np.random.seed(149)
    a_np = np.random.uniform(size=(M, K)).astype(np.float32)
    b_np = np.random.uniform(size=(K, N)).astype(np.float32)
    c_np = np.random.uniform(size=(M, N)).astype(np.float32)
    out_np = a_np.dot(b_np) + c_np

    dev = tvm.cpu()
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.array(c_np, device=dev)
    out_tvm = tvm.nd.empty(out_np.shape, device=dev)

    module = tvm.runtime.load_module(sofile_path)
    fmatmul_add = module['matmul_add']
    print(f"Matmuladd Evaluation for M={M} N={N} K={K}")
    thread_nums = [0, 8, 16, 20]
    for threadnum in thread_nums:
        if threadnum == 0:
            print("ThreadNum by default")
        else:
            os.environ["TVM_NUM_THREADS"] = str(threadnum)
            print(f"ThreadNum = {threadnum}")

        for i in range(9):
            print(f"Iteration {i}:")
            fmatmul_add(a_tvm, b_tvm, c_tvm, out_tvm)
            np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
        print()

    print("[Step 4: timer evaluation] COMPLETED!")


if __name__ == "__main__":
    main()