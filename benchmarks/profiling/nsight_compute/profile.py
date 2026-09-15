"""Nsight Compute helper — prints ncu profile commands for Triton kernels."""
import argparse

KERNELS = {
    "rmsnorm": "python -c \"import torch; from src.models.triton_kernels.rmsnorm import rmsnorm; x=torch.randn(4,1024,device='cuda'); w=torch.randn(1024,device='cuda'); rmsnorm(x,w)\"",
    "rope": "python -c \"import torch; from src.models.triton_kernels.rope import rope; x=torch.randn(4,64,device='cuda'); c=torch.randn(512,32,device='cuda'); s=torch.randn(512,32,device='cuda'); rope(x,c,s,7)\"",
    "swiglu": "python -c \"import torch; from src.models.triton_kernels.activation import swiglu; g=torch.randn(4,1024,device='cuda'); u=torch.randn(4,1024,device='cuda'); swiglu(g,u)\"",
    "gqa_decode": "python -c \"import torch; from src.models.triton_kernels.attention import gqa_decode_attn; q=torch.randn(16,128,device='cuda'); K=torch.randn(8,128,128,device='cuda'); V=torch.randn(8,128,128,device='cuda'); gqa_decode_attn(q,K,V,0.088)\"",
    "fused_qkv": "python -c \"import torch; from src.models.triton_kernels.attention import fused_qkv; x=torch.randn(512,device='cuda'); w=torch.randn(512,512,device='cuda'); fused_qkv(x,w,w,w)\"",
    "layernorm": "python -c \"import torch; from src.models.triton_kernels.layernorm import layernorm; x=torch.randn(32,512,device='cuda'); w=torch.randn(512,device='cuda'); b=torch.randn(512,device='cuda'); layernorm(x,w,b)\"",
    "conv1d_silu": "python -c \"import torch; from src.models.triton_kernels.conv1d import conv1d_silu; x=torch.randn(1,32,200,device='cuda'); w=torch.randn(32,32,3,device='cuda'); b=torch.randn(32,device='cuda'); conv1d_silu(x,w,b,padding=1)\"",
}

METRICS = "--metrics sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"

def main():
    parser = argparse.ArgumentParser(description="Nsight Compute — individual Triton kernels")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--kernel", choices=list(KERNELS.keys()), default=None)
    args = parser.parse_args()
    base = "ncu --set full --target-processes all --import-source yes"
    if args.list or args.kernel is None:
        print("Nsight Compute — individual Triton kernels (occupancy, SM util, DRAM/L2, FLOP/s, mem, regs, shared, stalls, instr throughput):")
        for k, cmd in KERNELS.items():
            ncu = f'{base} -o profiling/nsight_compute/{k} {cmd}'
            ncu_metrics = f'ncu {METRICS} -o profiling/nsight_compute/{k}_metrics {cmd}'
            print(f"\n  {k}:")
            print(f"    {ncu}")
            print(f"    {ncu_metrics}  # key metrics (achieved occupancy, SM, DRAM, L2, FLOP/s, regs, shared, stall)")
        print("\nUnsupported metrics on some GPUs are skipped automatically (ncu reports not available).")
        print("View: ncu-ui <report>.ncu-rep  or  ncu --import <report>.ncu-rep --page details")
        print("Documented metrics: achieved_occupancy, SM utilization, DRAM/L2 throughput, achieved FLOP/s, register/shared, warp stalls, instruction throughput")
    else:
        cmd = KERNELS[args.kernel]
        print(f"{base} -o profiling/nsight_compute/{args.kernel} {cmd}")

if __name__ == "__main__":
    main()
