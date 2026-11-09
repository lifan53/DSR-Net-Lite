import torch
from thop import profile

from utils.tools import get_config
from utils.learning import load_model


def measure_complexity(config_path):

    print("\n==============================")
    print("Config:", config_path)

    args = get_config(config_path)

    model = load_model(args)
    model = model.cuda()
    model.eval()

    torch.backends.cudnn.benchmark = True

    # MPI-INF-3DHP input
    x = torch.randn(
        1,
        args.n_frames,
        args.num_joints,
        3
    ).cuda()

    # =============================
    # Params
    # =============================
    params = sum(
        p.numel()
        for p in model.parameters()
    )

    # =============================
    # MACs
    # =============================
    with torch.no_grad():
        macs, _ = profile(
            model,
            inputs=(x,),
            verbose=False
        )

    # =============================
    # Warm-up
    # =============================
    with torch.no_grad():
        for _ in range(50):
            _ = model(x)

    torch.cuda.synchronize()

    # =============================
    # Latency
    # =============================
    repeat = 500

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():

        starter.record()

        for _ in range(repeat):
            _ = model(x)

        ender.record()

    torch.cuda.synchronize()

    total_time = starter.elapsed_time(ender)

    latency = total_time / repeat
    fps = 1000.0 / latency

    # =============================
    # Results
    # =============================
    print("--------------------------------")
    print(f"Input shape : {tuple(x.shape)}")
    print(f"Params      : {params / 1e6:.2f} M")
    print(f"MACs        : {macs / 1e9:.2f} G")
    print(f"Latency     : {latency:.2f} ms")
    print(f"FPS         : {fps:.2f}")
    print("--------------------------------")

    return {
        "params": params / 1e6,
        "macs": macs / 1e9,
        "latency": latency,
        "fps": fps,
    }


if __name__ == "__main__":

    teacher_config = "configs/mpi/MotionAGFormer-DCF.yaml"

    student_config = "configs/mpi/MotionAGFormer-DCF-T10-W112.yaml"

    print("\n===== MPI Teacher =====")
    teacher_result = measure_complexity(teacher_config)

    print("\n===== MPI Final Student =====")
    student_result = measure_complexity(student_config)