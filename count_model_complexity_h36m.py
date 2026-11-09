import time
import torch
from thop import profile

from utils.tools import get_config
from utils.learning import load_model
from utils.tools import count_param_numbers


def count_model(config_path, warmup=30, runs=100):

    print("=" * 50)
    print("Config:", config_path)

    args = get_config(config_path)

    # =========================
    # Load model
    # =========================
    model = load_model(args)
    model.eval()

    # 如果有 GPU，则使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # =========================
    # Params
    # =========================
    params = count_param_numbers(model)

    print(f"Params : {params / 1e6:.2f} M")

    # =========================
    # Input
    # Human3.6M:
    # batch = 1
    # frames = 243
    # joints = 17
    # xyz = 3
    # =========================
    x = torch.randn(
        1,
        args.n_frames,
        args.num_joints,
        3,
        device=device
    )

    print(f"Input shape : {tuple(x.shape)}")

    # =========================
    # MACs
    # =========================
    try:

        macs, params_thop = profile(
            model,
            inputs=(x,),
            verbose=False
        )

        print(f"MACs   : {macs / 1e9:.2f} G")

        # 仅作为参考：
        # 按 1 MAC ≈ 2 FLOPs 估算
        print(f"FLOPs  : {2 * macs / 1e9:.2f} G")

    except Exception as e:

        print("THOP ERROR:")
        print(e)

        macs = None

    # =========================
    # Latency
    # =========================
    print("Measuring latency...")

    with torch.no_grad():

        # Warm-up
        for _ in range(warmup):
            _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Start timing
        start = time.perf_counter()

        for _ in range(runs):
            _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

    # 单次推理平均时间
    latency = (end - start) / runs * 1000

    # =========================
    # FPS
    # =========================
    fps = 1000.0 / latency

    print(f"Latency : {latency:.2f} ms")
    print(f"FPS     : {fps:.2f}")

    return {
        "params": params / 1e6,
        "macs": macs / 1e9 if macs is not None else None,
        "latency": latency,
        "fps": fps
    }


if __name__ == "__main__":

    # =========================
    # Human3.6M Teacher
    # =========================
    teacher_config = \
        "configs/h36m/MotionAGFormer_DCF3.yaml"

    # =========================
    # Human3.6M Final Student
    # 10 layers + width 96
    # =========================
    student_config = \
        "configs/h36m/MotionAGFormer_DCF3_CKA_student_w96.yaml"

    # =========================
    # Teacher
    # =========================
    print("\n===== Human3.6M Teacher =====")

    teacher_result = count_model(
        teacher_config,
        warmup=30,
        runs=100
    )

    # =========================
    # Final Student
    # =========================
    print("\n===== Human3.6M Final Student =====")

    student_result = count_model(
        student_config,
        warmup=30,
        runs=100
    )

    # =========================
    # Summary
    # =========================
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    print("\nTeacher DSR-Net:")
    print(f"Params  : {teacher_result['params']:.2f} M")
    print(f"MACs    : {teacher_result['macs']:.2f} G")
    print(f"Latency : {teacher_result['latency']:.2f} ms")
    print(f"FPS     : {teacher_result['fps']:.2f}")

    print("\nFinal Student:")
    print(f"Params  : {student_result['params']:.2f} M")
    print(f"MACs    : {student_result['macs']:.2f} G")
    print(f"Latency : {student_result['latency']:.2f} ms")
    print(f"FPS     : {student_result['fps']:.2f}")