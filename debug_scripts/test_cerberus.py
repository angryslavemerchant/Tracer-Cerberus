import numpy as np
import hailo_platform as hpf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEF_PATH = str(ROOT / "Models" / "CerberusCoreS.hef")

TEMPLATE_SIZE = 128
SEARCH_SIZE   = 256


def main():
    print(f"[INIT] Loading HEF from {HEF_PATH}")
    hef = hpf.HEF(HEF_PATH)

    print("\n[MODEL] Input streams:")
    for info in hef.get_input_vstream_infos():
        print(f"  name={info.name}")
        print(f"  shape={info.shape}")
        print(f"  format={info.format}")

    print("\n[MODEL] Output streams:")
    for info in hef.get_output_vstream_infos():
        print(f"  name={info.name}")
        print(f"  shape={info.shape}")
        print(f"  format={info.format}")

    print("\n[INIT] Opening VDevice")
    with hpf.VDevice() as target:
        configure_params = hpf.ConfigureParams.create_from_hef(
            hef, interface=hpf.HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_infos  = hef.get_input_vstream_infos()
        output_infos = hef.get_output_vstream_infos()

        input_params = hpf.InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)
        output_params = hpf.OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

        print("\n[INF] Running dummy inference with random inputs...")
        with network_group.activate(network_group_params):
            with hpf.InferVStreams(network_group, input_params, output_params) as pipeline:
                # Build input dict — random float32 in [0, 255] for each input
                input_data = {
                    info.name: np.random.randint(0, 255, (1, *info.shape)).astype(np.float32)
                    for info in input_infos
                }

                print(f"[INF] Input keys: {list(input_data.keys())}")
                for k, v in input_data.items():
                    print(f"  {k}: shape={v.shape} dtype={v.dtype}")

                results = pipeline.infer(input_data)

                print(f"\n[OUT] Output keys: {list(results.keys())}")
                for name, val in results.items():
                    print(f"\n  '{name}':")
                    if isinstance(val, np.ndarray):
                        print(f"    type=ndarray  shape={val.shape}  dtype={val.dtype}")
                        print(f"    min={val.min():.4f}  max={val.max():.4f}")
                        print(f"    first values: {val.flat[:10]}")
                    elif isinstance(val, list):
                        print(f"    type=list  len={len(val)}")
                        def inspect(obj, depth=0):
                            pad = "  " * depth
                            if isinstance(obj, np.ndarray):
                                print(f"{pad}ndarray shape={obj.shape} dtype={obj.dtype}", end="")
                                if obj.size > 0:
                                    print(f"  min={obj.min():.4f} max={obj.max():.4f}")
                                else:
                                    print(" (empty)")
                            elif isinstance(obj, list):
                                print(f"{pad}list len={len(obj)}")
                                for i, item in enumerate(obj[:4]):
                                    inspect(item, depth+1)
                            else:
                                print(f"{pad}{type(obj).__name__}: {obj}")
                        inspect(val)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
