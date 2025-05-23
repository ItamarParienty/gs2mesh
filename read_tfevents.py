import tensorflow as tf
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("example")
    parser.add_argument("path")
    args = parser.parse_args()

    for e in tf.compat.v1.train.summary_iterator(args.path):
        print("Step:", e.step)
        for v in e.summary.value:
            print(f"  Tag: {v.tag}")
            if v.HasField("simple_value"):
                print(f"    Value: {v.simple_value}")
            elif v.HasField("tensor"):
                print(f"    Tensor: {v.tensor}")
