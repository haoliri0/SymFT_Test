import argparse

import symft


def main():
    parser = argparse.ArgumentParser(description="Sample logical-error counts from a Stim file.")
    parser.add_argument("circuit", help="path to a .stim circuit")
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    circuit = symft.read_stim_file(args.circuit)
    sampler = circuit.compile_counts_sampler(threads=args.threads)

    print(circuit)
    print(sampler.info)
    print(sampler.sample(args.shots))


if __name__ == "__main__":
    main()
