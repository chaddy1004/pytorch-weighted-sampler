from weighted_sampler import OverSampler, UnderSampler, NormalSampler, VanillaDataset
from collections import defaultdict


def test(sampler, name):
    freqs = defaultdict(int)
    for item in sampler:
        freqs[int(item)] += 1

    print(name)
    print(f"n_negatives(0): {freqs[0]}, n_positives(1): {freqs[1]}")
    print(f"total: {freqs[0] + freqs[1]}")
    print("\n")


if __name__ == '__main__':
    dataset = VanillaDataset(ratio=10)

    print(f"n_negatives (0): {dataset.n_negatives}, n_positives (1): {dataset.n_positives}")

    over_sampler = OverSampler(dataset=dataset)
    under_sampler = UnderSampler(dataset=dataset)
    normal_sampler = NormalSampler(dataset=dataset)

    test(over_sampler, name="oversampler")
    test(under_sampler, name="undersampler")
    test(normal_sampler, name="normalsampler")
