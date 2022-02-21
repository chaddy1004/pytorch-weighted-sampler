from weighted_sampler import OverSampler, UnderSampler, NormalSampler, VanillaDataset, VanillaDataset2
from collections import defaultdict


def test1():
    # test for checking different types of sampler
    dataset = VanillaDataset(ratio=10)
    print(f"n_negatives (0): {dataset.n_negatives}, n_positives (1): {dataset.n_positives}")

    over_sampler = OverSampler(dataset=dataset)
    under_sampler = UnderSampler(dataset=dataset)
    normal_sampler = NormalSampler(dataset=dataset)
    # checks for oversampler and undersampler
    samplers = [("oversampler", over_sampler), ("undersampler", under_sampler), ("normalsampler", normal_sampler)]
    for name, sampler in samplers:
        freqs = defaultdict(int)
        for item in sampler:
            _, label = item
            freqs[int(label)] += 1

        print(name)
        print(f"n_negatives(0): {freqs[0]}, n_positives(1): {freqs[1]}")
        print(f"total: {freqs[0] + freqs[1]}")
        print(f"Original Dataset Size: {len(sampler.dataset)}")
        print("\n")


def dict_to_sorted_list(dict):
    final = []
    for data, freq in dict.items():
        final.append((data, freq))

    return sorted(final, key=lambda x:x[0])


def test2():
    # test for checking different types of sampler
    dataset = VanillaDataset(len_normal=5, ratio=2)
    print(f"n_negatives (0): {dataset.n_negatives}, n_positives (1): {dataset.n_positives}")

    num_epochs = 3

    over_sampler = OverSampler(dataset=dataset)
    under_sampler = UnderSampler(dataset=dataset)
    normal_sampler = NormalSampler(dataset=dataset)
    # checks for oversampler and undersampler
    samplers = [("oversampler", over_sampler), ("undersampler", under_sampler), ("normalsampler", normal_sampler)]
    for name, sampler in samplers:
        freqs = defaultdict(int)
        num_datas = []
        pos_samples = 0
        neg_samples = 0
        for epoch in range(num_epochs):
            num_data = 0
            epoch_freqs = defaultdict(int)
            for item in sampler:
                data, label = item
                freqs[int(data)] += 1
                num_data += 1
                if label == 0:
                    neg_samples += 1
                else:
                    pos_samples += 1
                epoch_freqs[int(data)] += 1
            num_datas.append(num_data)

            intermediate = dict_to_sorted_list(epoch_freqs)
            print(f"{name} epoch {epoch}: {intermediate}")

        print(name)
        final = dict_to_sorted_list(freqs)



        print(final)
        print(f"Original Dataset Size: {len(sampler.dataset)}")
        print(f"Num datas per epoch: {num_datas}")

        print(f"(AVG per epoch) n neg samples:{neg_samples/5}, n pos samples:{pos_samples/5}")
        print("*********************************************************************")
        print("\n")



if __name__ == '__main__':
    # test1()
    test2()
