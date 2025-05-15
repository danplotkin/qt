from datasets import load_dataset, Dataset


def get_minipile():
    ds = load_dataset("JeanKaddour/minipile")
    ds.save_to_disk('data/minipile')


if __name__ == '__main__':
    get_minipile()