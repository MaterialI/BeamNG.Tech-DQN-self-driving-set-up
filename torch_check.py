import torch


def check_gpu():
    return torch.cuda.is_available()


if __name__ == "__main__":
    print(check_gpu())
