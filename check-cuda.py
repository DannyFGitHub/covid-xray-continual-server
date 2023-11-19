import torch

def main():
    torch.cuda.is_available()
    print(torch.cuda.is_available())


if __name__ == "__main__":
    main()