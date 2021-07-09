import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--boolean", default=False, type=str2bool, const=True, nargs='?')
    parser.add_argument("--arg2", default='string', type=str)
    args = parser.parse_args(('--boolean', 'True', '--arg2', 'test'))
    print(args)
    args = parser.parse_args()
    print(args)
    args = parser.parse_args(())
    print(args)
    args = parser.parse_args(None)
    print(args)