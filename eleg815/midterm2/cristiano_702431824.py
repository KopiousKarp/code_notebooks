"""
This template will help you solve the coding problem in the midterm.

This script contains a function to help you read the command-line 
arguments and some other to give you an initial structure with some
hints in the comments.

Import the necessary libraries and define any functions required for 
your implementation.

"""

import argparse

# Create your model

# Create your data loader

def read_args():
    """This function reads the arguments 
    from the command line. 

    Returns:
        list with the input arguments
    """
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("true"):
            return True
        elif v.lower() in ("false"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()

    parser.add_argument("is_testing", type=str2bool, help="Set Testing Stage")
    parser.add_argument("ckpt_path", nargs='?', type=str, help="Checkpoint path", default='./')
    parser.add_argument("testing_data_path", nargs='?', type=str, help="Testing data path", default='./')
    parser.add_argument("output_path", nargs='?', type=str, help="output path", default='./')

    return parser.parse_args()

def train():
    """
    Do everything related to the training of your model
    """

    # Instantiate your model

    # Instantiate your datasets

    # Instantiate your dataloaders

    # Setup your optimizer

    # Implement the training loop

    # Save the model

    print('Start training')

def test(input_args):
    """
    Do everything related to the testing of your model
    """

    print('Start testing')
    # read data from input_args.testing_data_path
    # load model with weights from input_args.ckpt_path

    # compute model predictions

    # save [studentname_udid].npy file in input_args.output_path

def main():
    """This is the main function of yor script.
    From here you call your training or 
    testing functions.
    """
    input_args = read_args()

    if input_args.is_testing:
        test(input_args)
    else:
        train()

if __name__=="__main__":
    main()
