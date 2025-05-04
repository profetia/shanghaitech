import torch
import argparse

parser = argparse.ArgumentParser(description='Parse Pytorch Model')

parser.add_argument('--input', type=str, default='./runs/clf_simp_nocnn/classifier_model.pt',
                    help='location of input model')

parser.add_argument('--output', type=str, default='./runs/clf_simp_nocnn/classifier_model.pth',
                    help='location of output model')

args = parser.parse_args()

model = torch.load(args.input)

torch.save(model.state_dict(), args.output)