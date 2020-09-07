import torch
import numpy as np 

x = np.arange(-10,10,10)
x = torch.from_numpy(x)

def model(x, unpacked_params):
	y = torch.nn.functional.linear(x,unpacked_params[0],unpacked_params[1])
	print(y)
	for layer in range(1,int(len(unpacked_params)/2)):
		y = torch.sigmoid(y)
		print("No negative values:")
		print(y)
		y = torch.nn.functional.linear(y,unpacked_params[layer*2],unpacked_params[layer*2+1])
		print(y)
	print("before softmax")
	y = my_softmax(y)
	print("Now softmax:")
	print(y)
	return y