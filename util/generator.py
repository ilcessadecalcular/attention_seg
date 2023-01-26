import torch


def generator(model,x,crop_size):
    _,t,_,_,_=x.size()
    # print(t)
    over_lap = t % crop_size
    iter = t // crop_size
    output = torch.zeros_like(x)
    output = output[:,0]
    output = output.unsqueeze(1)
    for i in range(iter):
        start_slice = i * crop_size
        end_slice = (i+1) * crop_size
        input = x[:,start_slice:end_slice]
        # print(input.size())
        output1 = model(input)
        output = torch.cat((output,output1),1)
        # print(output.size())
    # print(output.size())
    # print(t-crop_size)
    input = x[:,t-crop_size:]
    # print(end_slice)
    output1 = model(input)
    output1 = output1[:, (crop_size-over_lap+1):, :, :, :]
    # print(output1.size())
    output = torch.cat((output, output1),1)
    # print(output.size())
    return output