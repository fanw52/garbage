import numpy as np
from torch import nn
nn.LSTM()
nn.Softmax()


class Conv:
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding='SAME', channel_mode='NWHC'):
        assert padding in ['SAME', 'VALID']

        self.kernel_size = kernel_size
        self.padding = padding
        self.c_in = c_in
        self.c_out = c_out
        self.stride = stride

        self.kernel = np.random.normal(size=[kernel_size, kernel_size, c_in, c_out])

    def padding_op(self, input):
        if self.padding == 'SAME':
            padding_width = (self.kernel_size - 1) // 2
            return np.pad(input, ((0, 0), (padding_width, padding_width), (padding_width, padding_width), (0, 0)))

    def flatten_op(self, input):
        flatted_input, flatted_kernel = [], []

        self.n, self.w, self.h, self.c = input.shape
        input = self.padding_op(input)  # [N,W,H,C]
        kernel = self.kernel.reshape([self.c_in * self.kernel_size ** 2, self.c_out])
        for i in range(0, self.w, self.stride):
            for j in range(0, self.h, self.stride):
                x = input[:, i:i + self.kernel_size, i:i + self.kernel_size, :].reshape([self.n, -1])
                flatted_input.append(x)
                flatted_kernel.append(kernel)
        flatted_input = np.array(flatted_input)
        flatted_kernel = np.array(flatted_kernel)
        return flatted_input, flatted_kernel

    def forword(self, input):
        flatted_input, flatted_kernel = self.flatten_op(input)
        out = np.einsum('ijk,ikh->ijh', flatted_input, flatted_kernel)
        out = np.reshape(out, (self.n, self.w, self.h, self.c_out))
        return out


if __name__ == '__main__':
    input = np.random.normal(size=[10, 5, 5, 31])
    conv = Conv(c_in=31, c_out=5)

    x = conv.forword(input)
    print(x.shape)
