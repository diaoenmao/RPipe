class Linear(nn.Module):
    def __init__(self, data_shape, target_size):
        super().__init__()
        input_size = math.prod(data_shape)
        self.ln = nn.LayerNorm(input_size, elementwise_affine=False)
        self.linear = nn.Linear(input_size, target_size, bias=False)
        self.scaler = nn.Parameter(torch.ones(1, ))

    def feature(self, x):
        x = x.reshape(x.size(0), -1)
        return x

    def output(self, x):
        x = self.linear(x)
        return x

    def f(self, x):
        x = self.feature(x)
        x = self.output(x)
        return x

    def forward(self, input):
        output = {}
        x = input['data']
        x = self.feature(x)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        # norm = (x.pow(2) / x.size(0)).sum(dim=-1, keepdim=True).sqrt()
        # x = x / norm.detach()
        # x = (x - mean) / std
        # x = self.ln(x)
        x = self.output(x)
        weight_norm = torch.linalg.norm(self.linear.weight, dim=-1)
        # weight_norm = (self.linear.weight.pow(2) / self.linear.weight.size(0)).sum(dim=-1).sqrt()
        x = x / weight_norm.detach()
        # x = x * self.scaler
        output['target'] = x
        output['loss'] = torch.nn.functional.cross_entropy(output['target'], input['target'])

        # output = {}
        # x = input['data']
        # x = self.f(x)
        # output['target'] = x
        # output['loss'] = make_loss(output, input)

        return output