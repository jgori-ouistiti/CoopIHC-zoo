Check for the gradients

    for name, p in self.policy.named_parameters():
        if p.requires_grad:
            print(name, p.grad)