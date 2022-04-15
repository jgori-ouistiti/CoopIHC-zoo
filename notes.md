Check for the gradients in Torch

    for name, p in self.policy.named_parameters():
        if p.requires_grad:
            print(name, p.grad)


For `imitation`, install from source:
    
    git clone http://github.com/HumanCompatibleAI/imitation
    cd imitation
    pip install -e .


For errors due to multiprocessing:

    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

