# Cross Entropy

**Usage:** To calculate loss

The challenge is to understand expected input and output format. Lets understand it,

Suppose our training data has

```python
batch_size = 4
input_size = 8
number_of_classes = 10
```

after performing forward pass in model, lets say this is our output,

```python
out = model(input)
out.shape

# [4, 8, 32] here 32 will be size of output layer, it can be probabilities or anything
```

Cross Entropy expects input in following format,

```python
def forward(self, idx, targets):
        logits = self.layer_output(idx)

        # understand input format
        # b = number of batches, l = size of 1 batch, c = classes
        b, l, c = logits.shape
        
        # cross_entroy expects format to be
        # input = (c, n)
        # targets = (n) here n is batch * ...

        resized_logits = logits.view(b*l, c)
        resized_targets = targets.view(b*l)

        loss = F.cross_entropy(resized_logits, resized_targets)
        return logits, loss
```

### References:

[1. pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)

[2. pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)
