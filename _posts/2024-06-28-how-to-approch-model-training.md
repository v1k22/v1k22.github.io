# Model Trainig is no joke 🥲

Hi, While learning about GPT2 & GPT3 training, I came to know that there can be many thing which could go wrong while training model from scratch. This post is just an high level guide on how
you should normalize the flow while writing new models.

### Step 1. 
Write down simple model that returns ```logits & loss```. Test the flow with just 1 epoch. Make sure to use smaller batch size, input length (if flexible) for faster flow testing. Later we 
will modified them.

### Step 2. 
Run the flow for 50 to 100 epochs and get the worst to best loss interval. For example, ```[11, 7.21]``` This will be your base set to beat with following more deeper optimizations.

### Step 3. 
Now implement weight initialization properly (look at pytorch doc for gain parameter). Get the loss after 50 epochs and compare them with earlier loss interval.

### Step 4.
Check & Add batch norm layer if required. This you will learn more by performing many hands on.

### Step 5. 
Add residual layer if required. This will come with more hands on experience.

### Step 6.
**Speed Optimization**: Add following for more speed,
- ```torch.set_float32_matmul_precision('high')``` for faster calculation (only supported on cuda devices Ampere=>)
- use ```torch.compile```
- changing ugly numbers (i.e. not power of 2s) to beautiful numbers (i.e. power of 2s), Andrej showed 30% gain using this

### Step 7.
**Speed Optimization**: Applying up to date new techniques such as (self-attention 130ms -> flash attention 96ms)

### Step 8.
clipping the grad norm using ```norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)```

### Step 9. 
Introduce more parameters to your Optimizer AdamW, add decay, biases etc.

### Step 10. (Most effective)
Add Gradient Accumulation steps for larger batch size. This will slow down your performance but accuracy gain is worth more.
For sample output, I am attaching the loss results of GPT2 on smaller dataset, they look really good, feels like it is learning accuratly,

![image](https://github.com/v1k22/v1k22.github.io/assets/8783818/27341132-8230-4078-840e-233420bf1057)

```
step 0, loss: 10.962733268737793, norm: 27.9385, dt: 34471.31ms
step 1, loss: 9.589015007019043, norm: 5.3188, dt: 32755.56ms
step 2, loss: 9.13232707977295, norm: 4.1961, dt: 32001.27ms
step 3, loss: 8.880989074707031, norm: 4.5941, dt: 32389.64ms
step 4, loss: 8.560948371887207, norm: 2.3522, dt: 32078.73ms
step 5, loss: 8.29920482635498, norm: 2.8669, dt: 31999.96ms
step 6, loss: 8.04935073852539, norm: 2.5591, dt: 32183.77ms
step 7, loss: 7.833332538604736, norm: 1.7445, dt: 32352.16ms
step 8, loss: 7.630918025970459, norm: 1.5868, dt: 32235.64ms
step 9, loss: 7.399796485900879, norm: 1.4583, dt: 32203.54ms
step 10, loss: 7.177652359008789, norm: 1.5820, dt: 32261.35ms
step 11, loss: 6.982940673828125, norm: 1.9159, dt: 32309.55ms
step 12, loss: 6.805536270141602, norm: 1.4001, dt: 32387.50ms
step 13, loss: 6.699013710021973, norm: 1.5037, dt: 32443.15ms
step 14, loss: 6.596346855163574, norm: 1.2662, dt: 32331.29ms
step 15, loss: 6.435312747955322, norm: 1.0102, dt: 32180.41ms
step 16, loss: 6.385463714599609, norm: 1.3616, dt: 32259.49ms
step 17, loss: 6.368933200836182, norm: 0.7513, dt: 32291.44ms
step 18, loss: 6.2556681632995605, norm: 0.9366, dt: 32202.13ms
step 19, loss: 6.235473155975342, norm: 0.6751, dt: 32234.96ms
step 20, loss: 6.2077107429504395, norm: 1.2539, dt: 32205.70ms
step 21, loss: 6.181078910827637, norm: 0.6101, dt: 32337.98ms
step 22, loss: 6.103849411010742, norm: 0.7067, dt: 32381.10ms
step 23, loss: 6.116004467010498, norm: 0.6489, dt: 32402.43ms
step 24, loss: 6.153818130493164, norm: 0.5105, dt: 32326.94ms
step 25, loss: 6.1066484451293945, norm: 0.5766, dt: 32366.29ms
step 26, loss: 6.059800624847412, norm: 0.4205, dt: 32460.03ms
step 27, loss: 6.021355152130127, norm: 0.4427, dt: 32454.79ms
step 28, loss: 6.0182414054870605, norm: 0.3283, dt: 32486.34ms
step 29, loss: 6.018798351287842, norm: 0.3673, dt: 32487.47ms
step 30, loss: 5.9879655838012695, norm: 0.4408, dt: 32466.07ms
step 31, loss: 5.945150852203369, norm: 0.3967, dt: 32420.57ms
step 32, loss: 5.936180591583252, norm: 0.3984, dt: 32527.74ms
step 33, loss: 5.908405303955078, norm: 0.4562, dt: 32468.13ms
step 34, loss: 5.965937614440918, norm: 0.5021, dt: 32412.33ms
step 35, loss: 5.948241710662842, norm: 0.5184, dt: 32474.06ms
step 36, loss: 5.875123500823975, norm: 0.3643, dt: 32480.88ms
step 37, loss: 5.8688225746154785, norm: 0.2565, dt: 32457.33ms
step 38, loss: 5.878945350646973, norm: 0.3597, dt: 32460.10ms
step 39, loss: 5.841239929199219, norm: 0.3006, dt: 32499.71ms
step 40, loss: 5.890498638153076, norm: 0.3253, dt: 32589.11ms
step 41, loss: 5.838111400604248, norm: 0.6233, dt: 32576.72ms
step 42, loss: 5.809896945953369, norm: 1.4170, dt: 32547.44ms
step 43, loss: 5.755493640899658, norm: 0.7072, dt: 32579.02ms
step 44, loss: 5.796072959899902, norm: 0.6466, dt: 32524.84ms
step 45, loss: 5.780944347381592, norm: 0.9231, dt: 32619.49ms
step 46, loss: 5.745007038116455, norm: 0.5361, dt: 32617.70ms
step 47, loss: 5.766404628753662, norm: 0.5035, dt: 32603.72ms
step 48, loss: 5.736897945404053, norm: 0.7908, dt: 32647.66ms
step 49, loss: 5.733333110809326, norm: 1.1660, dt: 32577.68ms
```



### References:
[Lets Reproduce GPT2(124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
