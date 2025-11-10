##Train the differential IDS channel in advance. 

**The user may wanna use the pretrained file:** `ids_channel_dl_embsize512_numlayer1_nhead4_150-200.pth`

Readers who wanna train their own differential IDS channel, can modify the parameters as they need in file "train_ids_channel_dl.py", and execute:

```
python3 train_ids_channel_dl.py
```

## Train model
```
python3 main.py --gpu=1 --seed=1000 --epoch=600 --length-dna=100 --ratio=0.01 --nins=10 --ndel=10 --nsub=10 --path-prefix='THEA-Code' --batchsize=64 --aux-q=1 --gumbel --gumbel-temperature=1 --gumbel-q=1 --dropout=0.2 --emb-size 512 --hid-size 512 --nhead 8 --enc-layer=3 --dec-layer=3  --focal-loss-gamma=0 --warmup=200 --train-x=1
```

## Standalone test 

```
python3 standalone_test.py --model-path='./results/1000_THEA-Code/' --length-dna=100 --gpu=0
```

## Citation

```
@inproceedings{guo2025disturbance,
  author    = {Guo, Alan JX and Wei, Mengyi and Dai, Yufan and Wei, Yali and Zhang, Pengchen},
  title     = {{Disturbance-based Discretization, Differentiable IDS Channel, and an IDS-Correcting Code for DNA-based Storage}},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
}
```
