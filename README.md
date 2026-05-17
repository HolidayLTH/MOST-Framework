# MOST Project

## How to Run

Example:

```bash
python train.py --events E1 E2 E3 E4 E5 --model TFT-GAT > record.out 2>&1
```

You can switch the event pool, for example:

```bash
python train.py --events E6 E7 E8 E9 E10 --model TFT-GAT > record.out 2>&1
```

You can also switch the model, for example:

```bash
python train.py --events E1 E2 E3 E4 E5 --model LSTM > record.out 2>&1
```

Supported models:
- TFT-GAT
- LSTM
- Transformer
- STGCN
- DCRNN
- GraphWaveNet
- GMAN
- STTN
- PI-MPN
- PAG-STAN

## How to Examine

After running all event pools and models, the Weights and Log folders will be created to store model checkpoints and evaluation results.

You can run the summary script to view aggregated results:

```bash
python summary.py
```

You can edit the event sets inside [summary.py](summary.py) to inspect different training setups.

## Repository Structure

- EWE_data: OD flow and weather records for 17 extreme weather events over 10 years. 
- Cache: cached preprocessed data.
- Network Features: network structure features for different metro network periods.
- Model: model implementations mentioned in the paper. TFT-GAT is MOST, others are baselines.
- Profile: configuration parameters.
- Local Outputs: local results consistent with the paper.
- Weights: saved model checkpoints.
- Log: evaluation logs and metrics.
- [train.py](train.py): training entry point.
- [summary.py](summary.py): performance summary entry point.

## Notes

- Due to CUDA randomness and differences in algorithm environments, results may vary slightly.
- Only the first 10 events are currently available. The remaining 7 events will be released later. If needed, please email wslth2016@gmail.com for access.
