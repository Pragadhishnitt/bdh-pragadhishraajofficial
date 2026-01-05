# Sentiment Analysis with BDH Architecture

A modular, research-grade implementation of sentiment analysis using the BDH (Biologically-inspired Deep Hebbian) architecture with emphasis on continual learning and concept drift adaptation.

## Project Structure

```
sentiment_analysis/
├── configs/          # Configuration files
├── data/            # Data loaders and processors
├── models/          # Model architectures
├── utils/           # Utility functions
├── evaluation/      # Metrics and evaluation
├── visualization/   # Plotting and visualization
├── checkpoints/     # Model checkpoints
├── results/         # Training results and logs
├── train.py         # Main training script
└── README.md        # This file
```

## Features

- **Continual Learning**: Preserves temporal ordering to demonstrate concept drift adaptation
- **Modular Design**: OOP-based architecture for extensibility
- **Comprehensive Metrics**: Standard classification + continual learning metrics
- **Baseline Comparison**: Compare BDH against DistilBERT
- **Rich Visualizations**: Temporal analysis, confusion matrices, and more

## Usage

```bash
# Train BDH model (warm-up run with 100 samples)
python sentiment_analysis/train.py --config configs/bdh_warmup.yaml

# Train baseline model
python sentiment_analysis/train_baseline.py --config configs/baseline_warmup.yaml

# Generate visualizations
python sentiment_analysis/visualize.py --results_dir results/
```

## Requirements

See `requirements.txt` in the parent directory.
