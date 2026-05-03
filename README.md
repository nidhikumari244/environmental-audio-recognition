# Environmental Audio Recognition for Surveillance

Streamlit deployment package for a CNN-based environmental sound recognition system built on the ESC-50 dataset.

## Included

- `app.py` - Streamlit app entrypoint
- `demo_config.py` - class names and prepared demo samples
- `demo_audio/` - 15 prepared demo WAV files
- `demo_images/` - representative demo images
- `models/` - saved Keras model files
- `cnn_results_final_79.json` - final evaluation result used in report/presentation

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Push this folder to a GitHub repository and deploy it on Streamlit Community Cloud with:

- Repository: your GitHub repo
- Branch: `main`
- Entrypoint file: `app.py`
