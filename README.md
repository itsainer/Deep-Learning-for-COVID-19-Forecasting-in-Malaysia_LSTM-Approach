# Deep-Learning-for-COVID-19-Forecasting-in-Malaysia_LSTM-Approach
Propose a deep learning model using LSTM neural network to predict new Covid-19 cases in Malaysia using past 30 days of number of cases. The approaches applied are Single-Step LSTM and Multi-Steps LSTM.

Source of dataset: https://github.com/MoH-Malaysia/covid19-public
Citation: MoH-Malaysia/covid19-public: Official data on the COVID-19 epidemic in Malaysia. Powered by CPRC, CPRC Hospital System, MKAK, and MySejahtera.

This repository consists of:

1. Code (python)
2. Dataframes used (.csv)
3. Saved models (.h5 format)
4. Models' architecture
5. Models' performance
6. Graphs of predicted cases vs actual cases

### Single Step LSTM
- Model architecture
  ![ss_architecture](https://github.com/itsainer/Deep-Learning-for-COVID-19-Forecasting-in-Malaysia_LSTM-Approach/assets/106145330/ce042185-2259-40d1-ab91-7b7e8f705e65)

- Model performance (Loss)
  ![ss_loss](https://github.com/itsainer/Deep-Learning-for-COVID-19-Forecasting-in-Malaysia_LSTM-Approach/assets/106145330/89ed73c4-8dab-4cf6-93e9-efd21cfcd176)

- Graphs (Predicted vs Actual Cases)
  ![ss_model_pred](https://github.com/itsainer/Deep-Learning-for-COVID-19-Forecasting-in-Malaysia_LSTM-Approach/assets/106145330/1095b8e7-dd97-4f25-9abf-ba419c7a5f0e)

### Multi-Steps LSTM
- Model architecture
  ![ms_architecture](https://github.com/itsainer/Deep-Learning-for-COVID-19-Forecasting-in-Malaysia_LSTM-Approach/assets/106145330/c52fa039-96c2-473c-8f2d-e998737889ef)

- Model performance (Loss)
  ![ms_loss](https://github.com/itsainer/Deep-Learning-for-COVID-19-Forecasting-in-Malaysia_LSTM-Approach/assets/106145330/09744160-626d-452c-ab2f-c14a34746174)

- Graphs (Predicted vs Actual Cases)
  ![ms_model_pred](https://github.com/itsainer/Deep-Learning-for-COVID-19-Forecasting-in-Malaysia_LSTM-Approach/assets/106145330/ec5ec4b4-fbeb-4656-87e8-2c15d46e572c)
