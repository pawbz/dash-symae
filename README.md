# Dash App for SymAE

## About this app

This app showcases the ability of SymAE to process synthetic passive seismic data.
SymAE is a novel physics-embedded deep-network architecture that can easily make sense of large amounts of passive seismic datasets. SymAE learns to represent seismic noise such that its source and path effects are disentangled in order to facilitate easy monitoring of these effects. For the first time, we can now "mix and match" seismic data, i.e. synthesize data with the source of one (baseline) seismogram, and with the medium of another (monitor) one.


## How to run this app

(The following instructions apply to Windows command line.)

To run this app first clone repository and then open a terminal to the app folder.

```
git clone https://github.com/plotly/dash-sample-apps.git
cd dash-sample-apps/apps/studybrowser
```

Create and activate a new virtual environment (recommended) by running
the following:

On Windows

```
virtualenv venv 
\venv\scripts\activate
```

Or if using linux

```bash
python3 -m venv myvenv
source myvenv/bin/activate
```

Install the requirements:

```
pip install -r requirements.txt
```
Run the app:

```
python app.py
```
You can run the app on your browser at http://127.0.0.1:8050

## Resources

To learn more about Dash, please visit [documentation](https://plot.ly/dash).
