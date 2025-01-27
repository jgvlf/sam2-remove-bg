# sam2-remove-bg
Use the Sam2 Meta segmentation model to get the object of interest and save this as image.

## Setup
To setup your environment, you need **create a python venv** and **activate** it with:
For Windows PowerShell:

```bash
python -m venv .venv
.\venv\Scripts\activate.ps1
pip install -r .\requirements.txt
```

For Windows CMD:
```bash
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

After that, you need the **model weight** that is available in the **[Sam2 Github Repo](https://github.com/facebookresearch/sam2#:~:text=Download%20Checkpoints)** and to complete this part, you need the **model configuration files**. You can't simple copy the configuration file to your work dir, you should **copy the entire config folder**. To do so, you have might installed the dependencies in your python venv (and the sam2 dependence is in this file) so, copy the **config** folder in this path: **`.venv\Lib\site-packages\sam2\configs`** to your root work directory.

With this, your setup is now configured.