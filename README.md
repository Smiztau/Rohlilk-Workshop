This project includes a setup.sh script that installs all necessary dependencies and a main.py file that runs the program with the option to configure it on the fly.

Setup Instructions

_Step 0: Get the data from kaggle_

Go to this website: https://www.kaggle.com/competitions/rohlik-orders-forecasting-challenge/data

Enter with user, click "Late Submit", and then click "Download All" button to download the data.

do as follows:

1.Unzip the rohlik-orders-forecasting-challenge dir.

2.In youre local repo/clone, make a dir called: csv

3.In csv make a dir called csv_input

4.Copy all content of rohlik-orders-forecasting-challenge to csv_input


_Step 1: Grant Execution Permission_

Before running the script, give it execution permission:

chmod +x setup.sh


_Step 2: Run the Installation Script_

Execute the script to install all required dependencies:

./setup.sh

This will:

Update the package lists.

Install Python and required system dependencies.

Install Python libraries like xgboost, pandas, numpy, scikit-learn, matplotlib, nltk, gensim, and tsfresh.

Download necessary NLTK dependencies.


_Step 3: Run main_
Once the dependencies are installed, run your Python program:

streamlit run main.py


Notes

-Ensure you have Python 3 installed on your system.

-If any dependency fails, manually install it using pip3 install <package>.


Troubleshooting
If you encounter permission issues, run:
sudo chmod +x setup.sh

If the script fails to run due to missing dependencies, try running it with sudo:
sudo ./setup.sh

For any issues, check error messages and install missing packages manually.
