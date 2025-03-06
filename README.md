This project includes a file.sh script that installs all necessary dependencies and a main.py file that runs the program with the option to configure it on the fly.

Setup Instructions

Step 1: Grant Execution Permission

    Before running the script, give it execution permission:

    chmod +x file.sh

Step 2: Run the Installation Script

    Execute the script to install all required dependencies:

    ./file.sh

    This will:

    Update the package lists.

    Install Python and required system dependencies.

    Install Python libraries like xgboost, pandas, numpy, scikit-learn, matplotlib, nltk, gensim, and tsfresh.

    Download necessary NLTK dependencies.

Step 3: Run main.py

    Once the dependencies are installed, run your Python program:

    python3 main.py

    If main.py has configurable options, you can pass them as arguments. For example:

    python3 main.py --config config.json

Notes

    -Ensure you have Python 3 installed on your system.

    -If any dependency fails, manually install it using pip3 install <package>.

Troubleshooting

    If you encounter permission issues, run:

    sudo chmod +x file.sh

    If the script fails to run due to missing dependencies, try running it with sudo:

    sudo ./file.sh

For any issues, check error messages and install missing packages manually.