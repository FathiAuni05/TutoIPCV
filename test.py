pip show opencv-python
pip install opencv-python
pip install opencv-contrib-python
source /home/adminuser/venv/bin/activate

rm -rf /home/adminuser/venv
python3.12 -m venv /home/adminuser/venv
source /home/adminuser/venv/bin/activate
pip install -r requirements.txt
