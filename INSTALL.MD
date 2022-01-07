conda create -n object_detection python=3.7
conda activate object_detection
conda install -y matplotlib scipy
pip3 install Cython
git clone https://github.com/philferriere/cocoapi.git
cd cocoapi && cd PythonAPI && python3 setup.py install